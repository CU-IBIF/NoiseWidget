"""Image panes: the object, the optical image, the recording, and the noise.

Pixels are drawn as blocks (nearest-neighbour), never smoothed. Sampling is one of
the levers, and an interpolated display would hide exactly what changing the
magnification or binning does to the picture.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..physics.fluorophores import wavelength_to_rgb
from . import style

LUTS = ("grey", "emission colour", "fire", "inverted")


def _lut_table(name: str, wavelength_nm: float) -> np.ndarray:
    """256 x 3 uint8 colour table."""
    t = np.linspace(0.0, 1.0, 256)
    if name == "emission colour":
        r, g, b = wavelength_to_rgb(wavelength_nm)
        rgb = np.stack([t * r, t * g, t * b], axis=1)
    elif name == "fire":
        rgb = np.stack([np.clip(t * 3, 0, 1), np.clip(t * 3 - 1, 0, 1), np.clip(t * 3 - 2, 0, 1)], axis=1)
    elif name == "inverted":
        rgb = np.stack([1 - t] * 3, axis=1)
    else:
        rgb = np.stack([t] * 3, axis=1)
    return (rgb * 255).astype(np.uint8)


def to_qimage(
    image: np.ndarray, lo: float, hi: float, lut: str = "grey", wavelength_nm: float = 520.0,
    saturated: np.ndarray | None = None, diverging: bool = False,
) -> QtGui.QImage:
    arr = np.asarray(image, dtype=float)
    span = hi - lo
    norm = np.clip((arr - lo) / span, 0.0, 1.0) if span > 0 else np.zeros_like(arr)
    idx = (norm * 255).astype(np.uint8)
    if diverging:
        # Noise is signed: blue below the expectation, orange above, grey at zero.
        t = norm[..., None]
        neg = np.array([0x39, 0x87, 0xE5]) / 255.0
        pos = np.array([0xD9, 0x59, 0x26]) / 255.0
        mid = np.array([0.16, 0.17, 0.19])
        rgb = np.where(t < 0.5, mid + (neg - mid) * (0.5 - t) * 2, mid + (pos - mid) * (t - 0.5) * 2)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    else:
        rgb = _lut_table(lut, wavelength_nm)[idx]
    if saturated is not None and saturated.any():
        rgb = rgb.copy()
        rgb[saturated] = (217, 95, 95)
    h, w = arr.shape
    buf = np.ascontiguousarray(rgb)
    return QtGui.QImage(buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class ImagePane(QtWidgets.QWidget):
    """One image, a title, a caption, a 1 um scale bar, optional overlays."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.caption = ""
        self.image: QtGui.QImage | None = None
        self.field_um = 1.0  # x extent
        self.field_y_um = 1.0
        self.origin_um = (0.0, 0.0)  # specimen coordinates of the image's top-left corner
        self.window_um: QtCore.QRectF | None = None  # measured region, in um
        self.region_um: QtCore.QRectF | None = None  # e.g. the area a scanner covers
        self.message = "--"
        self.setMinimumSize(170, 190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def show_image(self, qimage: QtGui.QImage, field_um: float, caption: str = "",
                   field_y_um: float | None = None, origin_um: tuple[float, float] = (0.0, 0.0)) -> None:
        """Show an image covering ``field_um`` x ``field_y_um`` of the specimen.

        Pixels need not be square -- a scanner can sample x and y differently --
        so the image is drawn at its physical aspect, not its pixel aspect.
        """
        self.image = qimage
        self.field_um = field_um
        self.field_y_um = field_um if field_y_um is None else field_y_um
        self.origin_um = origin_um
        self.caption = caption
        self.update()

    def set_window(self, rect_um: QtCore.QRectF | None) -> None:
        """The measured region, in specimen coordinates (um)."""
        self.window_um = rect_um
        self.update()

    def image_rect(self) -> QtCore.QRectF:
        top, bottom = 18, 18
        box_w = max(self.width() - 4, 10)
        box_h = max(self.height() - top - bottom, 10)
        aspect = self.field_y_um / self.field_um if self.field_um > 0 else 1.0
        w = min(box_w, box_h / aspect)
        h = w * aspect
        return QtCore.QRectF((self.width() - w) / 2, top, w, h)

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(style.BACKGROUND))
        font = p.font()
        font.setPointSizeF(9)
        p.setFont(font)
        bold = QtGui.QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QtGui.QColor(style.TEXT))
        p.drawText(QtCore.QRectF(4, 0, self.width() - 8, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.title)
        p.setFont(font)
        rect = self.image_rect()
        if self.image is None:
            p.setPen(QtGui.QColor(style.AXIS))
            p.drawText(rect, QtCore.Qt.AlignCenter, self.message)
            p.end()
            return
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        p.drawImage(rect, self.image)

        scale = rect.width() / self.field_um  # screen px per um
        if self.region_um is not None:
            r = self.region_um.translated(-self.origin_um[0], -self.origin_um[1])
            p.setPen(QtGui.QPen(QtGui.QColor("#d9a441"), 1.5))
            p.drawRect(QtCore.QRectF(rect.left() + r.left() * scale, rect.top() + r.top() * scale,
                                     r.width() * scale, r.height() * scale))
        if self.window_um is not None:
            pen = QtGui.QPen(QtGui.QColor("#e6edf3"), 1, QtCore.Qt.DashLine)
            p.setPen(pen)
            w = self.window_um.translated(-self.origin_um[0], -self.origin_um[1])
            p.drawRect(QtCore.QRectF(rect.left() + w.left() * scale, rect.top() + w.top() * scale,
                                     w.width() * scale, w.height() * scale))

        # Scale bar: 1 um, or 5 um when 1 um would be a few screen pixels.
        bar_um = 1.0 if scale * 1.0 >= 25 else 5.0
        x1 = rect.right() - 8
        x0 = x1 - bar_um * scale
        y = rect.bottom() - 8
        p.fillRect(QtCore.QRectF(x0 - 1, y - 2, x1 - x0 + 2, 6), QtGui.QColor(0, 0, 0, 160))
        p.fillRect(QtCore.QRectF(x0, y - 1, x1 - x0, 4), QtGui.QColor("#ffffff"))
        p.setPen(QtGui.QColor("#ffffff"))
        p.drawText(QtCore.QRectF(x0 - 30, y - 18, x1 - x0 + 30, 14), QtCore.Qt.AlignRight, f"{bar_um:g} um")

        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(QtCore.QRectF(4, rect.bottom() + 1, self.width() - 8, 16),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.caption)
        p.end()


class ImageGrid(QtWidgets.QWidget):
    """The four panes side by side: object -> optics -> camera -> noise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.object = ImagePane("object (ground truth)")
        self.optical = ImagePane("optical image (noise-free)")
        self.recorded = ImagePane("recorded image")
        self.noise = ImagePane("noise (recorded - expected)")
        for i, pane in enumerate((self.object, self.optical, self.recorded, self.noise)):
            layout.addWidget(pane, 0, i)
            layout.setColumnStretch(i, 1)

    def panes(self) -> list[ImagePane]:
        return [self.object, self.optical, self.recorded, self.noise]
