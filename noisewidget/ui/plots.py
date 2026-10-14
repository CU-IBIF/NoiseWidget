"""Small QPainter plots: no plotting library, as in MicroscopeVuilder.

A frozen executable should not carry matplotlib to draw a few lines, and painting
directly keeps every redraw inside the lever's frame budget. :class:`PlotCanvas`
is one axis pair -- linear or log -- with lines, points, bars, reference lines and
a hover crosshair that reads every series at the pointer, so the value of each
curve is always one mouse move away rather than a number painted on every point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import style


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    color: str
    label: str = ""
    width: float = 2.0
    dashed: bool = False
    kind: str = "line"  # line, points, bars, step
    direct_label: bool = False  # write the label at the right-hand end
    hover: bool = True


@dataclass
class Marker:
    value: float
    color: str
    label: str = ""
    vertical: bool = True
    dashed: bool = True


@dataclass
class Band:
    lo: float
    hi: float
    color: str
    label: str = ""


@dataclass
class PlotSpec:
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlog: bool = False
    ylog: bool = False
    xrange: tuple[float, float] | None = None
    yrange: tuple[float, float] | None = None
    series: list[Series] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    bands: list[Band] = field(default_factory=list)
    legend: bool = True
    legend_left: bool = False  # put the legend top-left, clear of right-hand direct labels
    x_unit: str = ""
    y_unit: str = ""


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 1e-6 and len(ticks) < 20:
        ticks.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return ticks


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-2:
        exp = int(math.floor(math.log10(a)))
        mant = v / 10**exp
        return f"1e{exp}" if abs(mant - 1) < 1e-9 else f"{mant:g}e{exp}"
    return f"{v:g}"


class PlotCanvas(QtWidgets.QWidget):
    """One plot. Give it a :class:`PlotSpec`; it draws and handles hover."""

    MARGIN = (54, 14, 16, 36)  # left, right, top, bottom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spec = PlotSpec()
        self.setMinimumSize(240, 160)
        self.setMouseTracking(True)
        self._hover: QtCore.QPointF | None = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_spec(self, spec: PlotSpec) -> None:
        self.spec = spec
        self.update()

    # --- coordinates --------------------------------------------------------

    def _ranges(self) -> tuple[float, float, float, float]:
        s = self.spec
        xs = [v for ser in s.series for v in np.asarray(ser.x, float).ravel() if math.isfinite(v)]
        ys = [v for ser in s.series for v in np.asarray(ser.y, float).ravel() if math.isfinite(v)]
        if s.xlog:
            xs = [v for v in xs if v > 0]
        if s.ylog:
            ys = [v for v in ys if v > 0]
        x0, x1 = s.xrange if s.xrange else ((min(xs), max(xs)) if xs else (0.0, 1.0))
        y0, y1 = s.yrange if s.yrange else ((min(ys), max(ys)) if ys else (0.0, 1.0))
        if not s.yrange and not s.ylog:
            pad = (y1 - y0) * 0.06 or 1.0
            y0, y1 = min(y0, 0.0) if y0 >= 0 else y0 - pad, y1 + pad
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        return x0, x1, y0, y1

    def _plot_rect(self) -> QtCore.QRectF:
        l, r, t, b = self.MARGIN
        top = t + (16 if self.spec.title else 0)
        return QtCore.QRectF(l, top, max(self.width() - l - r, 10), max(self.height() - top - b, 10))

    def _tx(self, v, x0, x1, rect, log):
        if log:
            v, x0, x1 = (math.log10(max(a, 1e-300)) for a in (v, x0, x1))
        return rect.left() + (v - x0) / (x1 - x0) * rect.width()

    def _ty(self, v, y0, y1, rect, log):
        if log:
            v, y0, y1 = (math.log10(max(a, 1e-300)) for a in (v, y0, y1))
        return rect.bottom() - (v - y0) / (y1 - y0) * rect.height()

    def _inverse_x(self, px, x0, x1, rect, log):
        f = (px - rect.left()) / rect.width()
        if log:
            return 10 ** (math.log10(x0) + f * (math.log10(x1) - math.log10(x0)))
        return x0 + f * (x1 - x0)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(style.BACKGROUND))
        s = self.spec
        rect = self._plot_rect()
        x0, x1, y0, y1 = self._ranges()
        font = p.font()
        font.setPointSizeF(8.5)
        p.setFont(font)
        fm = QtGui.QFontMetricsF(font)

        if s.title:
            p.setPen(QtGui.QColor(style.TEXT))
            bold = QtGui.QFont(font)
            bold.setBold(True)
            p.setFont(bold)
            p.drawText(QtCore.QPointF(rect.left(), 13), s.title)
            p.setFont(font)

        # Shaded bands under everything.
        for band in s.bands:
            a = self._tx(max(band.lo, x0), x0, x1, rect, s.xlog)
            b = self._tx(min(band.hi, x1), x0, x1, rect, s.xlog)
            if b > a:
                c = QtGui.QColor(band.color)
                c.setAlpha(28)
                p.fillRect(QtCore.QRectF(a, rect.top(), b - a, rect.height()), c)
                if band.label:
                    p.setPen(QtGui.QColor(style.MUTED))
                    p.drawText(QtCore.QRectF(a + 3, rect.top() + 2, b - a - 6, 14),
                               QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, band.label)

        # Grid and ticks: recessive.
        grid = QtGui.QPen(QtGui.QColor(style.GRID), 1)
        p.setPen(grid)
        xt = self._log_ticks(x0, x1) if s.xlog else _nice_ticks(x0, x1)
        yt = self._log_ticks(y0, y1) if s.ylog else _nice_ticks(y0, y1)
        for t in xt:
            X = self._tx(t, x0, x1, rect, s.xlog)
            p.drawLine(QtCore.QPointF(X, rect.top()), QtCore.QPointF(X, rect.bottom()))
        for t in yt:
            Y = self._ty(t, y0, y1, rect, s.ylog)
            p.drawLine(QtCore.QPointF(rect.left(), Y), QtCore.QPointF(rect.right(), Y))
        p.setPen(QtGui.QColor(style.AXIS))
        p.drawRect(rect)
        for t in xt:
            X = self._tx(t, x0, x1, rect, s.xlog)
            label = _fmt_tick(t)
            p.drawText(QtCore.QPointF(X - fm.horizontalAdvance(label) / 2, rect.bottom() + 13), label)
        for t in yt:
            Y = self._ty(t, y0, y1, rect, s.ylog)
            label = _fmt_tick(t)
            p.drawText(QtCore.QPointF(rect.left() - 5 - fm.horizontalAdvance(label), Y + 4), label)
        p.setPen(QtGui.QColor(style.MUTED))
        if s.xlabel:
            p.drawText(QtCore.QRectF(rect.left(), rect.bottom() + 17, rect.width(), 16),
                       QtCore.Qt.AlignHCenter, s.xlabel)
        if s.ylabel:
            p.save()
            p.translate(12, rect.center().y())
            p.rotate(-90)
            p.drawText(QtCore.QRectF(-rect.height() / 2, -10, rect.height(), 14), QtCore.Qt.AlignHCenter, s.ylabel)
            p.restore()

        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        for m in s.markers:
            pen = QtGui.QPen(QtGui.QColor(m.color), 1.5)
            if m.dashed:
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            if m.vertical:
                X = self._tx(m.value, x0, x1, rect, s.xlog)
                p.drawLine(QtCore.QPointF(X, rect.top()), QtCore.QPointF(X, rect.bottom()))
            else:
                Y = self._ty(m.value, y0, y1, rect, s.ylog)
                p.drawLine(QtCore.QPointF(rect.left(), Y), QtCore.QPointF(rect.right(), Y))
        for ser in s.series:
            self._draw_series(p, ser, rect, x0, x1, y0, y1)
        p.restore()

        # Marker labels outside the clip so they are never cut off.
        p.setPen(QtGui.QColor(style.MUTED))
        for m in s.markers:
            if not m.label:
                continue
            if m.vertical:
                X = self._tx(m.value, x0, x1, rect, s.xlog)
                if rect.left() <= X <= rect.right():
                    w = fm.horizontalAdvance(m.label)
                    tx = X + 3 if X + 3 + w < rect.right() else X - 3 - w
                    p.setPen(QtGui.QColor(m.color))
                    p.drawText(QtCore.QPointF(tx, rect.bottom() - 4), m.label)
            else:
                Y = self._ty(m.value, y0, y1, rect, s.ylog)
                if rect.top() <= Y <= rect.bottom():
                    p.setPen(QtGui.QColor(m.color))
                    p.drawText(QtCore.QPointF(rect.right() - fm.horizontalAdvance(m.label) - 4, Y - 3), m.label)

        # Direct labels at the right-hand end of a line.
        for ser in s.series:
            if not (ser.direct_label and ser.label) or ser.kind != "line":
                continue
            x = np.asarray(ser.x, float)
            y = np.asarray(ser.y, float)
            ok = np.isfinite(x) & np.isfinite(y) & ((y > 0) if s.ylog else True)
            ok &= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            if not ok.any():
                continue
            i = np.nonzero(ok)[0][-1]
            X = self._tx(x[i], x0, x1, rect, s.xlog)
            Y = self._ty(y[i], y0, y1, rect, s.ylog)
            w = fm.horizontalAdvance(ser.label)
            p.setPen(QtGui.QColor(style.TEXT))
            p.drawText(QtCore.QPointF(min(X, rect.right()) - w - 4, Y - 4), ser.label)

        if s.legend:
            self._draw_legend(p, rect, fm)
        if self._hover is not None and rect.contains(self._hover):
            self._draw_hover(p, rect, fm, x0, x1, y0, y1)
        p.end()

    def _log_ticks(self, lo: float, hi: float) -> list[float]:
        lo, hi = max(lo, 1e-300), max(hi, 1e-300)
        a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        step = max(1, (b - a) // 6)
        return [10.0**e for e in range(a, b + 1, step) if lo * 0.999 <= 10.0**e <= hi * 1.001]

    def _draw_series(self, p, ser: Series, rect, x0, x1, y0, y1) -> None:
        s = self.spec
        x = np.asarray(ser.x, float)
        y = np.asarray(ser.y, float)
        color = QtGui.QColor(ser.color)
        if ser.kind == "bars":
            if x.size == 0:
                return
            width = (x[1] - x[0]) if x.size > 1 else 1.0
            base = self._ty(max(y0, 0.0) if not s.ylog else y0, y0, y1, rect, s.ylog)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(color)
            for xi, yi in zip(x, y):
                if yi <= 0:
                    continue
                a = self._tx(xi - width / 2, x0, x1, rect, s.xlog)
                b = self._tx(xi + width / 2, x0, x1, rect, s.xlog)
                top = self._ty(yi, y0, y1, rect, s.ylog)
                # A 2 px surface gap between adjacent bars.
                p.drawRoundedRect(QtCore.QRectF(a + 1, top, max(b - a - 2, 1), base - top), 2, 2)
            p.setBrush(QtCore.Qt.NoBrush)
            return
        ok = np.isfinite(x) & np.isfinite(y)
        if s.xlog:
            ok &= x > 0
        if s.ylog:
            ok &= y > 0
        pts = [QtCore.QPointF(self._tx(a, x0, x1, rect, s.xlog), self._ty(b, y0, y1, rect, s.ylog))
               for a, b in zip(x[ok], y[ok])]
        if not pts:
            return
        if ser.kind == "points":
            p.setPen(QtGui.QPen(QtGui.QColor(style.BACKGROUND), 1.5))  # surface ring
            p.setBrush(color)
            r = max(ser.width, 3.0)
            for pt in pts:
                p.drawEllipse(pt, r, r)
            p.setBrush(QtCore.Qt.NoBrush)
            return
        pen = QtGui.QPen(color, ser.width)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        if ser.dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        p.setPen(pen)
        if ser.kind == "step" and len(pts) > 1:
            poly = QtGui.QPolygonF()
            for i, pt in enumerate(pts):
                if i:
                    poly.append(QtCore.QPointF((pts[i - 1].x() + pt.x()) / 2, pts[i - 1].y()))
                    poly.append(QtCore.QPointF((pts[i - 1].x() + pt.x()) / 2, pt.y()))
                poly.append(pt)
            p.drawPolyline(poly)
        else:
            p.drawPolyline(QtGui.QPolygonF(pts))

    def _draw_legend(self, p, rect, fm) -> None:
        items = [s for s in self.spec.series if s.label]
        if len(items) < 2:
            return
        row_h = fm.height() + 2
        width = max(fm.horizontalAdvance(s.label) for s in items) + 30
        left = rect.left() + 6 if self.spec.legend_left else rect.right() - width - 6
        # Clear of a band's label, which sits along the top edge.
        top = rect.top() + (20 if any(b.label for b in self.spec.bands) else 6)
        box = QtCore.QRectF(left, top, width, row_h * len(items) + 6)
        bg = QtGui.QColor(style.PANEL)
        bg.setAlpha(225)
        p.setPen(QtGui.QPen(QtGui.QColor(style.GRID), 1))
        p.setBrush(bg)
        p.drawRoundedRect(box, 3, 3)
        p.setBrush(QtCore.Qt.NoBrush)
        for i, ser in enumerate(items):
            y = box.top() + 4 + row_h * i + row_h / 2
            pen = QtGui.QPen(QtGui.QColor(ser.color), 2.5)
            if ser.dashed:
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            if ser.kind in ("points",):
                p.setBrush(QtGui.QColor(ser.color))
                p.drawEllipse(QtCore.QPointF(box.left() + 13, y), 3.5, 3.5)
                p.setBrush(QtCore.Qt.NoBrush)
            elif ser.kind == "bars":
                p.fillRect(QtCore.QRectF(box.left() + 7, y - 4, 12, 8), QtGui.QColor(ser.color))
            else:
                p.drawLine(QtCore.QPointF(box.left() + 6, y), QtCore.QPointF(box.left() + 20, y))
            p.setPen(QtGui.QColor(style.TEXT))
            p.drawText(QtCore.QPointF(box.left() + 25, y + fm.ascent() / 2 - 1), ser.label)

    def _draw_hover(self, p, rect, fm, x0, x1, y0, y1) -> None:
        s = self.spec
        xv = self._inverse_x(self._hover.x(), x0, x1, rect, s.xlog)
        X = self._hover.x()
        p.setPen(QtGui.QPen(QtGui.QColor(style.AXIS), 1, QtCore.Qt.DotLine))
        p.drawLine(QtCore.QPointF(X, rect.top()), QtCore.QPointF(X, rect.bottom()))
        lines = [f"{s.xlabel.split(' (')[0] or 'x'}: {xv:.4g}{(' ' + s.x_unit) if s.x_unit else ''}"]
        for ser in s.series:
            if not ser.hover or not ser.label or ser.kind == "bars":
                continue
            x = np.asarray(ser.x, float)
            y = np.asarray(ser.y, float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 1:
                continue
            xs, ys = x[ok], y[ok]
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            if xv < xs[0] or xv > xs[-1]:
                continue
            if s.xlog and (xs > 0).all():
                yv = float(np.interp(math.log10(xv), np.log10(xs), ys))
            else:
                yv = float(np.interp(xv, xs, ys))
            lines.append((ser.color, f"{ser.label}: {yv:.4g}{(' ' + s.y_unit) if s.y_unit else ''}"))
            Y = self._ty(yv, y0, y1, rect, s.ylog)
            if rect.top() <= Y <= rect.bottom():
                p.setPen(QtGui.QPen(QtGui.QColor(style.BACKGROUND), 2))
                p.setBrush(QtGui.QColor(ser.color))
                p.drawEllipse(QtCore.QPointF(X, Y), 4, 4)
                p.setBrush(QtCore.Qt.NoBrush)
        row_h = fm.height() + 1
        texts = [ln if isinstance(ln, str) else ln[1] for ln in lines]
        w = max(fm.horizontalAdvance(t) for t in texts) + 24
        h = row_h * len(lines) + 8
        bx = X + 12 if X + 12 + w < rect.right() else X - 12 - w
        by = rect.top() + 8
        bg = QtGui.QColor(style.PANEL)
        bg.setAlpha(240)
        p.setPen(QtGui.QPen(QtGui.QColor(style.GRID), 1))
        p.setBrush(bg)
        p.drawRoundedRect(QtCore.QRectF(bx, by, w, h), 3, 3)
        p.setBrush(QtCore.Qt.NoBrush)
        for i, ln in enumerate(lines):
            y = by + 4 + row_h * (i + 1) - 3
            if isinstance(ln, str):
                p.setPen(QtGui.QColor(style.MUTED))
                p.drawText(QtCore.QPointF(bx + 6, y), ln)
            else:
                color, text = ln
                p.fillRect(QtCore.QRectF(bx + 6, y - 7, 8, 3), QtGui.QColor(color))
                p.setPen(QtGui.QColor(style.TEXT))
                p.drawText(QtCore.QPointF(bx + 18, y), text)

    # --- hover --------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position()
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()


class BudgetBar(QtWidgets.QWidget):
    """Where the variance goes at the operating point: one stacked horizontal bar.

    Variance, not noise, because variances add -- so the segment widths are the
    true shares, which standard deviations are not.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(76)
        self.fractions: dict[str, float] = {}
        self.labels: dict[str, str] = {}
        self.caption = ""

    def set_budget(self, fractions: dict[str, float], labels: dict[str, str], caption: str) -> None:
        self.fractions = fractions
        self.labels = labels
        self.caption = caption
        self.setToolTip("\n".join(f"{labels[k]}: {v:.1%} of the variance" for k, v in fractions.items() if v > 0))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(style.BACKGROUND))
        font = p.font()
        font.setPointSizeF(8.5)
        p.setFont(font)
        fm = QtGui.QFontMetricsF(font)
        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(QtCore.QPointF(8, 13), self.caption)
        bar = QtCore.QRectF(8, 18, self.width() - 16, 14)
        x = bar.left()
        legend_x = 8.0
        legend_y = bar.bottom() + 15
        for key, frac in self.fractions.items():
            if frac <= 0:
                continue
            w = frac * bar.width()
            color = QtGui.QColor(style.TERM_COLORS.get(key, style.AXIS))
            if w >= 1:
                p.fillRect(QtCore.QRectF(x, bar.top(), max(w - 2, 1), bar.height()), color)
            x += w
            if frac >= 0.005:
                text = f"{self.labels.get(key, key)} {frac:.0%}"
                tw = fm.horizontalAdvance(text) + 22
                if legend_x + tw > self.width():
                    legend_x = 8.0
                    legend_y += fm.height() + 2
                p.fillRect(QtCore.QRectF(legend_x, legend_y - 8, 9, 9), color)
                p.setPen(QtGui.QColor(style.TEXT))
                p.drawText(QtCore.QPointF(legend_x + 13, legend_y), text)
                legend_x += tw
        p.end()
