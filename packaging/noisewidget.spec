# Copyright (c) 2026, Creighton University
# SPDX-License-Identifier: BSD-3-Clause
#
# DISCLAIMER: This software is provided "AS IS", without warranty or guarantee
# of any kind, express or implied, including but not limited to merchantability,
# fitness for a particular purpose, accuracy, and non-infringement. It is NOT FIT
# FOR ANY PURPOSE, STATED OR IMPLIED. It is a simplified educational model, not
# intended for clinical, diagnostic, or regulatory use or for the operation of
# any real instrument; its results may be wrong and must be independently
# verified. In no event shall Creighton University or the contributors be liable
# for any claim, damages, or other liability arising from its use. See LICENSE.

# PyInstaller spec: a single self-contained executable, no installer.
#
# One-file mode is deliberate, as for MicroscopeVuilder: the audience should be able
# to download one file and double-click it on a lab machine where they may not have
# permission to run an installer or to `pip install` anything.
#
# Build with:  pyinstaller packaging/noisewidget.spec --noconfirm

import sys
from pathlib import Path

block_cipher = None

# Qt ships a great deal that a few painted widgets never touch.
excludes = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth", "PySide6.QtNetworkAuth", "PySide6.QtPositioning",
    "PySide6.QtRemoteObjects", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtWebSockets", "PySide6.QtWebChannel",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtUiTools", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
    "matplotlib", "scipy", "pandas", "PIL", "tkinter", "pytest", "setuptools",
    "IPython", "notebook", "sphinx",
]

a = Analysis(
    ["entry.py"],
    pathex=[str(Path(SPECPATH).parent)],
    binaries=[],
    # BSD-3-Clause requires binary redistributions to carry the notice and disclaimer.
    datas=[(str(Path(SPECPATH).parent / "LICENSE"), ".")],
    hiddenimports=["noisewidget.ui.widget"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="NoiseWidget",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    # Windowed on macOS and Windows; on Linux keep the console so a crash is readable.
    console=sys.platform.startswith("linux"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
