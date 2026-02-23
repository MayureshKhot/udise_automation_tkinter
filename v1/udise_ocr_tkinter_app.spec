# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

try:
    project_root = Path(SPEC).resolve().parent
except Exception:
    project_root = Path.cwd()

datas = [
    (str(project_root / "models"), "models"),
    (str(project_root / "udise_roi.json"), "."),
    (str(project_root / "test_udise.xlsx"), "."),
]

hiddenimports = [
    "tensorflow",
    "keras",
    "PIL._tkinter_finder",
    "cv2",
    "openpyxl",
    "requests",
    "numpy",
]

a = Analysis(
    ["udise_ocr_tkinter_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UDISE_OCR_App",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="UDISE_OCR_App",
)
