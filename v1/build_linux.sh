#!/usr/bin/env bash
set -euo pipefail

APP_VERSION="${1:-1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv-linux-build"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install -r requirements.txt
"$PIP_BIN" install tensorflow pyinstaller pyinstaller-hooks-contrib

rm -rf build dist
"$PYTHON_BIN" -m PyInstaller --clean --noconfirm "./udise_ocr_tkinter_app.spec"

mkdir -p dist/installer
TARBALL="dist/installer/UDISE_OCR_Linux_${APP_VERSION}_x86_64.tar.gz"
tar -C dist -czf "$TARBALL" UDISE_OCR_App

echo
echo "Linux build complete."
echo "Portable app folder: ./dist/UDISE_OCR_App"
echo "Portable tar.gz: $TARBALL"
echo
echo "Optional user install:"
echo "  ./installer/install_linux_user.sh"
