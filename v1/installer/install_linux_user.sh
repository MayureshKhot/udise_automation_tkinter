#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$V1_DIR/dist/UDISE_OCR_App"
INSTALL_ROOT="${HOME}/.local/opt/udise-ocr"
BIN_DIR="${HOME}/.local/bin"
APP_LINK="${BIN_DIR}/udise-ocr"
DESKTOP_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${DESKTOP_DIR}/udise-ocr.desktop"
TARGET_EXE="${INSTALL_ROOT}/UDISE_OCR_App"

if [[ ! -d "$DIST_DIR" ]]; then
  echo "Build output not found at: $DIST_DIR"
  echo "Run ./build_linux.sh first."
  exit 1
fi

mkdir -p "$INSTALL_ROOT" "$BIN_DIR" "$DESKTOP_DIR"
rm -rf "$INSTALL_ROOT"
cp -a "$DIST_DIR" "$INSTALL_ROOT"

ln -sfn "$TARGET_EXE" "$APP_LINK"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=UDISE OCR Tool
Comment=UDISE OCR Tkinter App
Exec=${TARGET_EXE}
Terminal=false
Type=Application
Categories=Utility;
EOF

chmod +x "$TARGET_EXE"
chmod +x "$APP_LINK"

echo "Installed for current user."
echo "Run from terminal: ${APP_LINK}"
echo "Or launch from app menu: UDISE OCR Tool"
