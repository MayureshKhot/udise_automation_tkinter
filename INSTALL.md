# UDISE OCR App - Simple Install Guide (Linux + Windows)

This guide explains how to install and run the app from scratch.

## 1) Get the project
```bash
git clone <your-repo-url>
cd "UDISE automation VS"
```

## 2) Linux users (Ubuntu/Debian)

### Install system requirements
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-tk
```

### Build the Linux app package
```bash
cd v1
./build_linux.sh 1.0.0
```

### Install for current user
```bash
./installer/install_linux_user.sh
```

### Launch
```bash
~/.local/bin/udise-ocr
```

You can also launch from the app menu as `UDISE OCR Tool`.

## 3) Windows users

### Install requirements
1. Install Python 3.11 or 3.12 (64-bit) and enable **Add Python to PATH**.
2. (Recommended) Install Inno Setup 6 if you want a setup installer `.exe`.

### Build app + installer
Open PowerShell in project folder:
```powershell
cd v1
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -AppVersion 1.0.0
```

### Output files
- Portable app folder: `v1\dist\UDISE_OCR_App\`
- Installer `.exe`: `v1\dist\installer\UDISE_OCR_Installer.exe` (if Inno Setup is installed)

### Install
Run `UDISE_OCR_Installer.exe` and follow the setup wizard.

## Notes
- Build Linux artifacts on Linux, and Windows artifacts on Windows.
- The packaged app already includes default model/ROI/sample files.
