param(
    [string]$PythonExe = "python",
    [string]$AppVersion = "1.0.0"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPath = Join-Path $ScriptDir ".venv-win-build"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    & $PythonExe -m venv $VenvPath
}

& $VenvPip install --upgrade pip
& $VenvPip install -r requirements.txt
& $VenvPip install tensorflow pyinstaller pyinstaller-hooks-contrib

if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
if (Test-Path ".\dist") { Remove-Item ".\dist" -Recurse -Force }

& $VenvPython -m PyInstaller --clean --noconfirm ".\udise_ocr_tkinter_app.spec"

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    Write-Host "PyInstaller build done."
    Write-Host "Install Inno Setup and ensure 'iscc' is in PATH to create setup installer."
    exit 0
}

& $iscc.Source "/DMyAppVersion=$AppVersion" ".\installer\UDISE_OCR_App.iss"

Write-Host ""
Write-Host "Build complete."
Write-Host "Portable app folder: .\dist\UDISE_OCR_App"
Write-Host "Installer: .\dist\installer\UDISE_OCR_Installer.exe"
