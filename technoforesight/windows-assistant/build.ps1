# Builds the Windows app and installer. Used by GitHub Actions; also works on any Windows PC
# with Python 3.12 and Inno Setup 6 installed.
#   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$version = (python -c "import tf_assistant; print(tf_assistant.__version__)").Trim()
Write-Host "Building TechnoForesight Assistant $version"

python -m pip install --upgrade pip
python -m pip install pyinstaller pypdf pytest

python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

python -m PyInstaller --noconfirm --clean --windowed `
  --name TechnoForesightAssistant `
  --icon assets\icon.ico `
  --add-data "assets;assets" `
  --hidden-import pypdf `
  launcher.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Smoke test: the frozen exe must start and exit cleanly
$p = Start-Process -FilePath .\dist\TechnoForesightAssistant\TechnoForesightAssistant.exe -ArgumentList "--version" -Wait -PassThru
if ($p.ExitCode -ne 0) { throw "Frozen app failed to start (exit $($p.ExitCode))" }
Write-Host "Frozen app starts OK"

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { choco install innosetup -y --no-progress; }
& $iscc "/DAppVersion=$version" installer\TechnoForesightAssistant.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

Compress-Archive -Force -Path dist\TechnoForesightAssistant\* -DestinationPath "dist\TechnoForesightAssistant-Portable-$version.zip"
Get-ChildItem dist -File | Format-Table Name, Length
