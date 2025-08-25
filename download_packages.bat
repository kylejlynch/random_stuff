@echo off
REM Fraud Clustering Pipeline - Package Downloader
REM Downloads all required packages for offline installation

echo ========================================
echo Package Downloader for Offline Install
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.6+ first.
    pause
    exit /b 1
)

echo Step 1: Creating packages directory...
if not exist "packages" mkdir packages

echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo Step 3: Downloading packages (this may take several minutes)...
echo This will download approximately 113MB of packages.
echo.

REM Download all packages as wheels
pip download -r requirements-clean.txt -d packages/
if %errorlevel% neq 0 (
    echo ERROR: Failed to download some packages.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Step 4: Verifying downloads...
for /f %%i in ('dir /b packages\*.whl packages\*.tar.gz 2^>nul ^| find /c /v ""') do set file_count=%%i
echo Downloaded %file_count% package files.

if %file_count% LSS 20 (
    echo WARNING: Expected more package files. Some packages may be missing.
) else (
    echo SUCCESS: All packages downloaded successfully!
)

echo.
echo Step 5: Checking total size...
for /f "tokens=3" %%i in ('dir packages /s /-c 2^>nul ^| findstr /C:"bytes"') do set total_size=%%i
echo Total download size: %total_size% bytes

echo.
echo ========================================
echo Download Complete!
echo ========================================
echo.
echo The packages\ directory now contains all required files for offline installation.
echo You can now:
echo   1. Copy this entire folder to your work computer
echo   2. Run 'install_offline.bat' on the work computer
echo.
echo To create a zip archive for transfer:
echo   Right-click on the project folder and "Send to > Compressed folder"
echo.
pause