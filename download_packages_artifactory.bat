@echo off
REM Fraud Clustering Pipeline - Artifactory Compatible Package Downloader
REM Downloads artifactory-compatible packages for offline installation

echo ========================================
echo Artifactory Package Downloader
echo ========================================
echo.
echo This downloads packages compatible with corporate artifactory:
echo   scipy==1.5.4 (avoids coo_array import error)
echo   hdbscan==0.8.27 (last version compatible with scipy 1.5.4)
echo   numpy==1.19.5, pandas==1.1.5, etc.
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
if not exist "packages_artifactory" mkdir packages_artifactory

echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo Step 3: Downloading artifactory-compatible packages...
echo This will download packages that work with your corporate environment.
echo.

REM Download packages using artifactory requirements
pip download -r requirements-artifactory.txt -d packages_artifactory/
if %errorlevel% neq 0 (
    echo ERROR: Failed to download some packages.
    echo Trying individual downloads...
    
    REM Download core packages individually
    pip download pandas==1.1.5 numpy==1.19.5 scipy==1.5.4 -d packages_artifactory/
    pip download scikit-learn==0.24.2 joblib -d packages_artifactory/
    pip download numba==0.53.1 llvmlite==0.36.0 -d packages_artifactory/
    pip download hdbscan==0.8.27 umap-learn==0.5.8 -d packages_artifactory/
    pip download "plotly>=4.14.0,<5.0.0" "matplotlib>=3.2.0,<3.4.0" "seaborn>=0.10.0,<0.12.0" -d packages_artifactory/
)

echo.
echo Step 4: Verifying downloads...
for /f %%i in ('dir /b packages_artifactory\*.whl packages_artifactory\*.tar.gz 2^>nul ^| find /c /v ""') do set file_count=%%i
echo Downloaded %file_count% package files to packages_artifactory/

if %file_count% LSS 15 (
    echo WARNING: Expected more package files. Some packages may be missing.
) else (
    echo SUCCESS: Artifactory-compatible packages downloaded!
)

echo.
echo Step 5: Checking total size...
for /f "tokens=3" %%i in ('dir packages_artifactory /s /-c 2^>nul ^| findstr /C:"bytes"') do set total_size=%%i
echo Total download size: %total_size% bytes

echo.
echo ========================================
echo Artifactory Download Complete!
echo ========================================
echo.
echo The packages_artifactory\ directory now contains artifactory-compatible packages:
echo   scipy 1.5.4 (no coo_array - compatible with hdbscan 0.8.27)
echo   hdbscan 0.8.27 (avoids scipy compatibility issues)  
echo   All other packages matched to artifactory versions
echo.
echo You can now:
echo   1. Copy this folder to your work computer
echo   2. Run 'install_offline_artifactory.bat' on the work computer
echo.
echo To create a zip archive for transfer:
echo   Right-click on the project folder and "Send to > Compressed folder"
echo.
pause