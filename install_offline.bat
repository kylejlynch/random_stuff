@echo off
REM Fraud Clustering Pipeline - Offline Installation Script
REM Install packages from downloaded wheel files

echo ========================================
echo Fraud Clustering - Offline Installation
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

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3,6) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.6+ is required.
    python --version
    pause
    exit /b 1
)

REM Check if packages directory exists
if not exist "packages" (
    echo ERROR: packages directory not found.
    echo Please ensure the packages folder is in the current directory.
    echo You should have downloaded and extracted the wheel files.
    pause
    exit /b 1
)

echo Found packages directory with offline wheels.
echo.

echo Step 1: Creating virtual environment (recommended)...
set /p create_venv="Create virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv fraud_clustering_offline
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Continuing with system Python...
    ) else (
        echo Activating virtual environment...
        call fraud_clustering_offline\Scripts\activate.bat
        python -m pip install --upgrade pip
    )
)

echo.
echo Step 2: Installing packages from offline wheels...
echo This may take several minutes...

REM Install packages from local directory
pip install --find-links packages\ --no-index --no-deps -r requirements-clean.txt
if %errorlevel% neq 0 (
    echo WARNING: Installation with requirements file failed.
    echo Trying alternative installation method...
    
    REM Try installing all wheels directly
    for %%f in (packages\*.whl) do (
        echo Installing %%f...
        pip install --find-links packages\ --no-index "%%f"
    )
    
    REM Handle source distributions (like hdbscan)
    for %%f in (packages\*.tar.gz) do (
        echo Installing %%f...
        pip install --find-links packages\ --no-index "%%f"
    )
)

echo.
echo Step 3: Testing installation...
python -c "import pandas, numpy, sklearn, matplotlib, plotly, seaborn; print('Core packages imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Core packages installation failed.
    pause
    exit /b 1
)

REM Test optional packages
python -c "import hdbscan, umap; print('Clustering packages imported successfully!')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Clustering packages (hdbscan/umap) may not be installed correctly.
    echo You can still use the pipeline with reduced clustering functionality.
) else (
    echo SUCCESS: All packages installed successfully!
)

echo.
echo Step 4: Testing fraud clustering pipeline...
if exist "dataframe_pipeline.py" (
    python -c "from dataframe_pipeline import DataFrameFraudClusteringPipeline; print('Pipeline modules imported successfully!')"
    if %errorlevel% neq 0 (
        echo WARNING: Pipeline modules not found or have issues.
    ) else (
        echo SUCCESS: Pipeline is ready to use!
    )
) else (
    echo WARNING: Pipeline modules not found in current directory.
    echo Make sure you're running this script from the project directory.
)

echo.
echo ========================================
echo Offline Installation Complete!
echo ========================================
echo.
if exist "fraud_clustering_offline" (
    echo Virtual environment created: fraud_clustering_offline
    echo To activate: fraud_clustering_offline\Scripts\activate.bat
    echo.
)
echo To test the pipeline:
echo   python example_dataframe_usage.py
echo   python fraud_clustering_pipeline.py --help
echo.
pause