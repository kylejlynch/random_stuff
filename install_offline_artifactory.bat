@echo off
REM Fraud Clustering Pipeline - Offline Artifactory Installation Script
REM Install from downloaded artifactory-compatible wheel files

echo ========================================
echo Fraud Clustering - Offline Artifactory Install
echo ========================================
echo.
echo Installing from artifactory-compatible packages:
echo   scipy 1.5.4 (avoids coo_array import error)
echo   hdbscan 0.8.27 (compatible with scipy 1.5.4)
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
if not exist "packages_artifactory" (
    echo ERROR: packages_artifactory directory not found.
    echo Please ensure the packages_artifactory folder is in the current directory.
    echo You should have downloaded artifactory-compatible wheel files first.
    pause
    exit /b 1
)

echo Found packages_artifactory directory with offline wheels.
echo.

echo Step 1: Creating virtual environment (recommended)...
set /p create_venv="Create virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv fraud_clustering_artifactory_offline
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Continuing with system Python...
    ) else (
        echo Activating virtual environment...
        call fraud_clustering_artifactory_offline\Scripts\activate.bat
        python -m pip install --upgrade pip
    )
)

echo.
echo Step 2: Installing packages from offline artifactory wheels...
echo This may take several minutes...

REM Install packages from local directory using artifactory requirements
pip install --find-links packages_artifactory\ --no-index --no-deps -r requirements-artifactory.txt
if %errorlevel% neq 0 (
    echo WARNING: Installation with requirements file failed.
    echo Trying alternative installation method...
    
    REM Try installing all wheels directly from artifactory packages
    for %%f in (packages_artifactory\*.whl) do (
        echo Installing %%f...
        pip install --find-links packages_artifactory\ --no-index "%%f"
    )
    
    REM Handle source distributions
    for %%f in (packages_artifactory\*.tar.gz) do (
        echo Installing %%f...
        pip install --find-links packages_artifactory\ --no-index "%%f"
    )
)

echo.
echo Step 3: Testing core installation...
python -c "import pandas, numpy, scipy, sklearn, matplotlib, seaborn; print('Core packages imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Core packages installation failed.
    pause
    exit /b 1
)

REM Test clustering packages
python -c "import hdbscan, umap; print('Clustering packages imported successfully!')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Clustering packages (hdbscan/umap) may not be installed correctly.
    echo You can still use the pipeline with reduced clustering functionality.
) else (
    echo SUCCESS: All artifactory packages installed successfully!
)

echo.
echo Step 4: Testing scipy compatibility (coo_array issue check)...
python -c "
import scipy.sparse
import scipy
print('scipy version:', scipy.__version__)
if hasattr(scipy.sparse, 'coo_array'):
    print('WARNING: coo_array found - this may cause hdbscan import issues')
else:
    print('GOOD: No coo_array found - compatible with hdbscan 0.8.27')
    
# Test hdbscan import specifically
try:
    import hdbscan
    print('SUCCESS: hdbscan imports without coo_array error!')
except ImportError as e:
    if 'coo_array' in str(e):
        print('ERROR: Still getting coo_array import error:', e)
    else:
        print('ERROR: Other hdbscan import error:', e)
"
if %errorlevel% neq 0 (
    echo ERROR: Compatibility test failed.
)

echo.
echo Step 5: Testing fraud clustering pipeline...
if exist "dataframe_pipeline.py" (
    python -c "from dataframe_pipeline import DataFrameFraudClusteringPipeline; print('Pipeline modules imported successfully!')"
    if %errorlevel% neq 0 (
        echo WARNING: Pipeline modules not found or have issues.
    ) else (
        echo SUCCESS: Pipeline is ready to use with artifactory packages!
    )
) else (
    echo WARNING: Pipeline modules not found in current directory.
    echo Make sure you're running this script from the project directory.
)

echo.
echo ========================================
echo Offline Artifactory Installation Complete!
echo ========================================
echo.
if exist "fraud_clustering_artifactory_offline" (
    echo Virtual environment created: fraud_clustering_artifactory_offline
    echo To activate: fraud_clustering_artifactory_offline\Scripts\activate.bat
    echo.
)
echo Installed artifactory-compatible versions:
echo   scipy 1.5.4 (no coo_array - avoids import errors)
echo   hdbscan 0.8.27 (last version compatible with scipy 1.5.4)  
echo   numpy 1.19.5, pandas 1.1.5, scikit-learn 0.24.2
echo.
echo To test the pipeline:
echo   python example_dataframe_usage.py
echo   python fraud_clustering_pipeline.py --help
echo.
pause