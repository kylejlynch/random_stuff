@echo off
REM Fraud Clustering Pipeline - Artifactory Installation Script
REM Install packages from corporate artifactory with compatible versions

echo ========================================
echo Fraud Clustering - Artifactory Install
echo ========================================
echo.
echo This script installs packages compatible with:
echo - scipy 1.5.4 (avoids coo_array import error)
echo - hdbscan 0.8.27 (last version compatible with scipy 1.5.4)
echo - numpy 1.19.5, pandas 1.1.5, scikit-learn 0.24.2
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

echo Step 1: Creating virtual environment...
set /p create_venv="Create virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv fraud_clustering_artifactory
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Continuing with system Python...
    ) else (
        echo Activating virtual environment...
        call fraud_clustering_artifactory\Scripts\activate.bat
        python -m pip install --upgrade pip
    )
)

echo.
echo Step 2: Installing packages from artifactory...
echo Using artifactory-compatible versions to avoid scipy.sparse.coo_array error
echo.

REM Install packages using artifactory requirements
pip install -r requirements-artifactory.txt
if %errorlevel% neq 0 (
    echo WARNING: Installation with requirements file failed.
    echo Trying individual package installation...
    echo.
    
    REM Install packages individually with exact versions
    echo Installing core packages...
    pip install pandas==1.1.5 numpy==1.19.5 scipy==1.5.4
    pip install scikit-learn==0.24.2 joblib
    
    echo Installing performance packages...
    pip install numba==0.53.1 llvmlite==0.36.0
    
    echo Installing clustering packages (scipy 1.5.4 compatible)...
    pip install hdbscan==0.8.27
    pip install umap-learn==0.5.8
    
    echo Installing visualization packages...
    pip install "plotly>=4.14.0,<5.0.0"
    pip install "matplotlib>=3.2.0,<3.4.0" 
    pip install "seaborn>=0.10.0,<0.12.0"
    
    echo Installing compatibility packages...
    pip install "importlib-metadata>=4.0.0" "typing-extensions>=4.0.0"
)

echo.
echo Step 3: Testing installation...
python -c "import pandas, numpy, scipy, sklearn, matplotlib, seaborn; print('Core packages imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Core packages installation failed.
    pause
    exit /b 1
)

REM Test clustering packages
python -c "
import hdbscan, umap
print('Clustering packages imported successfully!')
print('hdbscan available')
print('umap-learn version available')
" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Clustering packages may have issues.
    echo This could be due to binary compatibility problems.
) else (
    echo SUCCESS: All packages installed successfully!
)

echo.
echo Step 4: Testing scipy compatibility (no coo_array error)...
python -c "
import scipy.sparse
# Test that we're NOT trying to use coo_array (which doesn't exist in scipy 1.5.4)
print('scipy.sparse available - version compatible with artifactory')
if hasattr(scipy.sparse, 'coo_array'):
    print('WARNING: coo_array found - may cause issues with hdbscan 0.8.27')
else:
    print('GOOD: No coo_array (using scipy < 1.8) - compatible with hdbscan 0.8.27')
"
if %errorlevel% neq 0 (
    echo ERROR: scipy compatibility test failed.
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
)

echo.
echo ========================================
echo Artifactory Installation Complete!
echo ========================================
echo.
if exist "fraud_clustering_artifactory" (
    echo Virtual environment created: fraud_clustering_artifactory
    echo To activate: fraud_clustering_artifactory\Scripts\activate.bat
    echo.
)
echo Compatible package versions installed:
echo   scipy==1.5.4 (avoids coo_array error)  
echo   hdbscan==0.8.27 (compatible with scipy 1.5.4)
echo   numpy==1.19.5, pandas==1.1.5, scikit-learn==0.24.2
echo.
echo To test the pipeline:
echo   python example_dataframe_usage.py
echo   python fraud_clustering_pipeline.py --help
echo.
pause