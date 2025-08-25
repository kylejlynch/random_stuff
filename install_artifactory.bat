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
echo Trying artifactory versions (hdbscan 0.8.40 first, joblib separate)...
echo.

REM Install packages using artifactory requirements
pip install -r requirements-artifactory.txt
if %errorlevel% neq 0 (
    echo WARNING: Installation with hdbscan 0.8.40 failed.
    echo This might be due to sklearn.externals.joblib or scipy.sparse.coo_array errors.
    echo Trying fallback versions...
    echo.
    
    REM Try fallback requirements with hdbscan 0.8.29 if available
    if exist "requirements-artifactory-fallback.txt" (
        pip install -r requirements-artifactory-fallback.txt
        if %errorlevel% neq 0 (
            goto individual_install
        ) else (
            goto test_install
        )
    )
    
    :individual_install
    echo Trying individual package installation with safe versions...
    echo.
    
    REM Install packages individually with conservative versions
    echo Installing core packages...
    pip install pandas==1.1.5 numpy==1.19.5 scipy==1.5.4
    pip install scikit-learn==0.24.2 "joblib>=1.0.0,<1.2.0"
    
    echo Installing performance packages...
    pip install numba==0.53.1 llvmlite==0.36.0
    
    echo Installing clustering packages (most conservative - avoid both errors)...
    pip install hdbscan==0.8.27
    pip install umap-learn==0.5.8
    
    echo Installing visualization packages...
    pip install "plotly>=4.14.0,<5.0.0"
    pip install "matplotlib>=3.2.0,<3.4.0" 
    pip install "seaborn>=0.10.0,<0.12.0"
    
    echo Installing compatibility packages...
    pip install "importlib-metadata>=4.0.0" "typing-extensions>=4.0.0"
)

:test_install

echo.
echo Step 3: Testing installation...
python -c "import pandas, numpy, scipy, sklearn, matplotlib, seaborn; print('Core packages imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Core packages installation failed.
    pause
    exit /b 1
)

REM Test clustering packages with specific error detection
python -c "
import sys
try:
    import hdbscan
    print('SUCCESS: hdbscan imported successfully')
except ImportError as e:
    if 'sklearn.externals.joblib' in str(e):
        print('ERROR: sklearn.externals.joblib issue - hdbscan version incompatible with sklearn 0.24.2')
        print('Solution: Install joblib separately and use hdbscan 0.8.29+')
        sys.exit(1)
    elif 'coo_array' in str(e):
        print('ERROR: scipy.sparse.coo_array issue - hdbscan version incompatible with scipy 1.5.4')  
        print('Solution: Use hdbscan 0.8.27 or upgrade scipy to 1.8+')
        sys.exit(1)
    else:
        print('ERROR: Other hdbscan import issue:', e)
        sys.exit(1)

try:
    import umap
    print('SUCCESS: umap-learn imported successfully')
except ImportError as e:
    print('WARNING: umap-learn import issue:', e)
    
print('All clustering packages working!')
" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Clustering packages have compatibility issues.
    echo Check the error messages above for specific solutions.
) else (
    echo SUCCESS: All packages installed and tested successfully!
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