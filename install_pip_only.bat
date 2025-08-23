@echo off
REM Fraud Clustering Pipeline - Pure pip installation (no conda)
REM Use this if you only have Python and pip available

echo ========================================
echo Fraud Clustering Pipeline - Pip Install
echo ========================================
echo.

REM Check if python is available
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
python -m venv fraud_clustering_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure you have venv module available.
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call fraud_clustering_env\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing packages from requirements...
pip install -r requirements-clean.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install.
    echo Trying individual installation...
    
    pip install pandas numpy scikit-learn matplotlib seaborn plotly
    pip install hdbscan umap-learn
    pip install importlib-metadata typing-extensions
)

echo.
echo Step 5: Testing installation...
python -c "import pandas, numpy, sklearn, hdbscan, umap, plotly, matplotlib, seaborn; print('All packages imported successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Installation verification failed.
    echo Some packages may not be working correctly.
    pause
    exit /b 1
)

echo Step 6: Testing pipeline...
python -c "from dataframe_pipeline import DataFrameFraudClusteringPipeline; print('Pipeline modules imported successfully!')"
if %errorlevel% neq 0 (
    echo WARNING: Pipeline modules not found. Make sure you're in the correct directory.
) else (
    echo SUCCESS: Pipeline is ready to use!
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Virtual environment created: fraud_clustering_env
echo.
echo To use the environment:
echo   fraud_clustering_env\Scripts\activate.bat
echo.
echo To run the pipeline:
echo   python fraud_clustering_pipeline.py --help
echo   python example_dataframe_usage.py
echo.
pause