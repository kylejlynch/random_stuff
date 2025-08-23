@echo off
REM Fraud Clustering Pipeline - Environment Setup Script
REM This script sets up the conda environment for the fraud clustering pipeline

echo ========================================
echo Fraud Clustering Environment Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Step 1: Creating conda environment...
conda env create -f environment-portable.yml
if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment.
    echo Trying alternative method...
    echo.
    
    REM Fallback: Manual installation
    echo Creating environment manually...
    conda create -n fraud_clustering python=3.6 -y
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create basic environment.
        pause
        exit /b 1
    )
    
    echo Installing packages...
    call conda activate fraud_clustering
    conda install pandas scikit-learn matplotlib seaborn plotly -y
    conda install -c conda-forge hdbscan umap-learn -y
    pip install importlib-metadata typing-extensions
)

echo.
echo Step 2: Testing installation...
call conda activate fraud_clustering
python -c "import pandas, numpy, sklearn, hdbscan, umap, plotly, matplotlib, seaborn; print('All packages imported successfully!')"
if %errorlevel% neq 0 (
    echo WARNING: Some packages may not be working correctly.
    echo You may need to install missing dependencies manually.
) else (
    echo SUCCESS: Environment setup completed!
)

echo.
echo Step 3: Testing fraud clustering pipeline...
python -c "from dataframe_pipeline import DataFrameFraudClusteringPipeline; print('Pipeline modules imported successfully!')"
if %errorlevel% neq 0 (
    echo WARNING: Pipeline modules not found. Make sure you're in the correct directory.
) else (
    echo SUCCESS: Pipeline is ready to use!
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the environment:
echo   conda activate fraud_clustering
echo.
echo To run the pipeline:
echo   python fraud_clustering_pipeline.py --help
echo   python example_dataframe_usage.py
echo.
pause