@echo off
REM Fraud Clustering Pipeline - Batch Script
REM This script activates the conda environment and runs the fraud clustering pipeline
REM Usage: Double-click this file or run from command line

echo ========================================
echo Fraud Clustering Pipeline
echo ========================================
echo.

REM Check if conda environment exists
echo Checking conda environment 'fraud_clustering'...
call conda env list | find "fraud_clustering" >nul
if errorlevel 1 (
    echo ERROR: Conda environment 'fraud_clustering' not found.
    echo Please create the environment first by running:
    echo   conda create -n fraud_clustering python=3.6 -y
    echo   conda activate fraud_clustering
    echo   conda install pandas scikit-learn matplotlib seaborn plotly -y
    echo   conda install -c conda-forge hdbscan umap-learn -y
    pause
    exit /b 1
)

REM Activate the conda environment
echo Activating conda environment...
call conda activate fraud_clustering
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'fraud_clustering'
    pause
    exit /b 1
)

echo Environment activated successfully.
echo.

REM Check if required data file exists
echo Checking for data file...
if not exist "fraudTrain.csv" (
    if not exist "fraudTest.csv" (
        echo ERROR: No fraud dataset found.
        echo Please ensure either 'fraudTrain.csv' or 'fraudTest.csv' exists in this directory.
        pause
        exit /b 1
    ) else (
        echo Using fraudTest.csv as the dataset...
        set DATA_FILE=fraudTest.csv
    )
) else (
    echo Using fraudTrain.csv as the dataset...
    set DATA_FILE=fraudTrain.csv
)

echo.
echo ========================================
echo Running Fraud Clustering Pipeline
echo ========================================
echo Dataset: %DATA_FILE%
echo Output Directory: results
echo.
echo The pipeline will perform:
echo   1. PCA Analysis (10 components)
echo   2. HDBSCAN Clustering
echo   3. 3D UMAP Visualization
echo.

REM Run the main pipeline script
python fraud_clustering_pipeline.py --data %DATA_FILE% --output results

REM Check if the pipeline completed successfully
if errorlevel 1 (
    echo.
    echo ========================================
    echo Pipeline completed with errors!
    echo ========================================
    echo Please check the error messages above.
) else (
    echo.
    echo ========================================
    echo Pipeline completed successfully!
    echo ========================================
    echo.
    echo Output files generated:
    echo   results\pca_results.csv
    echo   results\clustering_results.csv
    echo   results\fraud_clusters_3d.html
    echo.
    echo To view the interactive 3D visualization:
    echo   Open results\fraud_clusters_3d.html in your web browser
    echo.
    echo To run with custom parameters, use:
    echo   python fraud_clustering_pipeline.py --help
)

echo.
pause