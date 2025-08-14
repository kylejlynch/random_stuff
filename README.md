# Fraud Clustering Pipeline

This pipeline performs PCA (Principal Component Analysis) and HDBSCAN clustering on fraudulent transactions, followed by 3D UMAP visualization.

## Overview

The pipeline consists of three main steps:
1. **PCA Analysis**: Reduces dimensionality of fraud transaction features
2. **HDBSCAN Clustering**: Identifies clusters in the PCA-transformed space
3. **3D UMAP Visualization**: Creates interactive 3D visualization with dropdown menus

## Requirements

- Python 3.6 (as specified for your work environment)
- Conda/Anaconda for environment management
- Fraud dataset (CSV format)

## Setup

### 1. Create Virtual Environment

```bash
conda create -n fraud_clustering python=3.6 -y
conda activate fraud_clustering
```

### 2. Install Required Packages

```bash
# Install core packages
conda install pandas scikit-learn matplotlib seaborn plotly -y

# Install specialized packages
conda install -c conda-forge hdbscan umap-learn -y
```

### 3. Prepare Your Dataset

Ensure your fraud dataset CSV file is in the project directory. The pipeline expects:
- A CSV file with fraud transaction data
- An `is_fraud` column indicating fraudulent transactions (1 = fraud, 0 = legitimate)
- Numerical features for analysis

## Usage

### Option 1: DataFrame Usage (Programmatic)

Work directly with pandas DataFrames in your Python code:

```python
import pandas as pd
from dataframe_pipeline import analyze_fraud_dataframe

# Load your fraud data into a DataFrame
df = pd.read_csv('your_fraud_data.csv')

# Quick analysis
results = analyze_fraud_dataframe(
    df,
    feature_columns=['amt', 'lat', 'long', 'city_pop', 'unix_time'],
    pca_components=8,
    min_cluster_size=50,
    output_html_path='my_clusters.html'
)

# Access results
fraud_data = results['fraud_data']          # Filtered fraud transactions
pca_results = results['pca_results']        # PCA-transformed data  
clusters = results['clustering_results']    # Clustering labels and probabilities
visualization = results['visualization']    # Interactive Plotly figure
stats = results['cluster_stats']           # Clustering statistics
```

#### Step-by-Step DataFrame Analysis
```python
from dataframe_pipeline import DataFrameFraudClusteringPipeline

# Initialize pipeline
pipeline = DataFrameFraudClusteringPipeline(
    pca_components=8,
    hdbscan_min_cluster_size=50,
    verbose=True
)

# Run step by step
fraud_df = pipeline.prepare_fraud_data(df)
pca_df = pipeline.run_pca_analysis(['amt', 'lat', 'long', 'city_pop'])
clustering_df = pipeline.run_clustering_analysis()
fig = pipeline.run_visualization('clusters.html')

# Get summary
summary = pipeline.get_cluster_summary()
```

### Option 2: Quick Start (Batch File)

Simply double-click `run_fraud_clustering.bat` or run it from command line:

```bash
run_fraud_clustering.bat
```

### Command Line Usage

```bash
# Activate environment
conda activate fraud_clustering

# Run with default parameters
python fraud_clustering_pipeline.py

# Run with custom parameters
python fraud_clustering_pipeline.py --data fraudTrain.csv --output results --pca-components 15

# Run with custom clustering parameters  
python fraud_clustering_pipeline.py --min-cluster-size 100 --min-samples 20

# See all options
python fraud_clustering_pipeline.py --help
```

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | Path to fraud dataset CSV | `fraudTrain.csv` |
| `--output` | Output directory for results | `results` |
| `--pca-components` | Number of PCA components | `10` |
| `--min-cluster-size` | HDBSCAN minimum cluster size | `50` |
| `--min-samples` | HDBSCAN minimum samples | `10` |
| `--umap-neighbors` | UMAP number of neighbors | `15` |
| `--umap-min-dist` | UMAP minimum distance | `0.1` |
| `--verbose`, `-v` | Enable verbose logging output | `False` |

## Logging

The pipeline features comprehensive logging for better process tracking and debugging:

### Log Files
- **Location**: `logs/` directory (created automatically)
- **Format**: `fraud_clustering_YYYYMMDD_HHMMSS.log`
- **Content**: Timestamped entries with module identification and log levels

### Log Levels
```bash
# Standard logging (INFO level) - clean console output
python fraud_clustering_pipeline.py --data fraudTrain.csv

# Verbose logging (DEBUG level) - detailed debugging info
python fraud_clustering_pipeline.py --data fraudTrain.csv --verbose
```

### Example Log Output
```
INFO - __main__ - FRAUD CLUSTERING PIPELINE
INFO - pca_analysis - Starting PCA Analysis  
INFO - pca_analysis - Filtered to 7506 fraudulent transactions from 1296675 total
INFO - hdbscan_clustering - Number of clusters: 4
INFO - umap_visualization - UMAP embedding shape: (7506, 3)
INFO - __main__ - PIPELINE COMPLETED SUCCESSFULLY!
```

### Error Handling
The pipeline provides structured error reporting with:
- Error type and detailed message
- Module where error occurred
- Full traceback (in DEBUG mode)
- Pipeline step context

## Output Files

The pipeline generates several output files in the results directory:

### 1. PCA Results (`pca_results.csv`)
- PCA-transformed data with principal components
- Metadata columns for visualization (alerted, return_reason, etc.)

### 2. Clustering Results (`clustering_results.csv`)
- All PCA data plus clustering labels
- Cluster assignments and noise point identification
- Additional files:
  - `clustering_results_centroids.csv`: Cluster centroids
  - `clustering_results_stats.txt`: Clustering statistics

### 3. Interactive Visualization (`fraud_clusters_3d.html`)
- 3D scatter plot using UMAP embedding
- Interactive dropdown menus for coloring by:
  - `alerted`: Whether the model alerted on the transaction
  - `return_reason`: Type of fraud 
  - `cluster_or_noise`: Cluster assignment or noise
  - `amount_tier`: Transaction amount tiers
  - `category`: Transaction category

## File Descriptions

### Core Modules

- `pca_analysis.py`: PCA analysis and dimensionality reduction
- `hdbscan_clustering.py`: HDBSCAN clustering implementation  
- `umap_visualization.py`: 3D UMAP visualization with interactive features
- `fraud_clustering_pipeline.py`: Main orchestration script (CSV-based)
- `dataframe_pipeline.py`: DataFrame-based pipeline for programmatic use
- `example_dataframe_usage.py`: Examples showing DataFrame usage patterns
- `logging_config.py`: Centralized logging configuration and utilities
- `config.py`: Configuration parameters
- `run_fraud_clustering.bat`: Windows batch file for easy execution

### Features Used for PCA

The pipeline uses these numerical features by default:
- `amt`: Transaction amount
- `lat`: Customer latitude
- `long`: Customer longitude  
- `city_pop`: City population
- `unix_time`: Transaction timestamp
- `merch_lat`: Merchant latitude
- `merch_long`: Merchant longitude
- `zip`: ZIP code

### Simulated Columns

Since the original dataset may not have required visualization columns, the pipeline adds:
- `alerted`: Simulated model alerts (70% of fraud cases alerted)
- `return_reason`: Simulated fraud types:
  - `card_theft`
  - `identity_theft`
  - `account_takeover` 
  - `synthetic_identity`
  - `first_party_fraud`

## Customization

### Modifying Parameters

Edit `config.py` to change default parameters:

```python
# Example: Increase cluster size requirement
HDBSCAN_MIN_CLUSTER_SIZE = 100

# Example: Use more PCA components  
PCA_COMPONENTS = 15

# Example: Add different features
PCA_FEATURES = ['amt', 'lat', 'long', 'city_pop', 'unix_time']
```

### Adding New Features

To add new numerical features to the PCA analysis:

1. Edit the `PCA_FEATURES` list in `config.py`
2. Ensure the features exist in your dataset
3. Re-run the pipeline

### Customizing Visualization

Modify visualization parameters in `config.py`:

```python
VIZ_POINT_SIZE = 8      # Larger points
VIZ_OPACITY = 0.5       # More transparent
VIZ_WIDTH = 1200        # Wider visualization
```

## Troubleshooting

### Common Issues

1. **Environment not found**: Ensure you've created the conda environment
2. **Missing packages**: Install all required packages as shown in setup
3. **No fraud data**: Ensure your CSV has an `is_fraud` column with 1s for fraudulent transactions
4. **Memory issues**: Reduce `PCA_COMPONENTS` or increase `HDBSCAN_MIN_CLUSTER_SIZE`

### Performance Tips

- For large datasets (>100k fraud transactions), increase `HDBSCAN_MIN_CLUSTER_SIZE`
- If clustering takes too long, reduce `PCA_COMPONENTS` or sample your data
- For better cluster separation, experiment with `UMAP_N_NEIGHBORS` and `UMAP_MIN_DIST`

## Expected Runtime

- **Small dataset** (1k-5k fraud transactions): 1-3 minutes
- **Medium dataset** (5k-20k fraud transactions): 3-10 minutes  
- **Large dataset** (20k+ fraud transactions): 10+ minutes

The visualization step is typically the fastest, while HDBSCAN clustering may take the longest for large datasets.

## Viewing Results

After the pipeline completes successfully:

1. Open `results/fraud_clusters_3d.html` in your web browser
2. Use the dropdown menu to color points by different attributes
3. Rotate and zoom the 3D plot to explore clusters
4. Hover over points to see transaction details

The interactive visualization allows you to:
- Rotate the 3D plot by clicking and dragging
- Zoom in/out using mouse wheel
- Change coloring scheme using the dropdown menu
- View transaction details by hovering over points

## Next Steps

For production deployment, you can:
1. Replace CSV file I/O with database connections (Hive/Impala)
2. Add automated parameter tuning
3. Implement cluster quality metrics
4. Add cluster interpretation and profiling
5. Schedule regular re-clustering of new fraud data