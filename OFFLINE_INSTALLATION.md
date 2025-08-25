# Offline Installation Guide

This guide explains how to install the Fraud Clustering Pipeline on computers without internet access using pre-downloaded package wheels.

## Package Size and GitHub Limitations

The complete package collection is **~113MB**, which exceeds GitHub's individual file limit of 100MB. Here are your options:

### Option 1: Download Individual Components (Recommended)

Download packages in parts by category:

**Core Data Packages** (~50MB):
- pandas, numpy, scikit-learn, scipy

**Visualization Packages** (~25MB):
- plotly, matplotlib, seaborn, pillow

**Clustering Packages** (~35MB):
- hdbscan, umap-learn, numba, llvmlite

**Utility Packages** (~3MB):
- All other dependencies

### Option 2: Use a File Sharing Service

Upload the complete `packages/` folder to:
- Google Drive, Dropbox, OneDrive
- Internal company file sharing
- Network drive accessible from work computer

### Option 3: Split Into Multiple Archives

```bash
# Split the packages directory
7zip a packages_part1.7z packages/*.whl -v50m
7zip a packages_part2.7z packages/*.tar.gz
```

## Installation Steps

### On Internet-Connected Computer:

1. **Download the repository**:
```bash
git clone https://github.com/kylejlynch/random_stuff.git
cd random_stuff
```

2. **Download packages** (already done - packages are in `packages/` directory):
```bash
pip download -r requirements-clean.txt -d packages/
```

3. **Transfer to work computer**:
   - Copy entire project folder including `packages/` directory
   - OR zip the entire folder: `zip -r fraud-clustering-offline.zip .`

### On Work Computer:

1. **Extract files** (if zipped):
```bash
unzip fraud-clustering-offline.zip
cd fraud-clustering-offline
```

2. **Run offline installation**:
```bash
install_offline.bat
```

3. **Manual installation** (if script fails):
```bash
python -m venv fraud_clustering_offline
fraud_clustering_offline\Scripts\activate.bat
pip install --find-links packages\ --no-index -r requirements-clean.txt
```

## Package Contents

The `packages/` directory contains:

### Core Dependencies (always needed):
- `pandas-1.1.5-cp36-cp36m-win_amd64.whl` (8.7MB)
- `numpy-1.19.5-cp36-cp36m-win_amd64.whl` (13.2MB)
- `scikit_learn-0.24.2-cp36-cp36m-win_amd64.whl` (6.8MB)
- `matplotlib-3.3.4-cp36-cp36m-win_amd64.whl` (8.5MB)
- `plotly-5.18.0-py3-none-any.whl` (15.6MB)

### Clustering Dependencies (optional but recommended):
- `hdbscan-0.8.27.tar.gz` (6.4MB) *requires compilation*
- `umap_learn-0.5.7-py3-none-any.whl` (88KB)
- `numba-0.53.1-cp36-cp36m-win_amd64.whl` (2.3MB)
- `llvmlite-0.36.0-cp36-cp36m-win_amd64.whl` (16.0MB)

### Utility Dependencies:
- Various small packages for compatibility and functionality

## Minimal Installation

If the full installation fails, install core packages only:

```bash
# Essential packages only
pip install --find-links packages\ --no-index pandas numpy scikit-learn matplotlib plotly

# Test basic functionality
python -c "import pandas, numpy; print('Basic installation works!')"
```

The pipeline will work with reduced functionality (no HDBSCAN clustering).

## Troubleshooting

### Common Issues:

1. **HDBSCAN compilation fails**:
   - Skip it: The pipeline can work without HDBSCAN
   - Solution: Use conda instead of pip (if available)

2. **Virtual environment creation fails**:
   - Install to system Python directly (not recommended but works)
   - Check if `venv` module is available

3. **Permission denied errors**:
   - Run command prompt as administrator
   - OR install to user directory: `pip install --user`

4. **Missing Visual C++ compiler**:
   - Only affects packages that need compilation (hdbscan)
   - Use conda packages when possible

### Verification Commands:

```bash
# Test core functionality
python -c "
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
print('✅ Core packages working!')
"

# Test clustering (optional)
python -c "
import hdbscan
import umap
print('✅ Clustering packages working!')
"

# Test pipeline
python -c "
from dataframe_pipeline import analyze_fraud_dataframe
print('✅ Pipeline ready!')
"
```

## Alternative for Corporate Environments

If offline installation is still problematic:

### Use Conda Packages:
```bash
# Download conda packages (often pre-compiled)
conda install --download-only -c conda-forge pandas numpy scikit-learn hdbscan umap-learn matplotlib plotly seaborn
```

### Use Docker:
```dockerfile
FROM python:3.6-slim
COPY requirements-clean.txt .
RUN pip install -r requirements-clean.txt
COPY . /app
WORKDIR /app
```

### Request IT Department:
Many companies can install packages through internal channels:
- Provide them with `requirements-clean.txt`
- Request specific packages: pandas, scikit-learn, hdbscan, umap-learn, plotly

## File Structure After Installation

```
fraud-clustering/
├── packages/                    # Downloaded wheels (113MB)
├── fraud_clustering_offline/    # Virtual environment (if created)
├── dataframe_pipeline.py        # Main pipeline module
├── example_dataframe_usage.py   # Usage examples
├── requirements-clean.txt       # Package requirements
├── install_offline.bat         # Installation script
└── OFFLINE_INSTALLATION.md     # This guide
```

## Success Verification

After installation, you should be able to run:

```bash
python example_dataframe_usage.py
```

This will create sample data and test the complete pipeline, generating an HTML visualization file you can open in your browser.