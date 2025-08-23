# Installation Guide - Fraud Clustering Pipeline

This guide provides multiple installation methods for different environments and dependency management scenarios.

## Quick Installation Options

### Option 1: Conda Environment (Recommended)
If you have Miniconda or Anaconda installed:

```bash
# Clone the repository
git clone https://github.com/kylejlynch/random_stuff.git
cd random_stuff

# Automated setup
install_environment.bat

# OR manual setup
conda env create -f environment-portable.yml
conda activate fraud_clustering
```

### Option 2: Pip Virtual Environment
If you only have Python and pip:

```bash
# Clone the repository
git clone https://github.com/kylejlynch/random_stuff.git
cd random_stuff

# Automated setup
install_pip_only.bat

# OR manual setup
python -m venv fraud_clustering_env
fraud_clustering_env\Scripts\activate.bat
pip install -r requirements-clean.txt
```

### Option 3: System Python (Not Recommended)
Install directly to system Python:

```bash
pip install -r requirements-clean.txt
```

## Dependency Files Explained

### `requirements-clean.txt`
- Clean, portable pip requirements
- Version ranges for flexibility
- Compatible with Python 3.6+
- No conda-specific dependencies

### `environment-portable.yml`
- Conda environment specification
- Uses conda-forge for better compatibility
- Includes both conda and pip dependencies
- Optimized for cross-platform use

### `requirements.txt`
- Full pip freeze output (auto-generated)
- Contains exact versions from development environment
- May include system-specific packages
- Use for exact reproduction if needed

## Troubleshooting Installation Issues

### Common Problems and Solutions

#### 1. Conda Not Found
```
ERROR: Conda is not installed or not in PATH
```
**Solution**: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

#### 2. Python Version Issues
```
ERROR: Python 3.6+ is required
```
**Solution**: Upgrade Python or use conda to install Python 3.6:
```bash
conda create -n fraud_clustering python=3.6
```

#### 3. HDBSCAN Installation Fails
```
ERROR: Failed building wheel for hdbscan
```
**Solutions**:
```bash
# Option A: Use conda-forge
conda install -c conda-forge hdbscan

# Option B: Install build tools
pip install Cython
pip install hdbscan

# Option C: Use pre-compiled wheels
pip install --only-binary=all hdbscan
```

#### 4. UMAP Installation Fails
```
ERROR: Failed to build umap-learn
```
**Solutions**:
```bash
# Option A: Use conda-forge
conda install -c conda-forge umap-learn

# Option B: Install dependencies first
pip install numba
pip install umap-learn
```

#### 5. Windows Compiler Issues
```
ERROR: Microsoft Visual C++ 14.0 is required
```
**Solutions**:
```bash
# Option A: Use conda (bypasses compilation)
conda install -c conda-forge hdbscan umap-learn

# Option B: Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Alternative Package Sources

#### For Corporate/Restricted Networks:

1. **Conda with custom channels**:
```bash
conda install -c conda-forge -c defaults pandas scikit-learn hdbscan umap-learn matplotlib plotly seaborn
```

2. **Pip with trusted hosts**:
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-clean.txt
```

3. **Offline installation**:
```bash
# Download packages on connected machine
pip download -r requirements-clean.txt -d packages/

# Install on offline machine
pip install --find-links packages/ -r requirements-clean.txt --no-index
```

## Minimal Installation (Core Only)

If you have dependency issues, try installing only core packages:

```bash
pip install pandas numpy scikit-learn matplotlib plotly
```

Then run with reduced functionality:
```python
from dataframe_pipeline import DataFrameFraudClusteringPipeline

# This will still work but may have reduced clustering capabilities
pipeline = DataFrameFraudClusteringPipeline(use_hdbscan=False)
```

## Docker Alternative

For ultimate portability, use Docker:

```dockerfile
FROM python:3.6-slim

WORKDIR /app
COPY requirements-clean.txt .
RUN pip install -r requirements-clean.txt

COPY . .
CMD ["python", "example_dataframe_usage.py"]
```

## Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.6 | ✅ Supported | Primary target version |
| 3.7 | ✅ Supported | Fully compatible |
| 3.8 | ✅ Supported | All features work |
| 3.9 | ✅ Supported | May need newer package versions |
| 3.10+ | ⚠️ Experimental | Some packages may not be available |

## Package Versions

### Core Requirements (Minimum)
- Python: 3.6+
- pandas: 1.1.0+
- numpy: 1.19.0+
- scikit-learn: 0.24.0+

### Optional but Recommended
- hdbscan: 0.8.27+
- umap-learn: 0.5.1+
- plotly: 5.0.0+
- matplotlib: 3.3.0+
- seaborn: 0.11.0+

## Testing Installation

After installation, test with:

```python
# Test core imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Test specialized packages
import hdbscan
import umap

# Test pipeline
from dataframe_pipeline import analyze_fraud_dataframe
print("✅ All packages installed successfully!")
```

## Getting Help

If you encounter issues:

1. **Check Python version**: `python --version`
2. **Check pip version**: `pip --version`
3. **Try conda instead of pip**: Often resolves compilation issues
4. **Use virtual environments**: Prevents conflicts
5. **Check corporate firewall**: May block package downloads

For specific errors, search the error message online or check:
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/en/latest/installing.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/index.html)
- [Scikit-learn Installation Guide](https://scikit-learn.org/stable/install.html)