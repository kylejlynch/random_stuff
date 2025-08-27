# Fraud Clustering Pipeline with Alert Coverage Analysis

A comprehensive fraud detection analysis tool that combines advanced clustering techniques with alert coverage analysis to identify weak "pockets" in fraud detection systems.

## 🚀 Features

### 🎯 Automatic Parameter Tuning
- **Smart Grid Search**: Automatically finds optimal clustering parameters
- **Adaptive Strategy**: Adjusts search complexity based on dataset size
- **Multiple Metrics**: Comprehensive validation using silhouette, Calinski-Harabasz, and Davies-Bouldin scores
- **Interactive Visualizations**: Explore parameter relationships and performance

### 🚨 Alert Coverage Analysis
- **Weak Pocket Detection**: Identifies clusters and regions with poor fraud alerting
- **Density-based Analysis**: Finds dense fraud areas with missing alerts  
- **Spatial Gap Analysis**: UMAP-based geographic/feature space analysis
- **Risk Scoring**: Identifies highest-risk missed transactions
- **Actionable Recommendations**: Prioritized improvement suggestions

### 🔧 Work Environment Ready
- **Package Constraints**: Compatible with restricted corporate environments
- **Version Pinning**: Tested with specific package versions for stability
- **Compatibility Handling**: Automatic sklearn.externals fixes
- **Easy Setup**: One-command environment validation

## 📦 Installation

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd clustering

# Install dependencies
pip install -r requirements.txt

# Validate environment
python setup_environment.py
```

### Work Environment Setup
For restricted corporate environments:
```bash
# Use the specific work-compatible versions
pip install pandas==1.1.5 numpy==1.19.5 scipy==1.5.4 scikit-learn==0.24.2 hdbscan==0.8.20 umap-learn==0.5.3 plotly>=4.14.0,<5.0.0 matplotlib>=3.2.0,<3.4.0 seaborn>=0.10.0,<0.12.0 pynndescent==0.5.13 joblib>=1.0.0 numba==0.53.1 llvmlite==0.36.0 six
```

### Conda Environment
```bash
conda env create -f environment.yml
conda activate fraud_clustering_work
```

## 🎯 Quick Start

### One-Line Analysis
```python
from auto_tuned_pipeline import quick_fraud_analysis

# Load your fraud data (must have 'is_fraud' column)
df = pd.read_csv('your_fraud_data.csv')

# Run complete analysis with automatic parameter tuning
results = quick_fraud_analysis(df)
```

### Alert Coverage Analysis
```python
from enhanced_pipeline_with_alerts import analyze_fraud_with_alert_coverage

# Your data needs 'is_fraud' and 'alerted' columns
df = pd.read_csv('fraud_data_with_alerts.csv')

# Find weak pockets in your alerting system
results, pipeline = analyze_fraud_with_alert_coverage(df)

# Get specific problem areas
worst_pockets = pipeline.find_worst_alert_pockets()
recommendations = pipeline.get_alert_improvement_recommendations()
```

### Parameter Tuning Only
```python
from parameter_tuning import quick_parameter_search

# Find optimal parameters for your dataset
search_results = quick_parameter_search(fraud_df)
best_params = get_best_parameters(search_results)
```

## 📊 What You Get

### Visualizations
- **Interactive 3D UMAP plots** colored by alerts, clusters, amounts
- **Parameter performance heatmaps** showing optimal combinations
- **Alert coverage analysis plots** highlighting weak pockets
- **Cluster performance dashboards** with detailed statistics

### Data Files
- **Parameter search results** (.csv) with all tested combinations
- **Alert coverage analysis** (.csv) with cluster and density metrics
- **High-risk missed transactions** (.csv) prioritized by risk score
- **Comprehensive reports** (.txt) with actionable recommendations

### Key Insights
- **Optimal clustering parameters** for your specific fraud patterns
- **Weak alert coverage areas** by cluster, geography, fraud type
- **High-risk missed transactions** requiring immediate attention
- **Improvement recommendations** prioritized by potential impact

## 💼 Use Cases

### For Data Scientists
- **Parameter optimization** without manual trial-and-error
- **Fraud pattern discovery** through advanced clustering
- **Model performance analysis** via comprehensive metrics

### For Fraud Analysts
- **Alert coverage gaps** identification and prioritization
- **High-risk case review** with automated risk scoring
- **System improvement** roadmap with specific recommendations

### For Management
- **Coverage metrics** and monitoring dashboards
- **ROI analysis** of fraud detection improvements
- **Risk quantification** of current alert gaps

## 🔍 Examples

### Example 1: Parameter Tuning
```python
# Run comprehensive parameter search
python example_parameter_tuning.py
```
**Outputs**: Parameter performance plots, best parameter combinations, comprehensive search results

### Example 2: Alert Coverage Analysis
```python  
# Analyze alert coverage gaps
python example_alert_coverage_analysis.py
```
**Outputs**: Weak pocket identification, improvement recommendations, monitoring metrics

### Example 3: Complete Pipeline
```python
# Full analysis with auto-tuning and alert coverage
from enhanced_pipeline_with_alerts import EnhancedFraudPipelineWithAlerts

pipeline = EnhancedFraudPipelineWithAlerts(verbose=True)
results = pipeline.run_comprehensive_analysis(your_fraud_df)
```

## 📁 Project Structure

```
clustering/
├── Core Modules
│   ├── compatibility.py              # Package compatibility handling
│   ├── config.py                    # Configuration settings
│   ├── logging_config.py            # Logging setup
│   ├── pca_analysis.py             # PCA implementation
│   ├── hdbscan_clustering.py       # HDBSCAN clustering
│   └── umap_visualization.py       # UMAP 3D visualization
│
├── Parameter Tuning
│   ├── parameter_tuning.py         # Grid search implementation
│   └── parameter_visualization.py  # Parameter analysis plots
│
├── Alert Coverage Analysis
│   ├── alert_coverage_analysis.py  # Alert gap detection
│   └── alert_coverage_visualization.py # Alert coverage plots
│
├── Integrated Pipelines
│   ├── dataframe_pipeline.py       # Basic DataFrame pipeline
│   ├── auto_tuned_pipeline.py      # Auto-tuned clustering
│   ├── enhanced_pipeline_with_alerts.py # Complete pipeline
│   └── fraud_clustering_pipeline.py # Original CSV-based pipeline
│
├── Examples & Setup
│   ├── example_dataframe_usage.py  # Basic usage examples
│   ├── example_parameter_tuning.py # Parameter tuning examples
│   ├── example_alert_coverage_analysis.py # Alert analysis examples
│   └── setup_environment.py        # Environment validation
│
└── Configuration
    ├── requirements.txt            # Work-compatible dependencies
    ├── environment.yml            # Conda environment
    └── README.md                  # This file
```

## 🔧 Configuration

### Work Environment Compatibility
The pipeline automatically handles sklearn compatibility issues common in corporate environments:

```python
# Automatically included in all modules
import sys, six, joblib
sys.modules['sklearn.externals.joblib'] = joblib  
sys.modules['sklearn.externals.six'] = six
```

### Clustering Parameters
Default parameters optimized for fraud data:
- **PCA Components**: Auto-tuned (typically 5-12)
- **HDBSCAN min_cluster_size**: Adaptive (20-150 based on data size)
- **HDBSCAN min_samples**: Adaptive (5-20 based on data size)
- **UMAP dimensions**: 3 (for interactive visualization)

### Alert Analysis Settings
- **Grid resolution**: 20x20 for UMAP spatial analysis
- **Density neighbors**: 20 for neighborhood analysis  
- **Risk scoring**: Multi-factor algorithm considering amount, cluster, density

## 📈 Performance

### Dataset Size Recommendations
- **Small (<500 fraud cases)**: Quick search, basic clustering
- **Medium (500-2000 fraud cases)**: Medium search, full analysis
- **Large (>2000 fraud cases)**: Comprehensive search, all features

### Runtime Expectations
- **Parameter tuning**: 2-15 minutes depending on search scope
- **Alert coverage analysis**: 1-5 minutes depending on dataset size
- **Visualization generation**: 30 seconds - 2 minutes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues
1. **Package compatibility errors**: Run `python setup_environment.py` to diagnose
2. **HDBSCAN parameter errors**: Update to remove `cluster_selection_epsilon` 
3. **Memory issues**: Reduce parameter search scope with `quick_search=True`

### Getting Help
- Check the examples in `example_*.py` files
- Review the changelog for recent updates
- Open an issue for bugs or feature requests

## 🔮 Roadmap

- [ ] Real-time monitoring dashboard
- [ ] Advanced ensemble clustering methods
- [ ] MLflow experiment tracking integration
- [ ] REST API for production deployment
- [ ] Automated alert model retraining recommendations