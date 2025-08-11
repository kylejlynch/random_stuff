"""
Configuration file for Fraud Clustering Pipeline
Modify these parameters to customize the analysis
"""

# Dataset Configuration
DATASET_PATH = "fraudTrain.csv"  # Path to your fraud dataset
OUTPUT_DIRECTORY = "results"      # Directory to save all outputs

# PCA Configuration
PCA_COMPONENTS = 8               # Number of principal components to retain
PCA_FEATURES = [
    'amt',           # Transaction amount
    'lat',           # Customer latitude  
    'long',          # Customer longitude
    'city_pop',      # City population
    'unix_time',     # Transaction timestamp
    'merch_lat',     # Merchant latitude
    'merch_long',    # Merchant longitude
    'zip'            # ZIP code
]

# HDBSCAN Clustering Configuration
HDBSCAN_MIN_CLUSTER_SIZE = 50    # Minimum size of clusters
HDBSCAN_MIN_SAMPLES = 10         # Minimum samples for core points
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0  # Distance threshold for cluster extraction

# UMAP Visualization Configuration
UMAP_N_NEIGHBORS = 15            # Number of neighbors for UMAP
UMAP_MIN_DIST = 0.1             # Minimum distance for UMAP
UMAP_N_COMPONENTS = 3           # Number of dimensions (should be 3 for 3D viz)
UMAP_RANDOM_STATE = 42          # Random state for reproducibility

# Visualization Configuration
VIZ_POINT_SIZE = 5              # Size of points in 3D scatter plot
VIZ_OPACITY = 0.7               # Opacity of points (0.0 to 1.0)
VIZ_WIDTH = 1000                # Width of the visualization
VIZ_HEIGHT = 800                # Height of the visualization

# Additional Features Configuration
SIMULATE_ALERTED_COLUMN = True   # Whether to simulate 'alerted' column
SIMULATE_RETURN_REASON = True    # Whether to simulate 'return_reason' column

# Simulated return reasons (fraud types)
FRAUD_TYPES = [
    'card_theft',
    'identity_theft', 
    'account_takeover',
    'synthetic_identity',
    'first_party_fraud'
]

# Alert simulation probability (probability that a fraud transaction was alerted on)
ALERT_PROBABILITY = 0.7