"""
DataFrame-based Fraud Clustering Pipeline
This module provides a convenient way to use the fraud clustering pipeline
directly with pandas DataFrames instead of CSV files.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
import tempfile
import os
from logging_config import setup_pipeline_logging, get_logger

# Import the existing pipeline components
from pca_analysis import PCAAnalyzer
from hdbscan_clustering import HDBSCANClusterer
from umap_visualization import UMAPVisualizer

class DataFrameFraudClusteringPipeline:
    """
    Fraud clustering pipeline that works directly with pandas DataFrames
    """
    
    def __init__(self, 
                 pca_components=8,
                 hdbscan_min_cluster_size=50,
                 hdbscan_min_samples=10,
                 umap_n_neighbors=15,
                 umap_min_dist=0.1,
                 verbose=False):
        """
        Initialize the DataFrame-based fraud clustering pipeline
        
        Args:
            pca_components (int): Number of PCA components to retain
            hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN
            hdbscan_min_samples (int): Minimum samples for HDBSCAN core points
            umap_n_neighbors (int): Number of neighbors for UMAP
            umap_min_dist (float): Minimum distance for UMAP
            verbose (bool): Enable verbose logging
        """
        # Setup logging
        setup_pipeline_logging(verbose=verbose)
        self.logger = get_logger(__name__)
        
        # Store parameters
        self.pca_components = pca_components
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        
        # Initialize components
        self.pca_analyzer = PCAAnalyzer(n_components=pca_components)
        self.hdbscan_clusterer = HDBSCANClusterer(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples
        )
        self.umap_visualizer = UMAPVisualizer(
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist
        )
        
        # Storage for intermediate results
        self.fraud_df = None
        self.pca_df = None
        self.clustering_df = None
        self.visualization_fig = None
    
    def validate_dataframe(self, df):
        """
        Validate that the input DataFrame has the required structure
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            bool: True if valid
        """
        required_columns = ['is_fraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check if we have fraud transactions
        fraud_count = df['is_fraud'].sum()
        if fraud_count == 0:
            self.logger.error("No fraudulent transactions found (is_fraud=1)")
            return False
        
        self.logger.info(f"Found {fraud_count} fraudulent transactions out of {len(df)} total")
        return True
    
    def prepare_fraud_data(self, df):
        """
        Filter and prepare fraud data with required columns
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            pandas.DataFrame: Processed fraud DataFrame
        """
        self.logger.info("Filtering for fraudulent transactions")
        
        # Filter for fraud transactions
        fraud_df = df[df['is_fraud'] == 1].copy()
        
        # Add required columns for visualization if they don't exist
        if 'alerted' not in fraud_df.columns:
            np.random.seed(42)
            fraud_df['alerted'] = np.random.choice([0, 1], size=len(fraud_df), p=[0.3, 0.7])
            self.logger.info("Added simulated 'alerted' column")
        
        if 'return_reason' not in fraud_df.columns:
            fraud_reasons = ['card_theft', 'identity_theft', 'account_takeover', 'synthetic_identity', 'first_party_fraud']
            fraud_df['return_reason'] = np.random.choice(fraud_reasons, size=len(fraud_df))
            self.logger.info("Added simulated 'return_reason' column")
        
        # Add transaction number if missing
        if 'trans_num' not in fraud_df.columns:
            fraud_df['trans_num'] = [f"txn_{i:06d}" for i in range(len(fraud_df))]
        
        self.fraud_df = fraud_df
        self.logger.info(f"Prepared {len(fraud_df)} fraud transactions for analysis")
        return fraud_df
    
    def run_pca_analysis(self, feature_columns=None):
        """
        Run PCA analysis on the fraud DataFrame
        
        Args:
            feature_columns (list): List of columns to use for PCA. If None, use defaults
            
        Returns:
            pandas.DataFrame: PCA results
        """
        if self.fraud_df is None:
            raise ValueError("Must call prepare_fraud_data() first")
        
        # Default numerical features
        if feature_columns is None:
            feature_columns = [
                'amt',           # Transaction amount
                'lat',           # Customer latitude  
                'long',          # Customer longitude
                'city_pop',      # City population
                'unix_time',     # Transaction timestamp
                'merch_lat',     # Merchant latitude
                'merch_long',    # Merchant longitude
                'zip'            # ZIP code
            ]
        
        # Check which features actually exist
        available_features = [col for col in feature_columns if col in self.fraud_df.columns]
        missing_features = [col for col in feature_columns if col not in self.fraud_df.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        if len(available_features) == 0:
            raise ValueError("No valid numerical features found in DataFrame")
        
        self.logger.info(f"Using {len(available_features)} features for PCA: {available_features}")
        
        # Adjust PCA components if necessary
        max_components = min(len(available_features), len(self.fraud_df))
        if self.pca_components > max_components:
            self.logger.warning(f"Reducing PCA components from {self.pca_components} to {max_components}")
            self.pca_analyzer.n_components = max_components
            self.pca_analyzer.pca = self.pca_analyzer.pca.__class__(n_components=max_components)
        
        # Prepare features
        X_scaled = self.pca_analyzer.prepare_features(self.fraud_df, available_features)
        
        # Fit and transform
        X_pca = self.pca_analyzer.fit_transform(X_scaled)
        
        # Create PCA DataFrame
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        self.pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=self.fraud_df.index)
        
        # Add metadata
        metadata_columns = ['alerted', 'return_reason', 'trans_num', 'amt', 'category']
        for col in metadata_columns:
            if col in self.fraud_df.columns:
                self.pca_df[col] = self.fraud_df[col].values
        
        self.logger.info(f"PCA analysis complete. Shape: {self.pca_df.shape}")
        return self.pca_df
    
    def run_clustering_analysis(self):
        """
        Run HDBSCAN clustering on PCA results
        
        Returns:
            pandas.DataFrame: Clustering results
        """
        if self.pca_df is None:
            raise ValueError("Must run PCA analysis first")
        
        # Get PCA features
        pca_columns = [col for col in self.pca_df.columns if col.startswith('PC')]
        X_pca = self.pca_df[pca_columns].values
        
        # Fit clustering
        labels = self.hdbscan_clusterer.fit_clustering(X_pca)
        
        # Create clustering DataFrame
        self.clustering_df = self.pca_df.copy()
        self.clustering_df['cluster_label'] = labels
        self.clustering_df['cluster_or_noise'] = self.hdbscan_clusterer.create_cluster_column()
        self.clustering_df['is_noise'] = (labels == -1)
        
        # Add cluster probabilities if available
        if hasattr(self.hdbscan_clusterer.clusterer, 'probabilities_'):
            self.clustering_df['cluster_probability'] = self.hdbscan_clusterer.clusterer.probabilities_
        
        self.logger.info(f"Clustering analysis complete. Shape: {self.clustering_df.shape}")
        return self.clustering_df
    
    def run_visualization(self, output_path=None):
        """
        Run 3D UMAP visualization
        
        Args:
            output_path (str): Path to save HTML file. If None, returns figure object
            
        Returns:
            plotly.graph_objects.Figure: Interactive visualization figure
        """
        if self.clustering_df is None:
            raise ValueError("Must run clustering analysis first")
        
        # Get PCA features
        pca_columns = [col for col in self.clustering_df.columns if col.startswith('PC')]
        X_pca = self.clustering_df[pca_columns].values
        
        # Fit UMAP
        embedding = self.umap_visualizer.fit_umap(X_pca)
        
        # Prepare color options
        color_options = self.umap_visualizer.prepare_color_options(self.clustering_df)
        
        # Create visualization
        fig = self.umap_visualizer.create_3d_scatter(embedding, color_options, self.clustering_df)
        
        # Save if path provided
        if output_path:
            self.umap_visualizer.save_html(fig, output_path)
            self.logger.info(f"Visualization saved to {output_path}")
        
        self.visualization_fig = fig
        return fig
    
    def run_full_pipeline(self, df, feature_columns=None, output_html_path=None):
        """
        Run the complete pipeline on a DataFrame
        
        Args:
            df (pandas.DataFrame): Input DataFrame with fraud transactions
            feature_columns (list): Features to use for PCA
            output_html_path (str): Path to save visualization HTML
            
        Returns:
            dict: Dictionary with all results
        """
        self.logger.info("Starting DataFrame-based fraud clustering pipeline")
        
        # Validate input
        if not self.validate_dataframe(df):
            raise ValueError("DataFrame validation failed")
        
        # Prepare fraud data
        fraud_df = self.prepare_fraud_data(df)
        
        # Run PCA
        pca_df = self.run_pca_analysis(feature_columns)
        
        # Run clustering
        clustering_df = self.run_clustering_analysis()
        
        # Run visualization
        fig = self.run_visualization(output_html_path)
        
        self.logger.info("DataFrame-based pipeline completed successfully")
        
        return {
            'fraud_data': fraud_df,
            'pca_results': pca_df,
            'clustering_results': clustering_df,
            'visualization': fig,
            'cluster_stats': self.hdbscan_clusterer.cluster_stats
        }
    
    def get_cluster_summary(self):
        """
        Get a summary of clustering results
        
        Returns:
            dict: Cluster summary statistics
        """
        if self.clustering_df is None:
            return None
        
        summary = {
            'total_transactions': len(self.clustering_df),
            'n_clusters': len(self.clustering_df['cluster_label'].unique()) - (1 if -1 in self.clustering_df['cluster_label'].values else 0),
            'n_noise': (self.clustering_df['cluster_label'] == -1).sum(),
            'noise_ratio': (self.clustering_df['cluster_label'] == -1).mean(),
            'cluster_sizes': self.clustering_df[self.clustering_df['cluster_label'] != -1]['cluster_label'].value_counts().to_dict()
        }
        
        return summary

# Convenience function for quick usage
def analyze_fraud_dataframe(df, 
                          feature_columns=None,
                          pca_components=8,
                          min_cluster_size=50,
                          output_html_path="fraud_clusters_3d.html",
                          verbose=False):
    """
    Quick function to analyze a fraud DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with fraud data
        feature_columns (list): Features to use for analysis
        pca_components (int): Number of PCA components
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        output_html_path (str): Where to save visualization
        verbose (bool): Enable verbose logging
        
    Returns:
        dict: Complete analysis results
    """
    pipeline = DataFrameFraudClusteringPipeline(
        pca_components=pca_components,
        hdbscan_min_cluster_size=min_cluster_size,
        verbose=verbose
    )
    
    return pipeline.run_full_pipeline(df, feature_columns, output_html_path)

if __name__ == "__main__":
    # Example usage
    print("DataFrame-based Fraud Clustering Pipeline")
    print("Example usage:")
    print("""
    import pandas as pd
    from dataframe_pipeline import analyze_fraud_dataframe
    
    # Load your fraud data
    df = pd.read_csv('your_fraud_data.csv')
    
    # Run analysis
    results = analyze_fraud_dataframe(
        df, 
        feature_columns=['amt', 'lat', 'long', 'city_pop'],
        min_cluster_size=30,
        output_html_path='my_clusters.html'
    )
    
    # Access results
    fraud_data = results['fraud_data']
    clusters = results['clustering_results']  
    visualization = results['visualization']
    stats = results['cluster_stats']
    """)