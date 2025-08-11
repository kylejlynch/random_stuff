"""
HDBSCAN Clustering Module for Fraud Transaction Analysis
This module performs HDBSCAN clustering on PCA-transformed fraud data and saves clustering results.
"""

import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logging_config import get_logger

class HDBSCANClusterer:
    def __init__(self, min_cluster_size=30, min_samples=10, cluster_selection_epsilon=0.0):
        """
        Initialize HDBSCAN Clusterer
        
        Args:
            min_cluster_size (int): Minimum size of clusters
            min_samples (int): Minimum number of samples in a neighborhood for a point to be core
            cluster_selection_epsilon (float): Distance threshold for cluster extraction
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.clusterer = None
        self.labels_ = None
        self.cluster_stats = {}
        self.logger = get_logger(__name__)
        
    def load_pca_data(self, pca_file_path):
        """
        Load PCA-transformed data from CSV file
        
        Args:
            pca_file_path (str): Path to PCA results CSV file
            
        Returns:
            tuple: (features_array, metadata_dataframe)
        """
        self.logger.info(f"Loading PCA data from {pca_file_path}")
        df_pca = pd.read_csv(pca_file_path)
        
        # Separate PCA features from metadata
        pca_columns = [col for col in df_pca.columns if col.startswith('PC')]
        metadata_columns = [col for col in df_pca.columns if not col.startswith('PC')]
        
        X_pca = df_pca[pca_columns].values
        metadata = df_pca[metadata_columns].copy() if metadata_columns else pd.DataFrame()
        
        self.logger.info(f"Loaded PCA data: {X_pca.shape[0]} samples, {X_pca.shape[1]} components")
        return X_pca, metadata, df_pca
    
    def fit_clustering(self, X_pca):
        """
        Fit HDBSCAN clustering on PCA-transformed data
        
        Args:
            X_pca (numpy.array): PCA-transformed feature matrix
            
        Returns:
            numpy.array: Cluster labels
        """
        self.logger.info("Fitting HDBSCAN clustering")
        self.logger.info(f"Parameters: min_cluster_size={self.min_cluster_size}, "
                        f"min_samples={self.min_samples}, "
                        f"cluster_selection_epsilon={self.cluster_selection_epsilon}")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean'
        )
        
        self.labels_ = self.clusterer.fit_predict(X_pca)
        
        # Calculate clustering statistics
        self._calculate_cluster_stats(X_pca)
        
        return self.labels_
    
    def _calculate_cluster_stats(self, X_pca):
        """
        Calculate clustering statistics and metrics
        
        Args:
            X_pca (numpy.array): PCA-transformed data
        """
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.labels_ == -1)
        
        self.cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(self.labels_),
            'cluster_sizes': {}
        }
        
        # Calculate cluster sizes
        for label in unique_labels:
            if label != -1:
                size = np.sum(self.labels_ == label)
                self.cluster_stats['cluster_sizes'][f'cluster_{label}'] = size
        
        # Calculate silhouette score (only if we have clusters)
        if n_clusters > 1:
            # For silhouette score, we need to exclude noise points
            non_noise_mask = self.labels_ != -1
            if np.sum(non_noise_mask) > 0:
                try:
                    silhouette_avg = silhouette_score(
                        X_pca[non_noise_mask], 
                        self.labels_[non_noise_mask]
                    )
                    self.cluster_stats['silhouette_score'] = silhouette_avg
                except:
                    self.cluster_stats['silhouette_score'] = 'N/A'
            else:
                self.cluster_stats['silhouette_score'] = 'N/A'
        else:
            self.cluster_stats['silhouette_score'] = 'N/A'
        
        # Log statistics
        self.logger.info("\nClustering Results:")
        self.logger.info(f"Number of clusters: {n_clusters}")
        self.logger.info(f"Number of noise points: {n_noise} ({n_noise/len(self.labels_)*100:.1f}%)")
        self.logger.info(f"Silhouette score: {self.cluster_stats['silhouette_score']}")
        self.logger.info("Cluster sizes:")
        for cluster, size in self.cluster_stats['cluster_sizes'].items():
            self.logger.info(f"  {cluster}: {size}")
    
    def extract_centroids(self, X_pca):
        """
        Extract cluster centroids/medoids
        
        Args:
            X_pca (numpy.array): PCA-transformed data
            
        Returns:
            dict: Dictionary with cluster centroids
        """
        centroids = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_mask = self.labels_ == label
                cluster_points = X_pca[cluster_mask]
                
                # Calculate centroid (mean of cluster points)
                centroid = np.mean(cluster_points, axis=0)
                centroids[f'cluster_{label}'] = centroid
        
        self.logger.info(f"Extracted centroids for {len(centroids)} clusters")
        return centroids
    
    def identify_noise_points(self):
        """
        Identify noise points (points not assigned to any cluster)
        
        Returns:
            numpy.array: Indices of noise points
        """
        noise_indices = np.where(self.labels_ == -1)[0]
        self.logger.info(f"Identified {len(noise_indices)} noise points")
        return noise_indices
    
    def create_cluster_column(self):
        """
        Create a cluster/noise indicator column
        
        Returns:
            numpy.array: Array with cluster labels or 'noise'
        """
        cluster_labels = []
        for label in self.labels_:
            if label == -1:
                cluster_labels.append('noise')
            else:
                cluster_labels.append(f'cluster_{label}')
        
        return np.array(cluster_labels)
    
    def save_clustering_results(self, X_pca, df_original, output_path):
        """
        Save clustering results to CSV file
        
        Args:
            X_pca (numpy.array): PCA-transformed data
            df_original (pandas.DataFrame): Original dataframe with PCA results and metadata
            output_path (str): Path to save clustering results CSV
        """
        self.logger.info(f"Saving clustering results to {output_path}")
        
        # Create results dataframe
        df_results = df_original.copy()
        
        # Add clustering results
        df_results['cluster_label'] = self.labels_
        df_results['cluster_or_noise'] = self.create_cluster_column()
        df_results['is_noise'] = (self.labels_ == -1)
        
        # Add cluster probabilities if available
        if hasattr(self.clusterer, 'probabilities_'):
            df_results['cluster_probability'] = self.clusterer.probabilities_
        
        # Extract and save centroids
        centroids = self.extract_centroids(X_pca)
        
        # Create separate centroids dataframe
        if centroids:
            centroid_data = []
            pca_columns = [col for col in df_original.columns if col.startswith('PC')]
            
            for cluster_name, centroid in centroids.items():
                centroid_row = {'cluster_name': cluster_name}
                for i, pc_col in enumerate(pca_columns):
                    if i < len(centroid):
                        centroid_row[pc_col] = centroid[i]
                centroid_data.append(centroid_row)
            
            df_centroids = pd.DataFrame(centroid_data)
            
            # Save centroids separately
            centroids_path = output_path.replace('.csv', '_centroids.csv')
            df_centroids.to_csv(centroids_path, index=False)
            self.logger.info(f"Centroids saved to {centroids_path}")
        
        # Save main results
        df_results.to_csv(output_path, index=False)
        self.logger.info(f"Clustering results saved with shape: {df_results.shape}")
        
        # Save cluster statistics
        stats_path = output_path.replace('.csv', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("HDBSCAN Clustering Statistics\n")
            f.write("=" * 30 + "\n")
            for key, value in self.cluster_stats.items():
                if key == 'cluster_sizes':
                    f.write(f"{key}:\n")
                    for cluster, size in value.items():
                        f.write(f"  {cluster}: {size}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Statistics saved to {stats_path}")
        
        return df_results
    
    def run_full_clustering(self, pca_file_path, output_path="clustering_results.csv"):
        """
        Run complete HDBSCAN clustering pipeline
        
        Args:
            pca_file_path (str): Path to PCA results CSV
            output_path (str): Output CSV path for clustering results
            
        Returns:
            pandas.DataFrame: Clustering results dataframe
        """
        self.logger.info("Starting HDBSCAN Clustering")
        
        # Load PCA data
        X_pca, metadata, df_pca = self.load_pca_data(pca_file_path)
        
        # Fit clustering
        labels = self.fit_clustering(X_pca)
        
        # Save results
        df_results = self.save_clustering_results(X_pca, df_pca, output_path)
        
        self.logger.info("HDBSCAN Clustering Complete")
        return df_results

if __name__ == "__main__":
    # Example usage
    clusterer = HDBSCANClusterer(
        min_cluster_size=50,    # Adjust based on dataset size
        min_samples=10,         # Minimum samples for core points
        cluster_selection_epsilon=0.0  # Use default cluster selection
    )
    
    # Run clustering
    df_results = clusterer.run_full_clustering(
        pca_file_path="pca_results.csv",
        output_path="clustering_results.csv"
    )
    
    logger = get_logger(__name__)
    logger.info("Clustering analysis completed. Results saved to clustering_results.csv")