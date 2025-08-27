"""
Parameter Tuning Module for Fraud Clustering Pipeline
This module provides grid search functionality to find optimal parameters for PCA, HDBSCAN, and UMAP.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
from itertools import product
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Clustering and validation
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Pipeline components
from pca_analysis import PCAAnalyzer
from hdbscan_clustering import HDBSCANClusterer
from logging_config import setup_pipeline_logging, get_logger

class ParameterTuner:
    """
    Parameter tuning class for fraud clustering pipeline
    """
    
    def __init__(self, verbose=False):
        """
        Initialize parameter tuner
        
        Args:
            verbose (bool): Enable verbose logging
        """
        setup_pipeline_logging(verbose=verbose)
        self.logger = get_logger(__name__)
        self.results = []
        
    def calculate_clustering_metrics(self, X, labels):
        """
        Calculate clustering validation metrics
        
        Args:
            X (numpy.array): Feature matrix
            labels (numpy.array): Cluster labels
            
        Returns:
            dict: Dictionary of metrics
        """
        # Remove noise points for some metrics
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        metrics = {
            'n_clusters': len(np.unique(labels_clean)) if len(labels_clean) > 0 else 0,
            'n_noise': np.sum(labels == -1),
            'noise_ratio': np.mean(labels == -1),
            'cluster_sizes': {},
            'silhouette_score': None,
            'calinski_harabasz_score': None,
            'davies_bouldin_score': None
        }
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels_clean, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels.astype(int), counts.astype(int)))
        
        # Calculate metrics only if we have clusters
        if metrics['n_clusters'] > 1 and len(labels_clean) > 0:
            try:
                metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
            except:
                pass
                
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
            except:
                pass
                
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
            except:
                pass
        
        return metrics
    
    def calculate_composite_score(self, metrics):
        """
        Calculate a composite score for parameter ranking
        
        Args:
            metrics (dict): Clustering metrics
            
        Returns:
            float: Composite score (higher is better)
        """
        score = 0.0
        weight_total = 0.0
        
        # Silhouette score (higher is better, range -1 to 1)
        if metrics['silhouette_score'] is not None:
            score += (metrics['silhouette_score'] + 1) / 2 * 0.4  # Normalize to 0-1
            weight_total += 0.4
        
        # Number of clusters (prefer moderate number, penalize too few or too many)
        n_clusters = metrics['n_clusters']
        if n_clusters > 0:
            if 3 <= n_clusters <= 15:
                cluster_score = 1.0
            elif n_clusters < 3:
                cluster_score = n_clusters / 3.0
            else:
                cluster_score = max(0, 1.0 - (n_clusters - 15) / 20.0)
            
            score += cluster_score * 0.3
            weight_total += 0.3
        
        # Noise ratio (prefer low noise, but some is acceptable)
        noise_ratio = metrics['noise_ratio']
        if 0 <= noise_ratio <= 0.2:
            noise_score = 1.0
        elif noise_ratio <= 0.5:
            noise_score = 1.0 - (noise_ratio - 0.2) / 0.3
        else:
            noise_score = 0.0
        
        score += noise_score * 0.2
        weight_total += 0.2
        
        # Davies-Bouldin score (lower is better)
        if metrics['davies_bouldin_score'] is not None:
            db_score = max(0, 1.0 - metrics['davies_bouldin_score'] / 5.0)  # Normalize roughly
            score += db_score * 0.1
            weight_total += 0.1
        
        # Normalize by total weight
        if weight_total > 0:
            score = score / weight_total
        
        return score
    
    def tune_pca_components(self, df, feature_columns, component_range=(2, 20)):
        """
        Tune number of PCA components
        
        Args:
            df (pandas.DataFrame): Fraud data
            feature_columns (list): Feature columns for PCA
            component_range (tuple): Range of components to test
            
        Returns:
            list: Results for different component numbers
        """
        self.logger.info(f"Tuning PCA components from {component_range[0]} to {component_range[1]}")
        
        pca_results = []
        
        for n_components in range(component_range[0], min(component_range[1] + 1, len(feature_columns) + 1)):
            try:
                # Create PCA analyzer
                analyzer = PCAAnalyzer(n_components=n_components)
                X_scaled = analyzer.prepare_features(df, feature_columns)
                X_pca = analyzer.fit_transform(X_scaled)
                
                # Calculate explained variance
                explained_variance_ratio = analyzer.pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                result = {
                    'n_components': n_components,
                    'explained_variance_ratio': explained_variance_ratio[-1] if len(explained_variance_ratio) > 0 else 0,
                    'cumulative_variance': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0,
                    'total_variance_explained': np.sum(explained_variance_ratio)
                }
                
                pca_results.append(result)
                self.logger.info(f"PCA {n_components} components: {result['total_variance_explained']:.3f} total variance")
                
            except Exception as e:
                self.logger.warning(f"Failed to test {n_components} components: {e}")
        
        return pca_results
    
    def tune_hdbscan_parameters(self, X_pca, param_grid):
        """
        Tune HDBSCAN parameters using grid search
        
        Args:
            X_pca (numpy.array): PCA-transformed features
            param_grid (dict): Parameter grid to search
            
        Returns:
            list: Results for each parameter combination
        """
        self.logger.info("Starting HDBSCAN parameter grid search")
        
        # Get parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            try:
                start_time = time.time()
                
                # Create and fit HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=params.get('min_cluster_size', 50),
                    min_samples=params.get('min_samples', 10),
                    metric=params.get('metric', 'euclidean'),
                    alpha=params.get('alpha', 1.0)
                )
                
                labels = clusterer.fit_predict(X_pca)
                
                # Calculate metrics
                metrics = self.calculate_clustering_metrics(X_pca, labels)
                composite_score = self.calculate_composite_score(metrics)
                
                runtime = time.time() - start_time
                
                result = {
                    'parameters': params,
                    'composite_score': composite_score,
                    'runtime': runtime,
                    **metrics
                }
                
                results.append(result)
                
                self.logger.info(f"Combo {i+1}/{len(param_combinations)}: "
                               f"Score={composite_score:.3f}, "
                               f"Clusters={metrics['n_clusters']}, "
                               f"Noise={metrics['noise_ratio']:.2f}, "
                               f"Time={runtime:.2f}s")
                
            except Exception as e:
                self.logger.warning(f"Failed parameter combination {params}: {e}")
        
        # Sort by composite score
        results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        
        return results
    
    def run_full_parameter_search(self, df, feature_columns, 
                                 pca_range=(3, 15),
                                 hdbscan_grid=None,
                                 max_pca_candidates=5):
        """
        Run complete parameter search for the pipeline
        
        Args:
            df (pandas.DataFrame): Fraud data
            feature_columns (list): Features for PCA
            pca_range (tuple): Range of PCA components to test
            hdbscan_grid (dict): HDBSCAN parameter grid
            max_pca_candidates (int): Maximum number of PCA configurations to test with HDBSCAN
            
        Returns:
            dict: Complete search results
        """
        self.logger.info("Starting full parameter search")
        start_time = time.time()
        
        # Default HDBSCAN grid if not provided
        if hdbscan_grid is None:
            # Adjusted for smaller datasets common in fraud analysis
            hdbscan_grid = {
                'min_cluster_size': [10, 20, 30, 50, 75],
                'min_samples': [5, 10, 15, 20],
                'metric': ['euclidean', 'manhattan'],
                'alpha': [0.8, 1.0, 1.2]
            }
        
        # Step 1: Find best PCA components
        self.logger.info("Step 1: Tuning PCA components")
        pca_results = self.tune_pca_components(df, feature_columns, pca_range)
        
        # Select top PCA configurations (aim for good variance explanation)
        pca_results_sorted = sorted(pca_results, 
                                  key=lambda x: x['total_variance_explained'], 
                                  reverse=True)
        
        top_pca_configs = pca_results_sorted[:max_pca_candidates]
        
        self.logger.info(f"Selected top {len(top_pca_configs)} PCA configurations")
        for config in top_pca_configs:
            self.logger.info(f"  {config['n_components']} components: "
                           f"{config['total_variance_explained']:.3f} variance explained")
        
        # Step 2: Test HDBSCAN parameters for each PCA configuration
        self.logger.info("Step 2: Tuning HDBSCAN parameters")
        all_results = []
        
        for pca_config in top_pca_configs:
            self.logger.info(f"Testing HDBSCAN parameters with {pca_config['n_components']} PCA components")
            
            # Prepare PCA data
            analyzer = PCAAnalyzer(n_components=pca_config['n_components'])
            X_scaled = analyzer.prepare_features(df, feature_columns)
            X_pca = analyzer.fit_transform(X_scaled)
            
            # Run HDBSCAN grid search
            hdbscan_results = self.tune_hdbscan_parameters(X_pca, hdbscan_grid)
            
            # Add PCA info to each result
            for result in hdbscan_results:
                result['pca_components'] = pca_config['n_components']
                result['pca_variance_explained'] = pca_config['total_variance_explained']
                all_results.append(result)
        
        # Sort all results by composite score
        all_results = sorted(all_results, key=lambda x: x['composite_score'], reverse=True)
        
        total_time = time.time() - start_time
        
        # Prepare summary
        search_summary = {
            'total_combinations_tested': len(all_results),
            'total_runtime': total_time,
            'best_parameters': all_results[0] if all_results else None,
            'top_10_results': all_results[:10] if len(all_results) >= 10 else all_results,
            'all_results': all_results,
            'pca_analysis': pca_results
        }
        
        self.logger.info(f"Parameter search completed in {total_time:.2f} seconds")
        self.logger.info(f"Tested {len(all_results)} total combinations")
        
        if search_summary['best_parameters']:
            best = search_summary['best_parameters']
            self.logger.info("Best parameters found:")
            self.logger.info(f"  PCA components: {best['pca_components']}")
            self.logger.info(f"  HDBSCAN min_cluster_size: {best['parameters']['min_cluster_size']}")
            self.logger.info(f"  HDBSCAN min_samples: {best['parameters']['min_samples']}")
            self.logger.info(f"  Metric: {best['parameters']['metric']}")
            self.logger.info(f"  Composite score: {best['composite_score']:.3f}")
            self.logger.info(f"  Number of clusters: {best['n_clusters']}")
            self.logger.info(f"  Noise ratio: {best['noise_ratio']:.3f}")
        
        return search_summary
    
    def save_results_to_csv(self, search_results, output_path="parameter_search_results.csv"):
        """
        Save parameter search results to CSV
        
        Args:
            search_results (dict): Results from run_full_parameter_search
            output_path (str): Output CSV path
        """
        if not search_results['all_results']:
            self.logger.warning("No results to save")
            return
        
        # Prepare data for CSV
        rows = []
        for result in search_results['all_results']:
            row = {
                'pca_components': result['pca_components'],
                'pca_variance_explained': result['pca_variance_explained'],
                'min_cluster_size': result['parameters']['min_cluster_size'],
                'min_samples': result['parameters']['min_samples'],
                'metric': result['parameters']['metric'],
                'alpha': result['parameters']['alpha'],
                'composite_score': result['composite_score'],
                'n_clusters': result['n_clusters'],
                'n_noise': result['n_noise'],
                'noise_ratio': result['noise_ratio'],
                'silhouette_score': result['silhouette_score'],
                'calinski_harabasz_score': result['calinski_harabasz_score'],
                'davies_bouldin_score': result['davies_bouldin_score'],
                'runtime': result['runtime']
            }
            rows.append(row)
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(output_path, index=False)
        
        self.logger.info(f"Parameter search results saved to: {output_path}")
    
    def get_best_parameters(self, search_results):
        """
        Extract the best parameters in a format ready for pipeline use
        
        Args:
            search_results (dict): Results from run_full_parameter_search
            
        Returns:
            dict: Best parameters formatted for pipeline
        """
        if not search_results['best_parameters']:
            return None
        
        best = search_results['best_parameters']
        
        return {
            'pca_components': best['pca_components'],
            'hdbscan_min_cluster_size': best['parameters']['min_cluster_size'],
            'hdbscan_min_samples': best['parameters']['min_samples'],
            'hdbscan_metric': best['parameters']['metric'],
            'hdbscan_alpha': best['parameters']['alpha'],
            'expected_clusters': best['n_clusters'],
            'expected_noise_ratio': best['noise_ratio'],
            'composite_score': best['composite_score']
        }


def quick_parameter_search(df, feature_columns=None, output_csv="quick_search_results.csv"):
    """
    Convenience function for quick parameter search
    
    Args:
        df (pandas.DataFrame): Fraud DataFrame
        feature_columns (list): Features for analysis
        output_csv (str): Output file path
        
    Returns:
        dict: Search results
    """
    # Default feature columns
    if feature_columns is None:
        feature_columns = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
        # Filter for available columns
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Filter for fraud only
    fraud_df = df[df['is_fraud'] == 1].copy() if 'is_fraud' in df.columns else df.copy()
    
    # Smaller grid for quick search
    quick_grid = {
        'min_cluster_size': [20, 50, 75],
        'min_samples': [5, 10, 15],
        'metric': ['euclidean'],
        'alpha': [1.0]
    }
    
    # Run search
    tuner = ParameterTuner(verbose=True)
    results = tuner.run_full_parameter_search(
        fraud_df, 
        feature_columns, 
        pca_range=(3, 10),
        hdbscan_grid=quick_grid,
        max_pca_candidates=3
    )
    
    # Save results
    tuner.save_results_to_csv(results, output_csv)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Parameter Tuning for Fraud Clustering")
    print("=" * 50)
    
    # This would typically load your fraud data
    print("To use this module:")
    print("1. Import: from parameter_tuning import quick_parameter_search")
    print("2. Load your data: df = pd.read_csv('your_fraud_data.csv')")
    print("3. Run search: results = quick_parameter_search(df)")
    print("4. Get best params: best = get_best_parameters(results)")
    print("5. Use in pipeline with the best parameters found")