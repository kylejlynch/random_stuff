"""
Auto-Tuned Fraud Clustering Pipeline
This pipeline automatically finds the best parameters using grid search before running the analysis.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Import pipeline components
from dataframe_pipeline import DataFrameFraudClusteringPipeline
from parameter_tuning import ParameterTuner
from parameter_visualization import visualize_parameter_search_results
from logging_config import setup_pipeline_logging, get_logger

class AutoTunedFraudClusteringPipeline:
    """
    Fraud clustering pipeline with automatic parameter tuning
    """
    
    def __init__(self, verbose=False, enable_tuning=True):
        """
        Initialize auto-tuned pipeline
        
        Args:
            verbose (bool): Enable verbose logging
            enable_tuning (bool): Enable parameter tuning (set False to use defaults)
        """
        setup_pipeline_logging(verbose=verbose)
        self.logger = get_logger(__name__)
        self.enable_tuning = enable_tuning
        self.tuning_results = None
        self.best_params = None
        
    def auto_tune_parameters(self, df, feature_columns=None, quick_search=False):
        """
        Automatically find best parameters for the dataset
        
        Args:
            df (pandas.DataFrame): Input DataFrame with fraud data
            feature_columns (list): Features to use for clustering
            quick_search (bool): Use smaller search space for faster results
            
        Returns:
            dict: Best parameters found
        """
        self.logger.info("Starting automatic parameter tuning")
        
        # Default feature columns
        if feature_columns is None:
            potential_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'zip']
            feature_columns = [col for col in potential_features if col in df.columns]
            self.logger.info(f"Using default features: {feature_columns}")
        
        # Filter for fraud only
        fraud_df = df[df['is_fraud'] == 1].copy() if 'is_fraud' in df.columns else df.copy()
        self.logger.info(f"Using {len(fraud_df)} fraud transactions for parameter tuning")
        
        # Adjust search space based on dataset size
        n_samples = len(fraud_df)
        
        if quick_search or n_samples < 500:
            # Quick search for small datasets
            self.logger.info("Using quick parameter search (small dataset or quick_search=True)")
            pca_range = (3, 8)
            hdbscan_grid = {
                'min_cluster_size': [10, 20, 30] if n_samples < 200 else [20, 50, 75],
                'min_samples': [5, 10],
                'metric': ['euclidean'],
                'alpha': [1.0]
            }
            max_pca_candidates = 2
            
        elif n_samples < 1000:
            # Medium search for medium datasets
            self.logger.info("Using medium parameter search")
            pca_range = (3, 12)
            hdbscan_grid = {
                'min_cluster_size': [20, 50, 75, 100],
                'min_samples': [5, 10, 15],
                'metric': ['euclidean', 'manhattan'],
                'alpha': [1.0, 1.2]
            }
            max_pca_candidates = 3
            
        else:
            # Full search for large datasets
            self.logger.info("Using comprehensive parameter search")
            pca_range = (3, 15)
            hdbscan_grid = {
                'min_cluster_size': [20, 50, 75, 100, 150],
                'min_samples': [5, 10, 15, 20],
                'metric': ['euclidean', 'manhattan'],
                'alpha': [0.8, 1.0, 1.2]
            }
            max_pca_candidates = 5
        
        # Run parameter search
        tuner = ParameterTuner(verbose=self.logger.level <= 20)  # Verbose if INFO or DEBUG
        
        self.tuning_results = tuner.run_full_parameter_search(
            fraud_df,
            feature_columns,
            pca_range=pca_range,
            hdbscan_grid=hdbscan_grid,
            max_pca_candidates=max_pca_candidates
        )
        
        # Extract best parameters
        self.best_params = tuner.get_best_parameters(self.tuning_results)
        
        if self.best_params:
            self.logger.info("Parameter tuning completed successfully")
            self.logger.info(f"Best parameters: {self.best_params}")
        else:
            self.logger.warning("Parameter tuning failed to find valid parameters")
            # Fallback to reasonable defaults
            self.best_params = {
                'pca_components': min(8, len(feature_columns)),
                'hdbscan_min_cluster_size': max(20, n_samples // 20),
                'hdbscan_min_samples': 10,
                'hdbscan_metric': 'euclidean',
                'hdbscan_alpha': 1.0
            }
            self.logger.info(f"Using fallback parameters: {self.best_params}")
        
        return self.best_params
    
    def run_optimized_analysis(self, df, feature_columns=None, output_html_path=None, 
                             quick_search=False, save_tuning_results=True):
        """
        Run the complete optimized analysis pipeline
        
        Args:
            df (pandas.DataFrame): Input DataFrame with fraud data
            feature_columns (list): Features to use for clustering
            output_html_path (str): Path for visualization output
            quick_search (bool): Use quick parameter search
            save_tuning_results (bool): Save parameter tuning results
            
        Returns:
            dict: Complete analysis results including tuning info
        """
        start_time = time.time()
        self.logger.info("Starting Auto-Tuned Fraud Clustering Pipeline")
        self.logger.info(f"Dataset shape: {df.shape}")
        
        # Step 1: Parameter tuning (if enabled)
        if self.enable_tuning:
            self.auto_tune_parameters(df, feature_columns, quick_search)
            
            # Save tuning results if requested
            if save_tuning_results and self.tuning_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save tuning CSV
                tuning_csv = f"tuning_results_{timestamp}.csv"
                tuner = ParameterTuner()
                tuner.save_results_to_csv(self.tuning_results, tuning_csv)
                
                # Create visualizations
                visualize_parameter_search_results(self.tuning_results, f"tuning_analysis_{timestamp}")
                
                self.logger.info(f"Parameter tuning results saved with timestamp: {timestamp}")
        
        else:
            self.logger.info("Parameter tuning disabled, using default parameters")
            # Use reasonable defaults
            if feature_columns is None:
                potential_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
                feature_columns = [col for col in potential_features if col in df.columns]
            
            self.best_params = {
                'pca_components': min(8, len(feature_columns)),
                'hdbscan_min_cluster_size': 50,
                'hdbscan_min_samples': 10,
                'hdbscan_metric': 'euclidean',
                'hdbscan_alpha': 1.0
            }
        
        # Step 2: Run clustering with optimized parameters
        self.logger.info("Running clustering analysis with optimized parameters")
        
        pipeline = DataFrameFraudClusteringPipeline(
            pca_components=self.best_params['pca_components'],
            hdbscan_min_cluster_size=self.best_params['hdbscan_min_cluster_size'],
            hdbscan_min_samples=self.best_params['hdbscan_min_samples'],
            verbose=self.logger.level <= 20
        )
        
        # Run the analysis
        analysis_results = pipeline.run_full_pipeline(
            df, 
            feature_columns, 
            output_html_path
        )
        
        # Step 3: Combine results
        total_time = time.time() - start_time
        
        final_results = {
            'analysis_results': analysis_results,
            'optimized_parameters': self.best_params,
            'tuning_results': self.tuning_results if self.enable_tuning else None,
            'total_runtime': total_time,
            'tuning_enabled': self.enable_tuning
        }
        
        self.logger.info(f"Auto-tuned pipeline completed in {total_time:.2f} seconds")
        
        # Print summary
        self._print_results_summary(final_results)
        
        return final_results
    
    def _print_results_summary(self, results):
        """
        Print a summary of the analysis results
        
        Args:
            results (dict): Complete results from run_optimized_analysis
        """
        print("\n" + "=" * 60)
        print("AUTO-TUNED FRAUD CLUSTERING RESULTS SUMMARY")
        print("=" * 60)
        
        # Parameters used
        params = results['optimized_parameters']
        print(f"Optimized Parameters:")
        print(f"  PCA Components: {params['pca_components']}")
        print(f"  HDBSCAN Min Cluster Size: {params['hdbscan_min_cluster_size']}")
        print(f"  HDBSCAN Min Samples: {params['hdbscan_min_samples']}")
        print(f"  HDBSCAN Metric: {params['hdbscan_metric']}")
        
        # Clustering results
        analysis = results['analysis_results']
        if analysis and 'clustering_results' in analysis:
            cluster_summary = analysis['clustering_results']['cluster_label'].value_counts()
            n_clusters = len(cluster_summary[cluster_summary.index != -1])
            n_noise = cluster_summary.get(-1, 0)
            total_points = len(analysis['clustering_results'])
            
            print(f"\nClustering Results:")
            print(f"  Total Fraud Transactions: {total_points}")
            print(f"  Number of Clusters: {n_clusters}")
            print(f"  Noise Points: {n_noise} ({n_noise/total_points*100:.1f}%)")
            
            if n_clusters > 0:
                cluster_sizes = cluster_summary[cluster_summary.index != -1]
                print(f"  Largest Cluster: {cluster_sizes.max()} transactions")
                print(f"  Smallest Cluster: {cluster_sizes.min()} transactions")
                print(f"  Average Cluster Size: {cluster_sizes.mean():.1f} transactions")
        
        # Tuning info
        if results['tuning_enabled'] and results['tuning_results']:
            tuning = results['tuning_results']
            print(f"\nParameter Tuning:")
            print(f"  Combinations Tested: {tuning['total_combinations_tested']}")
            print(f"  Tuning Runtime: {tuning['total_runtime']:.2f} seconds")
            if tuning['best_parameters']:
                best_score = tuning['best_parameters']['composite_score']
                print(f"  Best Composite Score: {best_score:.3f}")
        
        print(f"\nTotal Pipeline Runtime: {results['total_runtime']:.2f} seconds")
        print("=" * 60)


def quick_fraud_analysis(df, output_html="auto_tuned_clusters.html", quick_search=True):
    """
    Convenience function for quick fraud analysis with auto-tuning
    
    Args:
        df (pandas.DataFrame): DataFrame with fraud data
        output_html (str): Output path for visualization
        quick_search (bool): Use quick parameter search
        
    Returns:
        dict: Analysis results
    """
    pipeline = AutoTunedFraudClusteringPipeline(verbose=True, enable_tuning=True)
    
    results = pipeline.run_optimized_analysis(
        df,
        output_html_path=output_html,
        quick_search=quick_search,
        save_tuning_results=True
    )
    
    return results


if __name__ == "__main__":
    print("Auto-Tuned Fraud Clustering Pipeline")
    print("=" * 50)
    
    print("This pipeline automatically finds optimal parameters for your fraud data.")
    print("\nUsage examples:")
    print("1. Quick analysis:")
    print("   from auto_tuned_pipeline import quick_fraud_analysis")
    print("   df = pd.read_csv('fraud_data.csv')")
    print("   results = quick_fraud_analysis(df)")
    print()
    print("2. Custom analysis:")
    print("   pipeline = AutoTunedFraudClusteringPipeline(verbose=True)")
    print("   results = pipeline.run_optimized_analysis(df, quick_search=False)")
    print()
    print("The pipeline will:")
    print("- Automatically test different parameter combinations")
    print("- Find the best parameters for your specific dataset") 
    print("- Run the clustering analysis with optimized settings")
    print("- Generate visualizations and save tuning results")
    print("- Provide a comprehensive summary of results")