"""
Fraud Clustering Pipeline
Complete pipeline for PCA analysis, HDBSCAN clustering, and 3D UMAP visualization of fraudulent transactions.

This script orchestrates the entire workflow:
1. PCA analysis on fraudulent transactions
2. HDBSCAN clustering on PCA-transformed data
3. 3D UMAP visualization with interactive HTML output

Usage:
    python fraud_clustering_pipeline.py [options]
"""

import argparse
import os
import sys
import time
from datetime import datetime
import logging

# Import our custom modules
from pca_analysis import PCAAnalyzer
from hdbscan_clustering import HDBSCANClusterer
from umap_visualization import UMAPVisualizer
from logging_config import setup_pipeline_logging, get_logger, log_section_header, log_step_header, log_completion_summary, log_error_summary

class FraudClusteringPipeline:
    def __init__(self, 
                 data_path="fraudTrain.csv",
                 output_dir="results",
                 pca_components=8,
                 hdbscan_min_cluster_size=50,
                 hdbscan_min_samples=10,
                 umap_n_neighbors=15,
                 umap_min_dist=0.1):
        """
        Initialize the fraud clustering pipeline
        
        Args:
            data_path (str): Path to the fraud dataset CSV
            output_dir (str): Directory to save all output files
            pca_components (int): Number of PCA components to retain
            hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN
            hdbscan_min_samples (int): Minimum samples for HDBSCAN core points
            umap_n_neighbors (int): Number of neighbors for UMAP
            umap_min_dist (float): Minimum distance for UMAP
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # PCA parameters
        self.pca_components = pca_components
        
        # HDBSCAN parameters
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        
        # UMAP parameters
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize logger (will be set up properly in run_full_pipeline)
        self.logger = None
        
        # Define output file paths
        self.pca_output = os.path.join(self.output_dir, "pca_results.csv")
        self.clustering_output = os.path.join(self.output_dir, "clustering_results.csv")
        self.visualization_output = os.path.join(self.output_dir, "fraud_clusters_3d.html")
        
    def validate_inputs(self):
        """
        Validate input parameters and files
        
        Returns:
            bool: True if all inputs are valid
        """
        self.logger.info("Validating inputs...")
        
        # Check if data file exists
        if not os.path.exists(self.data_path):
            self.logger.error(f"Data file not found: {self.data_path}")
            return False
        
        # Validate parameters
        if self.pca_components <= 0:
            self.logger.error(f"PCA components must be positive, got {self.pca_components}")
            return False
        
        if self.hdbscan_min_cluster_size <= 0:
            self.logger.error(f"HDBSCAN min_cluster_size must be positive, got {self.hdbscan_min_cluster_size}")
            return False
        
        if self.hdbscan_min_samples <= 0:
            self.logger.error(f"HDBSCAN min_samples must be positive, got {self.hdbscan_min_samples}")
            return False
        
        self.logger.info("All inputs validated successfully")
        return True
    
    def run_pca_analysis(self):
        """
        Run PCA analysis step
        
        Returns:
            bool: True if successful
        """
        log_step_header(self.logger, 1, "PCA ANALYSIS")
        
        try:
            # Initialize PCA analyzer
            pca_analyzer = PCAAnalyzer(n_components=self.pca_components)
            
            # Define features for PCA
            features = [
                'amt',           # Transaction amount
                'lat',           # Customer latitude  
                'long',          # Customer longitude
                'city_pop',      # City population
                'unix_time',     # Transaction timestamp
                'merch_lat',     # Merchant latitude
                'merch_long',    # Merchant longitude
                'zip'            # ZIP code
            ]
            
            self.logger.info(f"Using {len(features)} features for PCA analysis")
            self.logger.info(f"Features: {features}")
            
            # Run analysis
            df_pca = pca_analyzer.run_full_analysis(
                data_path=self.data_path,
                feature_columns=features,
                output_path=self.pca_output
            )
            
            self.logger.info(f"PCA analysis completed successfully")
            self.logger.info(f"Results saved to: {self.pca_output}")
            self.logger.info(f"PCA dataset shape: {df_pca.shape}")
            
            return True
            
        except Exception as e:
            log_error_summary(self.logger, e, "PCA analysis")
            return False
    
    def run_clustering_analysis(self):
        """
        Run HDBSCAN clustering step
        
        Returns:
            bool: True if successful
        """
        log_step_header(self.logger, 2, "HDBSCAN CLUSTERING")
        
        try:
            # Check if PCA results exist
            if not os.path.exists(self.pca_output):
                self.logger.error(f"PCA results file not found: {self.pca_output}")
                return False
            
            # Initialize HDBSCAN clusterer
            clusterer = HDBSCANClusterer(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples
            )
            
            self.logger.info(f"HDBSCAN parameters:")
            self.logger.info(f"  min_cluster_size: {self.hdbscan_min_cluster_size}")
            self.logger.info(f"  min_samples: {self.hdbscan_min_samples}")
            
            # Run clustering
            df_results = clusterer.run_full_clustering(
                pca_file_path=self.pca_output,
                output_path=self.clustering_output
            )
            
            self.logger.info(f"HDBSCAN clustering completed successfully")
            self.logger.info(f"Results saved to: {self.clustering_output}")
            self.logger.info(f"Clustering dataset shape: {df_results.shape}")
            
            return True
            
        except Exception as e:
            log_error_summary(self.logger, e, "HDBSCAN clustering")
            return False
    
    def run_visualization(self):
        """
        Run 3D UMAP visualization step
        
        Returns:
            bool: True if successful
        """
        log_step_header(self.logger, 3, "3D UMAP VISUALIZATION")
        
        try:
            # Check if clustering results exist
            if not os.path.exists(self.clustering_output):
                self.logger.error(f"Clustering results file not found: {self.clustering_output}")
                return False
            
            # Initialize UMAP visualizer
            visualizer = UMAPVisualizer(
                n_components=3,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                random_state=42
            )
            
            self.logger.info(f"UMAP parameters:")
            self.logger.info(f"  n_components: 3")
            self.logger.info(f"  n_neighbors: {self.umap_n_neighbors}")
            self.logger.info(f"  min_dist: {self.umap_min_dist}")
            
            # Run visualization
            fig = visualizer.run_full_visualization(
                clustering_file_path=self.clustering_output,
                output_html_path=self.visualization_output
            )
            
            self.logger.info(f"3D UMAP visualization completed successfully")
            self.logger.info(f"Interactive HTML saved to: {self.visualization_output}")
            
            return True
            
        except Exception as e:
            log_error_summary(self.logger, e, "3D UMAP visualization")
            return False
    
    def run_full_pipeline(self, verbose=False):
        """
        Run the complete fraud clustering pipeline
        
        Args:
            verbose (bool): Enable verbose logging
            
        Returns:
            bool: True if all steps completed successfully
        """
        start_time = time.time()
        
        # Setup logging first
        setup_pipeline_logging(log_dir="logs", verbose=verbose)
        self.logger = get_logger(__name__)
        
        log_section_header(self.logger, "FRAUD CLUSTERING PIPELINE")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Data file: {self.data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
        
        # Validate inputs
        if not self.validate_inputs():
            self.logger.error("Pipeline failed during input validation")
            return False
        
        # Step 1: PCA Analysis
        if not self.run_pca_analysis():
            self.logger.error("Pipeline failed during PCA analysis")
            return False
        
        # Step 2: HDBSCAN Clustering
        if not self.run_clustering_analysis():
            self.logger.error("Pipeline failed during HDBSCAN clustering")
            return False
        
        # Step 3: 3D UMAP Visualization
        if not self.run_visualization():
            self.logger.error("Pipeline failed during 3D UMAP visualization")
            return False
        
        # Pipeline completed successfully
        end_time = time.time()
        
        output_files = [
            f"PCA Results: {self.pca_output}",
            f"Clustering Results: {self.clustering_output}",
            f"Interactive Visualization: {self.visualization_output}"
        ]
        
        log_completion_summary(self.logger, start_time, end_time, output_files)
        self.logger.info("\nTo view the interactive 3D visualization:")
        self.logger.info(f"  Open {self.visualization_output} in your web browser")
        
        return True

def main():
    """
    Main function to run the fraud clustering pipeline with command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Fraud Clustering Pipeline - PCA, HDBSCAN, and 3D UMAP Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python fraud_clustering_pipeline.py
  
  # Run with custom parameters
  python fraud_clustering_pipeline.py --data fraudTrain.csv --output results --pca-components 15
  
  # Run with custom clustering parameters
  python fraud_clustering_pipeline.py --min-cluster-size 100 --min-samples 20
        """
    )
    
    # File paths
    parser.add_argument(
        '--data',
        type=str,
        default='fraudTrain.csv',
        help='Path to the fraud dataset CSV file (default: fraudTrain.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for all results (default: results)'
    )
    
    # PCA parameters
    parser.add_argument(
        '--pca-components',
        type=int,
        default=8,
        help='Number of PCA components to retain (default: 8)'
    )
    
    # HDBSCAN parameters
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=50,
        help='Minimum size of clusters for HDBSCAN (default: 50)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum samples for HDBSCAN core points (default: 10)'
    )
    
    # UMAP parameters
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='Number of neighbors for UMAP (default: 15)'
    )
    
    parser.add_argument(
        '--umap-min-dist',
        type=float,
        default=0.1,
        help='Minimum distance for UMAP (default: 0.1)'
    )
    
    # Verbose logging argument
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = FraudClusteringPipeline(
        data_path=args.data,
        output_dir=args.output,
        pca_components=args.pca_components,
        hdbscan_min_cluster_size=args.min_cluster_size,
        hdbscan_min_samples=args.min_samples,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist
    )
    
    # Run the pipeline
    success = pipeline.run_full_pipeline(verbose=args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()