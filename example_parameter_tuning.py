"""
Example: Using Parameter Tuning for Fraud Clustering
This script demonstrates how to use the parameter tuning functionality to find optimal clustering parameters.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
from auto_tuned_pipeline import AutoTunedFraudClusteringPipeline, quick_fraud_analysis
from parameter_tuning import quick_parameter_search
from parameter_visualization import visualize_parameter_search_results

def load_sample_or_real_data():
    """
    Load fraud data - either from CSV file or create sample data
    
    Returns:
        pandas.DataFrame: Fraud dataset
    """
    # Try to load real fraud data first
    try:
        if pd.io.common.file_exists('fraudTrain.csv'):
            print("Loading real fraud data from fraudTrain.csv")
            df = pd.read_csv('fraudTrain.csv')
            print(f"Loaded {len(df)} total transactions, {df['is_fraud'].sum()} fraudulent")
            return df
        elif pd.io.common.file_exists('fraud_sample.csv'):
            print("Loading fraud data from fraud_sample.csv")
            df = pd.read_csv('fraud_sample.csv')
            print(f"Loaded {len(df)} total transactions, {df['is_fraud'].sum()} fraudulent")
            return df
    except Exception as e:
        print(f"Could not load real data: {e}")
    
    # Create sample data if no real data available
    print("Creating sample fraud data for demonstration")
    return create_sample_fraud_data()

def create_sample_fraud_data(n_transactions=2000, fraud_rate=0.08):
    """
    Create sample fraud dataset for demonstration
    
    Args:
        n_transactions (int): Total number of transactions
        fraud_rate (float): Proportion of fraudulent transactions
        
    Returns:
        pandas.DataFrame: Sample fraud dataset
    """
    np.random.seed(42)
    
    # Generate sample data with realistic fraud patterns
    data = {
        'transaction_id': range(n_transactions),
        'amt': np.random.lognormal(mean=3, sigma=1.5, size=n_transactions),
        'lat': np.random.uniform(25, 49, n_transactions),  # US latitude range
        'long': np.random.uniform(-125, -66, n_transactions),  # US longitude range
        'city_pop': np.random.randint(1000, 1000000, n_transactions),
        'unix_time': np.random.randint(1577836800, 1609459200, n_transactions),  # 2020 timestamps
        'merch_lat': np.random.uniform(25, 49, n_transactions),
        'merch_long': np.random.uniform(-125, -66, n_transactions),
        'zip': np.random.randint(10000, 99999, n_transactions),
        'category': np.random.choice(['grocery', 'gas', 'retail', 'restaurant', 'online'], n_transactions)
    }
    
    # Create fraud labels
    n_fraud = int(n_transactions * fraud_rate)
    fraud_labels = [1] * n_fraud + [0] * (n_transactions - n_fraud)
    np.random.shuffle(fraud_labels)
    data['is_fraud'] = fraud_labels
    
    # Make fraud transactions slightly different (create patterns for clustering)
    fraud_mask = np.array(fraud_labels) == 1
    fraud_indices = np.where(fraud_mask)[0]
    
    # Create 3 distinct fraud patterns
    pattern1 = fraud_indices[:len(fraud_indices)//3]  # High amount pattern
    pattern2 = fraud_indices[len(fraud_indices)//3:2*len(fraud_indices)//3]  # Geographic pattern  
    pattern3 = fraud_indices[2*len(fraud_indices)//3:]  # Time pattern
    
    # Pattern 1: High amounts in specific locations
    data['amt'][pattern1] *= np.random.uniform(2.0, 5.0, len(pattern1))
    data['lat'][pattern1] = np.random.normal(40.7, 0.5, len(pattern1))  # Around NYC
    data['long'][pattern1] = np.random.normal(-74.0, 0.5, len(pattern1))
    
    # Pattern 2: Normal amounts but clustered geographically  
    data['lat'][pattern2] = np.random.normal(34.0, 1.0, len(pattern2))  # Around LA
    data['long'][pattern2] = np.random.normal(-118.0, 1.0, len(pattern2))
    
    # Pattern 3: Time-based pattern (late night transactions)
    late_night_times = np.random.randint(1609459200-86400, 1609459200, len(pattern3))  # Last day of 2020
    data['unix_time'][pattern3] = late_night_times
    
    df = pd.DataFrame(data)
    print(f"Created sample dataset: {len(df)} transactions, {df['is_fraud'].sum()} fraudulent")
    print("Sample includes 3 distinct fraud patterns for clustering demonstration")
    
    return df

def example_1_quick_auto_tuned_analysis():
    """
    Example 1: Quick analysis with automatic parameter tuning
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Quick Auto-Tuned Analysis")
    print("=" * 60)
    
    # Load data
    df = load_sample_or_real_data()
    
    # Run quick analysis with auto-tuning
    print("\nRunning quick auto-tuned analysis...")
    results = quick_fraud_analysis(
        df, 
        output_html="example1_auto_tuned_clusters.html",
        quick_search=True
    )
    
    print("\nQuick analysis complete!")
    print("Files generated:")
    print("- example1_auto_tuned_clusters.html (3D visualization)")
    print("- tuning_results_*.csv (parameter search results)")
    print("- tuning_analysis_* files (parameter analysis plots)")
    
    return results

def example_2_comprehensive_parameter_search():
    """
    Example 2: Comprehensive parameter search with detailed analysis
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comprehensive Parameter Search")
    print("=" * 60)
    
    # Load data
    df = load_sample_or_real_data()
    
    # Run comprehensive parameter search
    print("\nRunning comprehensive parameter search...")
    pipeline = AutoTunedFraudClusteringPipeline(verbose=True, enable_tuning=True)
    
    results = pipeline.run_optimized_analysis(
        df,
        feature_columns=['amt', 'lat', 'long', 'city_pop', 'unix_time'],
        output_html_path="example2_comprehensive_clusters.html",
        quick_search=False,  # Full search
        save_tuning_results=True
    )
    
    print("\nComprehensive analysis complete!")
    print("Files generated:")
    print("- example2_comprehensive_clusters.html (3D visualization)")
    print("- Parameter tuning visualizations and CSV results")
    
    return results

def example_3_manual_parameter_tuning():
    """
    Example 3: Manual parameter tuning with custom grid
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Manual Parameter Tuning")
    print("=" * 60)
    
    # Load data
    df = load_sample_or_real_data()
    
    # Run manual parameter search with custom settings
    print("\nRunning manual parameter search with custom grid...")
    
    from parameter_tuning import ParameterTuner
    
    # Custom parameter grid
    custom_grid = {
        'min_cluster_size': [15, 25, 40, 60],
        'min_samples': [3, 6, 10, 15],
        'metric': ['euclidean', 'manhattan'],
        'alpha': [0.9, 1.0, 1.1]
    }
    
    # Features to use
    feature_columns = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Filter fraud data
    fraud_df = df[df['is_fraud'] == 1].copy()
    
    # Run tuning
    tuner = ParameterTuner(verbose=True)
    search_results = tuner.run_full_parameter_search(
        fraud_df,
        available_features,
        pca_range=(4, 10),
        hdbscan_grid=custom_grid,
        max_pca_candidates=3
    )
    
    # Save and visualize results
    tuner.save_results_to_csv(search_results, "example3_manual_search_results.csv")
    visualize_parameter_search_results(search_results, "example3_manual_analysis")
    
    # Get best parameters and run analysis
    best_params = tuner.get_best_parameters(search_results)
    if best_params:
        print(f"\nBest parameters found: {best_params}")
        
        # Run clustering with best parameters
        from dataframe_pipeline import DataFrameFraudClusteringPipeline
        
        optimized_pipeline = DataFrameFraudClusteringPipeline(
            pca_components=best_params['pca_components'],
            hdbscan_min_cluster_size=best_params['hdbscan_min_cluster_size'],
            hdbscan_min_samples=best_params['hdbscan_min_samples'],
            verbose=True
        )
        
        final_results = optimized_pipeline.run_full_pipeline(
            df, available_features, "example3_manual_optimized_clusters.html"
        )
        
        print("\nManual parameter tuning complete!")
        print("Files generated:")
        print("- example3_manual_search_results.csv")
        print("- example3_manual_analysis_* (parameter analysis)")
        print("- example3_manual_optimized_clusters.html")
        
        return final_results
    
    return None

def compare_tuned_vs_default():
    """
    Example 4: Compare auto-tuned vs default parameters
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Tuned vs Default Parameters Comparison")
    print("=" * 60)
    
    df = load_sample_or_real_data()
    
    # Run with default parameters
    print("\n1. Running with default parameters...")
    default_pipeline = AutoTunedFraudClusteringPipeline(verbose=False, enable_tuning=False)
    default_results = default_pipeline.run_optimized_analysis(
        df, output_html_path="example4_default_clusters.html"
    )
    
    # Run with auto-tuned parameters  
    print("\n2. Running with auto-tuned parameters...")
    tuned_pipeline = AutoTunedFraudClusteringPipeline(verbose=False, enable_tuning=True)
    tuned_results = tuned_pipeline.run_optimized_analysis(
        df, output_html_path="example4_tuned_clusters.html", quick_search=True
    )
    
    # Compare results
    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)
    
    def get_cluster_stats(results):
        if results['analysis_results'] and 'clustering_results' in results['analysis_results']:
            cluster_summary = results['analysis_results']['clustering_results']['cluster_label'].value_counts()
            n_clusters = len(cluster_summary[cluster_summary.index != -1])
            n_noise = cluster_summary.get(-1, 0)
            total = len(results['analysis_results']['clustering_results'])
            return n_clusters, n_noise, total
        return 0, 0, 0
    
    default_clusters, default_noise, default_total = get_cluster_stats(default_results)
    tuned_clusters, tuned_noise, tuned_total = get_cluster_stats(tuned_results)
    
    print(f"DEFAULT PARAMETERS:")
    print(f"  PCA Components: {default_results['optimized_parameters']['pca_components']}")
    print(f"  Min Cluster Size: {default_results['optimized_parameters']['hdbscan_min_cluster_size']}")
    print(f"  Clusters Found: {default_clusters}")
    print(f"  Noise Ratio: {default_noise/default_total*100:.1f}%" if default_total > 0 else "  Noise Ratio: N/A")
    
    print(f"\nTUNED PARAMETERS:")
    print(f"  PCA Components: {tuned_results['optimized_parameters']['pca_components']}")
    print(f"  Min Cluster Size: {tuned_results['optimized_parameters']['hdbscan_min_cluster_size']}")
    print(f"  Clusters Found: {tuned_clusters}")
    print(f"  Noise Ratio: {tuned_noise/tuned_total*100:.1f}%" if tuned_total > 0 else "  Noise Ratio: N/A")
    
    if tuned_results['tuning_results'] and tuned_results['tuning_results']['best_parameters']:
        best_score = tuned_results['tuning_results']['best_parameters']['composite_score']
        print(f"  Composite Score: {best_score:.3f}")
    
    print("\nFiles generated:")
    print("- example4_default_clusters.html")
    print("- example4_tuned_clusters.html")
    
    return default_results, tuned_results

def main():
    """
    Run all parameter tuning examples
    """
    print("FRAUD CLUSTERING PARAMETER TUNING EXAMPLES")
    print("=" * 60)
    print("This script demonstrates different ways to use parameter tuning")
    print("to find optimal clustering parameters for your fraud data.")
    
    try:
        # Example 1: Quick auto-tuned analysis
        example_1_quick_auto_tuned_analysis()
        
        # Example 2: Comprehensive search
        example_2_comprehensive_parameter_search()
        
        # Example 3: Manual parameter tuning
        example_3_manual_parameter_tuning()
        
        # Example 4: Compare tuned vs default
        compare_tuned_vs_default()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- example1_auto_tuned_clusters.html")
        print("- example2_comprehensive_clusters.html") 
        print("- example3_manual_optimized_clusters.html")
        print("- example4_default_clusters.html")
        print("- example4_tuned_clusters.html")
        print("- Various parameter analysis files (.png, .csv)")
        print("\nOpen the HTML files in your browser to explore the interactive 3D visualizations!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have all required dependencies installed.")

if __name__ == "__main__":
    main()