"""
Example: Using the Fraud Clustering Pipeline with pandas DataFrames
This script demonstrates how to use the pipeline directly with DataFrames
without needing to save/load CSV files.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
from dataframe_pipeline import DataFrameFraudClusteringPipeline, analyze_fraud_dataframe

def create_sample_fraud_data(n_transactions=1000, fraud_rate=0.1):
    """
    Create a sample fraud dataset for demonstration
    
    Args:
        n_transactions (int): Total number of transactions
        fraud_rate (float): Proportion of fraudulent transactions
        
    Returns:
        pandas.DataFrame: Sample fraud dataset
    """
    np.random.seed(42)
    
    # Generate sample data
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
    
    # Make fraud transactions slightly different (higher amounts, different locations)
    fraud_mask = np.array(fraud_labels) == 1
    data['amt'][fraud_mask] *= np.random.uniform(1.5, 3.0, n_fraud)  # Higher amounts for fraud
    
    return pd.DataFrame(data)

def example_1_quick_analysis():
    """
    Example 1: Quick analysis using the convenience function
    """
    print("=== Example 1: Quick DataFrame Analysis ===")
    
    # Create sample data
    df = create_sample_fraud_data(n_transactions=500, fraud_rate=0.1)
    print(f"Created sample dataset: {len(df)} transactions, {df['is_fraud'].sum()} fraudulent")
    
    # Quick analysis
    results = analyze_fraud_dataframe(
        df,
        feature_columns=['amt', 'lat', 'long', 'city_pop', 'unix_time'],
        pca_components=5,
        min_cluster_size=10,
        output_html_path="example_quick_clusters.html",
        verbose=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"- Fraud transactions analyzed: {len(results['fraud_data'])}")
    print(f"- PCA components: {results['pca_results'].shape[1] - 5}")  # Subtract metadata columns
    print(f"- Clusters found: {results['cluster_stats']['n_clusters']}")
    print(f"- Noise points: {results['cluster_stats']['n_noise']} ({results['cluster_stats']['noise_ratio']*100:.1f}%)")
    if results['cluster_stats']['cluster_sizes']:
        print(f"- Cluster sizes: {results['cluster_stats']['cluster_sizes']}")
    print(f"- Visualization saved to: example_quick_clusters.html")
    
    return results

def example_2_step_by_step():
    """
    Example 2: Step-by-step analysis with custom parameters
    """
    print("\n=== Example 2: Step-by-Step Analysis ===")
    
    # Create sample data
    df = create_sample_fraud_data(n_transactions=800, fraud_rate=0.12)
    print(f"Created sample dataset: {len(df)} transactions, {df['is_fraud'].sum()} fraudulent")
    
    # Initialize pipeline
    pipeline = DataFrameFraudClusteringPipeline(
        pca_components=6,
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=5,
        umap_n_neighbors=20,
        umap_min_dist=0.2,
        verbose=False
    )
    
    # Step 1: Validate and prepare data
    if not pipeline.validate_dataframe(df):
        print("DataFrame validation failed!")
        return
    
    fraud_df = pipeline.prepare_fraud_data(df)
    print(f"Prepared {len(fraud_df)} fraud transactions")
    
    # Step 2: Run PCA
    features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    pca_df = pipeline.run_pca_analysis(features)
    print(f"PCA completed: {pca_df.shape}")
    
    # Step 3: Run clustering
    clustering_df = pipeline.run_clustering_analysis()
    print(f"Clustering completed: {clustering_df.shape}")
    
    # Step 4: Create visualization
    fig = pipeline.run_visualization("example_stepbystep_clusters.html")
    print("Visualization created and saved")
    
    # Get detailed summary
    summary = pipeline.get_cluster_summary()
    print(f"\nCluster Summary:")
    print(f"- Total transactions: {summary['total_transactions']}")
    print(f"- Number of clusters: {summary['n_clusters']}")
    print(f"- Noise points: {summary['n_noise']} ({summary['noise_ratio']*100:.1f}%)")
    print(f"- Cluster sizes: {summary['cluster_sizes']}")
    
    return {
        'pipeline': pipeline,
        'fraud_data': fraud_df,
        'pca_results': pca_df,
        'clustering_results': clustering_df,
        'summary': summary
    }

def example_3_existing_dataframe():
    """
    Example 3: Using with an existing DataFrame (like loaded from CSV)
    """
    print("\n=== Example 3: Working with Existing Data ===")
    
    # Simulate loading existing data
    print("Loading existing fraud dataset...")
    
    # You would replace this with: df = pd.read_csv('your_data.csv')
    df = pd.read_csv('fraudTrain.csv') if pd.io.common.file_exists('fraudTrain.csv') else create_sample_fraud_data(1000, 0.08)
    
    print(f"Dataset loaded: {len(df)} total transactions")
    print(f"Fraud transactions: {df['is_fraud'].sum()}")
    print(f"Available columns: {list(df.columns)}")
    
    # Analyze with available features
    available_features = []
    potential_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'zip']
    
    for feature in potential_features:
        if feature in df.columns:
            available_features.append(feature)
    
    print(f"Using features: {available_features}")
    
    # Run analysis
    results = analyze_fraud_dataframe(
        df,
        feature_columns=available_features,
        pca_components=min(8, len(available_features)),
        min_cluster_size=30,
        output_html_path="example_existing_clusters.html"
    )
    
    # Show results
    cluster_summary = {
        'total_fraud': len(results['fraud_data']),
        'clusters': results['cluster_stats']['n_clusters'],
        'noise_points': results['cluster_stats']['n_noise'],
        'largest_cluster': max(results['cluster_stats']['cluster_sizes'].values()) if results['cluster_stats']['cluster_sizes'] else 0
    }
    
    print(f"\nAnalysis Results:")
    for key, value in cluster_summary.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
    
    return results

if __name__ == "__main__":
    print("Fraud Clustering Pipeline - DataFrame Examples")
    print("=" * 50)
    
    # Run examples
    try:
        # Example 1: Quick analysis
        results1 = example_1_quick_analysis()
        
        # Example 2: Step-by-step
        results2 = example_2_step_by_step()
        
        # Example 3: Existing data
        results3 = example_3_existing_dataframe()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("- example_quick_clusters.html")
        print("- example_stepbystep_clusters.html") 
        print("- example_existing_clusters.html")
        print("\nOpen these HTML files in your browser to explore the interactive 3D visualizations!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed and the pipeline modules available.")