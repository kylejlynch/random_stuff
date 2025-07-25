import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Clustering and visualization
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FraudClusterTracker:
    def __init__(self, spark_session=None):
        self.spark = spark_session or SparkSession.builder.appName("FraudClustering").getOrCreate()
        self.cluster_history = {}
        self.umap_reducer = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df, date_col='transaction_date', feature_cols=None):
        """
        Preprocess fraud data with PySpark
        """
        if feature_cols is None:
            # Default fraud-relevant features
            feature_cols = [
                'amount', 'hour_of_day', 'day_of_week', 
                'merchant_risk_score', 'velocity_1h', 'velocity_24h',
                'amount_zscore', 'freq_merchant_1d', 'geographic_risk'
            ]
        
        # Add time-based features
        df = df.withColumn('hour_of_day', hour(col(date_col))) \
               .withColumn('day_of_week', dayofweek(col(date_col))) \
               .withColumn('transaction_date_only', to_date(col(date_col)))
        
        # Calculate velocity features (example - adapt to your schema)
        window_1h = Window.partitionBy('user_id').orderBy('transaction_timestamp') \
                          .rangeBetween(-3600, 0)  # 1 hour in seconds
        window_24h = Window.partitionBy('user_id').orderBy('transaction_timestamp') \
                           .rangeBetween(-86400, 0)  # 24 hours in seconds
        
        df = df.withColumn('velocity_1h', count('*').over(window_1h)) \
               .withColumn('velocity_24h', count('*').over(window_24h))
        
        # Amount z-score by user
        user_stats = Window.partitionBy('user_id')
        df = df.withColumn('user_avg_amount', avg('amount').over(user_stats)) \
               .withColumn('user_std_amount', stddev('amount').over(user_stats)) \
               .withColumn('amount_zscore', 
                          (col('amount') - col('user_avg_amount')) / col('user_std_amount'))
        
        return df.select(['transaction_id', 'transaction_date_only'] + feature_cols + ['is_fraud'])
    
    def extract_daily_features(self, df, target_date):
        """
        Extract features for a specific date
        """
        daily_df = df.filter(col('transaction_date_only') == target_date)
        
        # Convert to Pandas for HDBSCAN (PySpark doesn't have native HDBSCAN)
        pandas_df = daily_df.toPandas()
        
        if len(pandas_df) == 0:
            return None, None
            
        feature_cols = [c for c in pandas_df.columns if c not in 
                       ['transaction_id', 'transaction_date_only', 'is_fraud']]
        
        features = pandas_df[feature_cols].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, pandas_df
    
    def perform_clustering(self, features, min_cluster_size=10, min_samples=5):
        """
        Perform HDBSCAN clustering
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.1
        )
        
        cluster_labels = clusterer.fit_predict(features)
        
        return clusterer, cluster_labels
    
    def create_umap_embedding(self, features, n_neighbors=15, min_dist=0.1):
        """
        Create UMAP embedding for visualization
        """
        if self.umap_reducer is None:
            self.umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='euclidean',
                random_state=42
            )
            embedding = self.umap_reducer.fit_transform(features)
        else:
            # Use existing reducer for consistency across time
            embedding = self.umap_reducer.transform(features)
            
        return embedding
    
    def match_clusters_over_time(self, current_clusters, previous_clusters, 
                                current_features, previous_features, threshold=0.7):
        """
        Match clusters between time periods using centroid similarity
        """
        if len(previous_clusters) == 0:
            return {}, list(range(len(current_clusters)))
        
        # Calculate centroids
        current_centroids = []
        previous_centroids = []
        
        for cluster_id in current_clusters:
            if cluster_id != -1:  # Exclude noise
                mask = current_clusters == cluster_id
                current_centroids.append(np.mean(current_features[mask], axis=0))
        
        for cluster_id in previous_clusters:
            if cluster_id != -1:
                mask = previous_clusters == cluster_id
                previous_centroids.append(np.mean(previous_features[mask], axis=0))
        
        if len(current_centroids) == 0 or len(previous_centroids) == 0:
            return {}, list(range(len(current_centroids)))
        
        # Calculate distance matrix
        distances = cdist(current_centroids, previous_centroids, metric='euclidean')
        
        # Use Hungarian algorithm for optimal matching
        current_indices, previous_indices = linear_sum_assignment(distances)
        
        matches = {}
        new_clusters = []
        
        for i, curr_idx in enumerate(current_indices):
            prev_idx = previous_indices[i]
            if distances[curr_idx, prev_idx] < threshold:
                matches[curr_idx] = prev_idx
            else:
                new_clusters.append(curr_idx)
        
        # Add unmatched current clusters as new
        all_current = set(range(len(current_centroids)))
        matched_current = set(matches.keys())
        new_clusters.extend(list(all_current - matched_current))
        
        return matches, new_clusters
    
    def analyze_daily_fraud(self, df, date_range=None, min_cluster_size=10):
        """
        Analyze fraud patterns over a date range
        """
        if date_range is None:
            # Default to last 7 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            date_range = pd.date_range(start_date, end_date, freq='D')
        
        results = []
        previous_clusters = np.array([])
        previous_features = np.array([]).reshape(0, -1)
        
        for date in date_range:
            print(f"Analyzing {date}")
            
            # Extract features for this date
            features, data = self.extract_daily_features(df, date)
            
            if features is None:
                continue
            
            # Perform clustering
            clusterer, cluster_labels = self.perform_clustering(features, min_cluster_size)
            
            # Create UMAP embedding
            embedding = self.create_umap_embedding(features)
            
            # Match clusters with previous day
            matches, new_clusters = self.match_clusters_over_time(
                cluster_labels, previous_clusters, features, previous_features
            )
            
            # Analyze clusters
            cluster_stats = self.analyze_clusters(data, cluster_labels, embedding)
            
            # Store results
            result = {
                'date': date,
                'n_transactions': len(data),
                'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
                'n_noise': np.sum(cluster_labels == -1),
                'noise_ratio': np.sum(cluster_labels == -1) / len(cluster_labels),
                'new_clusters': new_clusters,
                'cluster_matches': matches,
                'cluster_stats': cluster_stats,
                'embedding': embedding,
                'cluster_labels': cluster_labels,
                'features': features,
                'data': data
            }
            
            results.append(result)
            
            # Update for next iteration
            previous_clusters = cluster_labels.copy()
            previous_features = features.copy()
        
        return results
    
    def analyze_clusters(self, data, cluster_labels, embedding):
        """
        Analyze individual cluster characteristics
        """
        cluster_stats = []
        
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_data = data[mask]
            
            if cluster_id == -1:  # Noise cluster
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster_{cluster_id}"
            
            stats = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'size': np.sum(mask),
                'fraud_rate': cluster_data['is_fraud'].mean() if 'is_fraud' in cluster_data.columns else 0,
                'avg_amount': cluster_data['amount'].mean() if 'amount' in cluster_data.columns else 0,
                'centroid': np.mean(embedding[mask], axis=0),
                'is_noise': cluster_id == -1
            }
            
            cluster_stats.append(stats)
        
        return cluster_stats
    
    def create_evolution_visualization(self, results):
        """
        Create comprehensive visualization of fraud evolution
        """
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Evolution Over Time', 'UMAP Visualization', 
                          'Fraud Rate by Cluster', 'Noise Ratio Trend'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Cluster count evolution
        dates = [r['date'] for r in results]
        n_clusters = [r['n_clusters'] for r in results]
        noise_ratios = [r['noise_ratio'] for r in results]
        
        fig.add_trace(
            go.Scatter(x=dates, y=n_clusters, mode='lines+markers', 
                      name='Number of Clusters', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Latest UMAP visualization
        if results:
            latest = results[-1]
            embedding = latest['embedding']
            labels = latest['cluster_labels']
            fraud_flags = latest['data']['is_fraud'].values if 'is_fraud' in latest['data'].columns else np.zeros(len(labels))
            
            # Color by cluster, shape by fraud
            colors = labels
            symbols = ['circle' if f == 0 else 'diamond' for f in fraud_flags]
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0], y=embedding[:, 1],
                    mode='markers',
                    marker=dict(color=colors, symbol=symbols, size=6),
                    name='Latest Transactions',
                    text=[f'Cluster: {l}, Fraud: {f}' for l, f in zip(labels, fraud_flags)]
                ),
                row=2, col=1
            )
        
        # 3. Fraud rate analysis
        fraud_rates = []
        for result in results:
            for stat in result['cluster_stats']:
                if not stat['is_noise'] and stat['size'] > 5:  # Ignore small clusters
                    fraud_rates.append({
                        'date': result['date'],
                        'cluster': stat['cluster_name'],
                        'fraud_rate': stat['fraud_rate'],
                        'size': stat['size']
                    })
        
        if fraud_rates:
            fraud_df = pd.DataFrame(fraud_rates)
            for cluster in fraud_df['cluster'].unique():
                cluster_data = fraud_df[fraud_df['cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(x=cluster_data['date'], y=cluster_data['fraud_rate'],
                              mode='lines+markers', name=f'{cluster} Fraud Rate'),
                    row=2, col=2
                )
        
        # 4. Noise ratio trend
        fig.add_trace(
            go.Scatter(x=dates, y=noise_ratios, mode='lines+markers',
                      name='Noise Ratio', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Fraud Pattern Evolution Analysis")
        
        return fig
    
    def detect_anomalies(self, results, alert_thresholds=None):
        """
        Detect potential new fraud schemes based on cluster evolution
        """
        if alert_thresholds is None:
            alert_thresholds = {
                'new_cluster_min_size': 10,
                'high_fraud_rate': 0.8,
                'sudden_noise_increase': 0.3,
                'rapid_cluster_growth': 2.0
            }
        
        alerts = []
        
        for i, result in enumerate(results):
            date = result['date']
            
            # Alert 1: New large clusters
            for cluster_id in result['new_clusters']:
                cluster_stats = [s for s in result['cluster_stats'] if s['cluster_id'] == cluster_id][0]
                if cluster_stats['size'] >= alert_thresholds['new_cluster_min_size']:
                    alerts.append({
                        'date': date,
                        'type': 'New Large Cluster',
                        'description': f"New cluster with {cluster_stats['size']} transactions",
                        'severity': 'Medium',
                        'cluster_id': cluster_id
                    })
            
            # Alert 2: High fraud rate clusters
            for stat in result['cluster_stats']:
                if (not stat['is_noise'] and 
                    stat['fraud_rate'] >= alert_thresholds['high_fraud_rate'] and
                    stat['size'] >= 5):
                    alerts.append({
                        'date': date,
                        'type': 'High Fraud Cluster',
                        'description': f"Cluster {stat['cluster_name']} has {stat['fraud_rate']:.1%} fraud rate",
                        'severity': 'High',
                        'cluster_id': stat['cluster_id']
                    })
            
            # Alert 3: Sudden noise increase
            if (i > 0 and 
                result['noise_ratio'] - results[i-1]['noise_ratio'] > alert_thresholds['sudden_noise_increase']):
                alerts.append({
                    'date': date,
                    'type': 'Noise Spike',
                    'description': f"Noise ratio increased by {(result['noise_ratio'] - results[i-1]['noise_ratio']):.1%}",
                    'severity': 'Medium',
                    'cluster_id': -1
                })
        
        return alerts

# Example usage and testing
def example_usage():
    """
    Example of how to use the FraudClusterTracker
    """
    # Initialize Spark session
    spark = SparkSession.builder.appName("FraudAnalysis").getOrCreate()
    
    # Initialize tracker
    tracker = FraudClusterTracker(spark)
    
    # Sample data creation (replace with your actual data loading)
    # This would typically be: df = spark.read.table("your_fraud_table")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = []
    for i in range(7):  # 7 days of data
        date = datetime.now().date() - timedelta(days=6-i)
        
        for j in range(n_samples // 7):
            sample_data.append({
                'transaction_id': f'txn_{i}_{j}',
                'transaction_date': datetime.combine(date, datetime.min.time()),
                'user_id': f'user_{np.random.randint(1, 100)}',
                'amount': np.random.lognormal(3, 1),
                'merchant_risk_score': np.random.uniform(0, 1),
                'geographic_risk': np.random.uniform(0, 1),
                'is_fraud': np.random.choice([0, 1], p=[0.95, 0.05])
            })
    
    # Convert to Spark DataFrame
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("transaction_date", TimestampType(), True),
        StructField("user_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("merchant_risk_score", DoubleType(), True),
        StructField("geographic_risk", DoubleType(), True),
        StructField("is_fraud", IntegerType(), True)
    ])
    
    df = spark.createDataFrame(sample_data, schema)
    
    # Preprocess data
    processed_df = tracker.preprocess_data(df)
    
    # Analyze fraud patterns over time
    results = tracker.analyze_daily_fraud(processed_df)
    
    # Create visualizations
    fig = tracker.create_evolution_visualization(results)
    fig.show()
    
    # Detect anomalies
    alerts = tracker.detect_anomalies(results)
    
    print("\n=== FRAUD ALERTS ===")
    for alert in alerts:
        print(f"{alert['date']} - {alert['type']}: {alert['description']} (Severity: {alert['severity']})")
    
    return tracker, results, alerts

if __name__ == "__main__":
    # Run example
    tracker, results, alerts = example_usage()
