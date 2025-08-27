"""
Alert Coverage Analysis Module
This module analyzes alert performance to identify weak "pockets" where fraud detection is missing cases.
Uses clustering, density analysis, and statistical methods to find alert gaps.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Clustering and analysis
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from logging_config import get_logger

class AlertCoverageAnalyzer:
    """
    Analyzer to find weak pockets in fraud alerting
    """
    
    def __init__(self, verbose=False):
        """
        Initialize alert coverage analyzer
        
        Args:
            verbose (bool): Enable verbose logging
        """
        self.logger = get_logger(__name__)
        if verbose:
            self.logger.setLevel(10)  # DEBUG level
        
        self.fraud_df = None
        self.cluster_results = None
        self.umap_results = None
        self.alert_gaps = {}
        
    def load_clustering_results(self, clustering_df, umap_embedding=None):
        """
        Load clustering results with alert information
        
        Args:
            clustering_df (pandas.DataFrame): Clustering results with 'alerted' column
            umap_embedding (numpy.array): UMAP embedding coordinates (optional)
        """
        self.fraud_df = clustering_df.copy()
        
        # Validate required columns
        required_cols = ['alerted', 'cluster_label']
        missing_cols = [col for col in required_cols if col not in self.fraud_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add UMAP coordinates if provided
        if umap_embedding is not None:
            self.fraud_df['umap_x'] = umap_embedding[:, 0]
            self.fraud_df['umap_y'] = umap_embedding[:, 1] 
            if umap_embedding.shape[1] > 2:
                self.fraud_df['umap_z'] = umap_embedding[:, 2]
        
        self.logger.info(f"Loaded {len(self.fraud_df)} fraud transactions for alert analysis")
        self.logger.info(f"Alert rate: {self.fraud_df['alerted'].mean():.3f}")
        
    def analyze_cluster_alert_performance(self):
        """
        Analyze alert performance by cluster
        
        Returns:
            pandas.DataFrame: Cluster-level alert statistics
        """
        self.logger.info("Analyzing alert performance by cluster")
        
        # Calculate cluster-level statistics
        cluster_stats = []
        
        for cluster_id in self.fraud_df['cluster_label'].unique():
            if cluster_id == -1:  # Skip noise points for now
                continue
                
            cluster_data = self.fraud_df[self.fraud_df['cluster_label'] == cluster_id]
            
            stats_dict = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'alert_count': cluster_data['alerted'].sum(),
                'alert_rate': cluster_data['alerted'].mean(),
                'missed_count': len(cluster_data) - cluster_data['alerted'].sum(),
                'missed_rate': 1 - cluster_data['alerted'].mean()
            }
            
            # Add feature statistics if available
            if 'amt' in cluster_data.columns:
                stats_dict.update({
                    'avg_amount': cluster_data['amt'].mean(),
                    'median_amount': cluster_data['amt'].median(),
                    'amount_std': cluster_data['amt'].std()
                })
            
            # Add return reason statistics if available
            if 'return_reason' in cluster_data.columns:
                most_common_reason = cluster_data['return_reason'].mode()
                stats_dict['primary_fraud_type'] = most_common_reason.iloc[0] if len(most_common_reason) > 0 else 'unknown'
                stats_dict['fraud_type_diversity'] = cluster_data['return_reason'].nunique()
            
            cluster_stats.append(stats_dict)
        
        # Handle noise points separately
        noise_data = self.fraud_df[self.fraud_df['cluster_label'] == -1]
        if len(noise_data) > 0:
            noise_stats = {
                'cluster_id': -1,
                'size': len(noise_data),
                'alert_count': noise_data['alerted'].sum(),
                'alert_rate': noise_data['alerted'].mean(),
                'missed_count': len(noise_data) - noise_data['alerted'].sum(),
                'missed_rate': 1 - noise_data['alerted'].mean()
            }
            
            if 'amt' in noise_data.columns:
                noise_stats.update({
                    'avg_amount': noise_data['amt'].mean(),
                    'median_amount': noise_data['amt'].median(), 
                    'amount_std': noise_data['amt'].std()
                })
            
            if 'return_reason' in noise_data.columns:
                most_common_reason = noise_data['return_reason'].mode()
                noise_stats['primary_fraud_type'] = most_common_reason.iloc[0] if len(most_common_reason) > 0 else 'unknown'
                noise_stats['fraud_type_diversity'] = noise_data['return_reason'].nunique()
            
            cluster_stats.append(noise_stats)
        
        self.cluster_results = pd.DataFrame(cluster_stats)
        
        # Calculate alert gap score (higher = worse coverage)
        overall_alert_rate = self.fraud_df['alerted'].mean()
        self.cluster_results['alert_gap_score'] = (
            (overall_alert_rate - self.cluster_results['alert_rate']) * 
            self.cluster_results['size'] / len(self.fraud_df)
        )
        
        # Sort by alert gap score (worst first)
        self.cluster_results = self.cluster_results.sort_values('alert_gap_score', ascending=False)
        
        self.logger.info(f"Analyzed {len(self.cluster_results)} clusters")
        
        return self.cluster_results
    
    def find_density_based_alert_gaps(self, n_neighbors=20, contamination=0.1):
        """
        Find alert gaps using density-based analysis in feature space
        
        Args:
            n_neighbors (int): Number of neighbors for density analysis
            contamination (float): Expected proportion of outliers
            
        Returns:
            pandas.DataFrame: Transactions with density-based alert gap scores
        """
        self.logger.info("Finding density-based alert gaps")
        
        # Use PCA features for density analysis
        pca_cols = [col for col in self.fraud_df.columns if col.startswith('PC')]
        if len(pca_cols) == 0:
            self.logger.warning("No PCA features found, using available numerical features")
            numerical_cols = self.fraud_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col not in ['alerted', 'cluster_label']][:10]
        else:
            feature_cols = pca_cols
        
        if len(feature_cols) == 0:
            self.logger.error("No suitable features found for density analysis")
            return pd.DataFrame()
        
        X = self.fraud_df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Local Outlier Factor for density analysis
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_scores = lof.fit_predict(X_scaled)
        density_scores = -lof.negative_outlier_factor_
        
        # Find neighborhoods with low alert rates
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_scaled)
        
        alert_gap_scores = []
        
        for i in range(len(X_scaled)):
            # Find neighbors
            distances, indices = nn.kneighbors([X_scaled[i]])
            neighbor_indices = indices[0]
            
            # Calculate neighborhood alert rate
            neighborhood_alerts = self.fraud_df.iloc[neighbor_indices]['alerted'].mean()
            current_alert = self.fraud_df.iloc[i]['alerted']
            
            # Score combines density and alert gap
            # High score = dense area with low alerts (concerning)
            density_score = density_scores[i]
            alert_gap = max(0, self.fraud_df['alerted'].mean() - neighborhood_alerts)
            
            # Normalize density score (higher = more normal density)
            normalized_density = min(density_score / np.percentile(density_scores, 95), 1.0)
            
            # Alert gap score: high density + low alerts = high score
            gap_score = normalized_density * alert_gap * (1 - current_alert)  # Only flag unalerted cases
            alert_gap_scores.append(gap_score)
        
        # Add results to dataframe
        analysis_df = self.fraud_df.copy()
        analysis_df['density_score'] = density_scores
        analysis_df['alert_gap_score'] = alert_gap_scores
        analysis_df['is_density_outlier'] = outlier_scores == -1
        
        # Sort by alert gap score
        analysis_df = analysis_df.sort_values('alert_gap_score', ascending=False)
        
        self.logger.info(f"Found {sum(analysis_df['alert_gap_score'] > 0)} potential alert gaps using density analysis")
        
        return analysis_df
    
    def find_umap_alert_gaps(self, grid_size=20, min_density=5):
        """
        Find alert gaps in UMAP space using grid-based analysis
        
        Args:
            grid_size (int): Size of the grid for UMAP space division
            min_density (int): Minimum points per grid cell to analyze
            
        Returns:
            dict: Grid analysis results
        """
        if 'umap_x' not in self.fraud_df.columns or 'umap_y' not in self.fraud_df.columns:
            self.logger.warning("UMAP coordinates not available for gap analysis")
            return {}
        
        self.logger.info("Finding alert gaps in UMAP space")
        
        # Create grid over UMAP space
        x_min, x_max = self.fraud_df['umap_x'].min(), self.fraud_df['umap_x'].max()
        y_min, y_max = self.fraud_df['umap_y'].min(), self.fraud_df['umap_y'].max()
        
        x_bins = np.linspace(x_min, y_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        grid_results = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Define grid cell boundaries
                x_left, x_right = x_bins[i], x_bins[i+1]
                y_bottom, y_top = y_bins[j], y_bins[j+1]
                
                # Find points in this grid cell
                mask = (
                    (self.fraud_df['umap_x'] >= x_left) & 
                    (self.fraud_df['umap_x'] < x_right) &
                    (self.fraud_df['umap_y'] >= y_bottom) & 
                    (self.fraud_df['umap_y'] < y_top)
                )
                
                cell_data = self.fraud_df[mask]
                
                if len(cell_data) >= min_density:
                    cell_stats = {
                        'grid_x': i,
                        'grid_y': j,
                        'x_center': (x_left + x_right) / 2,
                        'y_center': (y_bottom + y_top) / 2,
                        'x_left': x_left,
                        'x_right': x_right,
                        'y_bottom': y_bottom,
                        'y_top': y_top,
                        'count': len(cell_data),
                        'alert_count': cell_data['alerted'].sum(),
                        'alert_rate': cell_data['alerted'].mean(),
                        'missed_count': len(cell_data) - cell_data['alerted'].sum()
                    }
                    
                    # Add feature statistics
                    if 'amt' in cell_data.columns:
                        cell_stats['avg_amount'] = cell_data['amt'].mean()
                    
                    if 'return_reason' in cell_data.columns:
                        most_common = cell_data['return_reason'].mode()
                        cell_stats['primary_fraud_type'] = most_common.iloc[0] if len(most_common) > 0 else 'unknown'
                    
                    grid_results.append(cell_stats)
        
        grid_df = pd.DataFrame(grid_results)
        
        if len(grid_df) > 0:
            # Calculate alert gap scores
            overall_alert_rate = self.fraud_df['alerted'].mean()
            grid_df['alert_gap'] = overall_alert_rate - grid_df['alert_rate']
            grid_df['weighted_gap_score'] = grid_df['alert_gap'] * grid_df['count']
            
            # Sort by weighted gap score
            grid_df = grid_df.sort_values('weighted_gap_score', ascending=False)
            
            self.logger.info(f"Analyzed {len(grid_df)} UMAP grid cells")
        
        return grid_df
    
    def identify_high_risk_missed_transactions(self, top_n=100):
        """
        Identify the highest risk missed transactions
        
        Args:
            top_n (int): Number of top missed transactions to return
            
        Returns:
            pandas.DataFrame: High-risk missed transactions
        """
        self.logger.info(f"Identifying top {top_n} high-risk missed transactions")
        
        # Filter to unalerted transactions only
        missed_transactions = self.fraud_df[self.fraud_df['alerted'] == 0].copy()
        
        if len(missed_transactions) == 0:
            self.logger.info("No missed transactions found (100% alert rate)")
            return pd.DataFrame()
        
        # Calculate risk scores based on multiple factors
        risk_scores = []
        
        for _, transaction in missed_transactions.iterrows():
            risk_score = 0
            
            # Factor 1: Cluster alert rate (lower cluster alert rate = higher risk)
            cluster_id = transaction['cluster_label']
            if cluster_id in self.cluster_results['cluster_id'].values:
                cluster_alert_rate = self.cluster_results[
                    self.cluster_results['cluster_id'] == cluster_id
                ]['alert_rate'].iloc[0]
                risk_score += (1 - cluster_alert_rate) * 0.3
            
            # Factor 2: Transaction amount (higher amount = higher risk)
            if 'amt' in transaction and pd.notna(transaction['amt']):
                amount_percentile = stats.percentileofscore(self.fraud_df['amt'], transaction['amt']) / 100
                risk_score += amount_percentile * 0.2
            
            # Factor 3: Density-based outlier score (if available)
            if hasattr(self, 'density_results') and 'alert_gap_score' in self.density_results.columns:
                # Find this transaction in density results
                matching_idx = self.density_results.index[
                    self.density_results.index == transaction.name
                ]
                if len(matching_idx) > 0:
                    density_gap_score = self.density_results.loc[matching_idx[0], 'alert_gap_score']
                    risk_score += min(density_gap_score, 1.0) * 0.3
            
            # Factor 4: Fraud type risk (if available)
            if 'return_reason' in transaction and pd.notna(transaction['return_reason']):
                high_risk_types = ['identity_theft', 'account_takeover', 'synthetic_identity']
                if transaction['return_reason'] in high_risk_types:
                    risk_score += 0.2
            
            risk_scores.append(risk_score)
        
        missed_transactions['risk_score'] = risk_scores
        missed_transactions = missed_transactions.sort_values('risk_score', ascending=False)
        
        high_risk_transactions = missed_transactions.head(top_n)
        
        self.logger.info(f"Identified {len(high_risk_transactions)} high-risk missed transactions")
        self.logger.info(f"Average risk score: {high_risk_transactions['risk_score'].mean():.3f}")
        
        return high_risk_transactions
    
    def generate_alert_improvement_recommendations(self):
        """
        Generate actionable recommendations for improving alert coverage
        
        Returns:
            dict: Recommendations organized by category
        """
        self.logger.info("Generating alert improvement recommendations")
        
        recommendations = {
            'cluster_based': [],
            'density_based': [],
            'feature_based': [],
            'overall': []
        }
        
        # Cluster-based recommendations
        if self.cluster_results is not None and len(self.cluster_results) > 0:
            worst_clusters = self.cluster_results.head(5)
            
            for _, cluster in worst_clusters.iterrows():
                if cluster['cluster_id'] != -1 and cluster['alert_rate'] < 0.5:
                    rec = {
                        'type': 'cluster_coverage',
                        'cluster_id': cluster['cluster_id'],
                        'current_alert_rate': cluster['alert_rate'],
                        'missed_transactions': cluster['missed_count'],
                        'recommendation': f"Improve coverage for Cluster {cluster['cluster_id']}: "
                                        f"Only {cluster['alert_rate']:.1%} alert rate, "
                                        f"missing {cluster['missed_count']} transactions",
                        'priority': 'high' if cluster['alert_rate'] < 0.3 else 'medium'
                    }
                    
                    if 'primary_fraud_type' in cluster and pd.notna(cluster['primary_fraud_type']):
                        rec['fraud_type'] = cluster['primary_fraud_type']
                        rec['recommendation'] += f" (mainly {cluster['primary_fraud_type']})"
                    
                    recommendations['cluster_based'].append(rec)
        
        # Feature-based recommendations
        if 'amt' in self.fraud_df.columns:
            # Analyze alert rates by amount ranges
            amount_ranges = pd.cut(self.fraud_df['amt'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            amount_alert_rates = self.fraud_df.groupby(amount_ranges)['alerted'].agg(['mean', 'count']).reset_index()
            
            for _, range_stats in amount_alert_rates.iterrows():
                if range_stats['mean'] < 0.6 and range_stats['count'] > 10:
                    recommendations['feature_based'].append({
                        'type': 'amount_range',
                        'range': range_stats['amt'],
                        'alert_rate': range_stats['mean'],
                        'count': range_stats['count'],
                        'recommendation': f"Improve alerting for {range_stats['amt']} amount range: "
                                        f"{range_stats['mean']:.1%} alert rate across {range_stats['count']} transactions",
                        'priority': 'medium'
                    })
        
        # Overall recommendations
        overall_alert_rate = self.fraud_df['alerted'].mean()
        total_missed = (self.fraud_df['alerted'] == 0).sum()
        
        recommendations['overall'].append({
            'type': 'coverage_summary',
            'current_alert_rate': overall_alert_rate,
            'total_missed': total_missed,
            'recommendation': f"Overall alert coverage: {overall_alert_rate:.1%}. "
                            f"Focus on the {total_missed} missed fraud transactions to improve coverage.",
            'priority': 'high' if overall_alert_rate < 0.7 else 'medium'
        })
        
        # Noise points recommendation
        noise_data = self.fraud_df[self.fraud_df['cluster_label'] == -1]
        if len(noise_data) > 0:
            noise_alert_rate = noise_data['alerted'].mean()
            recommendations['overall'].append({
                'type': 'noise_analysis',
                'noise_count': len(noise_data),
                'noise_alert_rate': noise_alert_rate,
                'recommendation': f"Analyze {len(noise_data)} unclustered fraud transactions "
                                f"with {noise_alert_rate:.1%} alert rate. These may represent new fraud patterns.",
                'priority': 'medium'
            })
        
        return recommendations
    
    def run_comprehensive_alert_analysis(self):
        """
        Run complete alert coverage analysis
        
        Returns:
            dict: Comprehensive analysis results
        """
        self.logger.info("Running comprehensive alert coverage analysis")
        
        results = {}
        
        # 1. Cluster-based analysis
        results['cluster_analysis'] = self.analyze_cluster_alert_performance()
        
        # 2. Density-based analysis
        results['density_analysis'] = self.find_density_based_alert_gaps()
        self.density_results = results['density_analysis']  # Store for risk scoring
        
        # 3. UMAP grid analysis (if UMAP coordinates available)
        results['umap_analysis'] = self.find_umap_alert_gaps()
        
        # 4. High-risk missed transactions
        results['high_risk_missed'] = self.identify_high_risk_missed_transactions()
        
        # 5. Recommendations
        results['recommendations'] = self.generate_alert_improvement_recommendations()
        
        # 6. Summary statistics
        results['summary'] = {
            'total_fraud_transactions': len(self.fraud_df),
            'total_alerted': self.fraud_df['alerted'].sum(),
            'overall_alert_rate': self.fraud_df['alerted'].mean(),
            'total_missed': (self.fraud_df['alerted'] == 0).sum(),
            'clusters_analyzed': len(results['cluster_analysis']) if len(results['cluster_analysis']) > 0 else 0,
            'worst_cluster_alert_rate': results['cluster_analysis']['alert_rate'].min() if len(results['cluster_analysis']) > 0 else None
        }
        
        self.logger.info("Comprehensive alert analysis completed")
        self._print_analysis_summary(results)
        
        return results
    
    def _print_analysis_summary(self, results):
        """
        Print analysis summary
        
        Args:
            results (dict): Analysis results
        """
        print("\n" + "=" * 60)
        print("ALERT COVERAGE ANALYSIS SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        print(f"Total Fraud Transactions: {summary['total_fraud_transactions']:,}")
        print(f"Total Alerted: {summary['total_alerted']:,}")
        print(f"Overall Alert Rate: {summary['overall_alert_rate']:.1%}")
        print(f"Total Missed: {summary['total_missed']:,}")
        
        if summary['clusters_analyzed'] > 0:
            print(f"\nCluster Analysis:")
            print(f"  Clusters Analyzed: {summary['clusters_analyzed']}")
            print(f"  Worst Cluster Alert Rate: {summary['worst_cluster_alert_rate']:.1%}")
        
        # Top recommendations
        recommendations = results['recommendations']
        high_priority_recs = []
        
        for category in recommendations:
            for rec in recommendations[category]:
                if rec.get('priority') == 'high':
                    high_priority_recs.append(rec['recommendation'])
        
        if high_priority_recs:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(high_priority_recs[:3], 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)


def analyze_alert_coverage(clustering_results, umap_embedding=None, output_prefix="alert_analysis"):
    """
    Convenience function to run complete alert coverage analysis
    
    Args:
        clustering_results (pandas.DataFrame): Clustering results with 'alerted' column
        umap_embedding (numpy.array): UMAP coordinates (optional)
        output_prefix (str): Prefix for output files
        
    Returns:
        dict: Analysis results
    """
    analyzer = AlertCoverageAnalyzer(verbose=True)
    analyzer.load_clustering_results(clustering_results, umap_embedding)
    
    results = analyzer.run_comprehensive_alert_analysis()
    
    # Save results
    if len(results['cluster_analysis']) > 0:
        results['cluster_analysis'].to_csv(f"{output_prefix}_cluster_analysis.csv", index=False)
    
    if len(results['density_analysis']) > 0:
        results['density_analysis'].to_csv(f"{output_prefix}_density_analysis.csv", index=False)
    
    if len(results['high_risk_missed']) > 0:
        results['high_risk_missed'].to_csv(f"{output_prefix}_high_risk_missed.csv", index=False)
    
    print(f"\nAlert analysis results saved:")
    print(f"- {output_prefix}_cluster_analysis.csv")
    print(f"- {output_prefix}_density_analysis.csv")
    print(f"- {output_prefix}_high_risk_missed.csv")
    
    return results


if __name__ == "__main__":
    print("Alert Coverage Analysis Module")
    print("=" * 50)
    print("This module helps identify weak 'pockets' in fraud alerting.")
    print("\nUsage:")
    print("1. Run clustering pipeline with 'alerted' column in your data")
    print("2. Use analyze_alert_coverage(clustering_results, umap_embedding)")
    print("3. Review the generated analysis files and recommendations")
    print("\nThe analysis will identify:")
    print("- Clusters with poor alert coverage")
    print("- Dense areas with missed fraud")
    print("- High-risk missed transactions")
    print("- Actionable improvement recommendations")