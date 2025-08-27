"""
Enhanced Fraud Clustering Pipeline with Alert Coverage Analysis
This pipeline combines clustering analysis with alert coverage analysis to identify weak pockets in fraud detection.
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
from auto_tuned_pipeline import AutoTunedFraudClusteringPipeline
from alert_coverage_analysis import AlertCoverageAnalyzer, analyze_alert_coverage
from alert_coverage_visualization import visualize_alert_coverage
from logging_config import setup_pipeline_logging, get_logger

class EnhancedFraudPipelineWithAlerts:
    """
    Enhanced fraud clustering pipeline with integrated alert coverage analysis
    """
    
    def __init__(self, verbose=False, enable_tuning=True):
        """
        Initialize enhanced pipeline
        
        Args:
            verbose (bool): Enable verbose logging
            enable_tuning (bool): Enable parameter tuning
        """
        setup_pipeline_logging(verbose=verbose)
        self.logger = get_logger(__name__)
        self.enable_tuning = enable_tuning
        
        # Initialize sub-components
        self.clustering_pipeline = AutoTunedFraudClusteringPipeline(verbose=verbose, enable_tuning=enable_tuning)
        self.alert_analyzer = AlertCoverageAnalyzer(verbose=verbose)
        
        # Results storage
        self.clustering_results = None
        self.alert_analysis_results = None
        self.umap_embedding = None
        
    def validate_input_data(self, df):
        """
        Validate that input data has required columns for alert analysis
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            tuple: (is_valid, missing_columns, suggestions)
        """
        required_cols = ['is_fraud']
        recommended_cols = ['alerted', 'return_reason', 'amt']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        missing_recommended = [col for col in recommended_cols if col not in df.columns]
        
        suggestions = []
        
        if missing_required:
            return False, missing_required, ["Add 'is_fraud' column to identify fraudulent transactions"]
        
        if 'alerted' not in df.columns:
            suggestions.append("Add 'alerted' column (0/1) to analyze alert coverage")
            
        if 'return_reason' not in df.columns:
            suggestions.append("Add 'return_reason' column to analyze fraud types")
            
        if 'amt' not in df.columns:
            suggestions.append("Add 'amt' column for transaction amount analysis")
        
        return True, missing_recommended, suggestions
    
    def prepare_data_for_alert_analysis(self, df):
        """
        Prepare data for alert analysis by adding missing columns with simulated data
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            pandas.DataFrame: Prepared DataFrame
        """
        df_prepared = df.copy()
        
        # Add alerted column if missing (simulate realistic alerting)
        if 'alerted' not in df_prepared.columns:
            self.logger.info("Adding simulated 'alerted' column")
            fraud_df = df_prepared[df_prepared['is_fraud'] == 1]
            
            # Simulate alert patterns with some bias
            np.random.seed(42)
            alert_probabilities = np.random.beta(2, 1, len(fraud_df))  # Skewed toward alerting
            
            # Add some bias based on amount if available
            if 'amt' in fraud_df.columns:
                # Higher amounts more likely to be alerted
                amount_percentiles = fraud_df['amt'].rank(pct=True)
                alert_probabilities = 0.7 * alert_probabilities + 0.3 * amount_percentiles
            
            # Generate alerts
            alerted = np.random.binomial(1, np.clip(alert_probabilities, 0, 1), len(fraud_df))
            
            # Add to full dataframe
            df_prepared['alerted'] = 0
            df_prepared.loc[df_prepared['is_fraud'] == 1, 'alerted'] = alerted
            
            alert_rate = df_prepared[df_prepared['is_fraud'] == 1]['alerted'].mean()
            self.logger.info(f"Simulated alert rate: {alert_rate:.1%}")
        
        # Add return_reason if missing (simulate fraud types)
        if 'return_reason' not in df_prepared.columns:
            self.logger.info("Adding simulated 'return_reason' column")
            fraud_types = ['card_theft', 'identity_theft', 'account_takeover', 'synthetic_identity', 'first_party_fraud']
            
            fraud_mask = df_prepared['is_fraud'] == 1
            fraud_reasons = np.random.choice(fraud_types, size=fraud_mask.sum())
            
            df_prepared['return_reason'] = ''
            df_prepared.loc[fraud_mask, 'return_reason'] = fraud_reasons
        
        return df_prepared
    
    def run_comprehensive_analysis(self, df, feature_columns=None, 
                                 output_html_path="enhanced_clusters.html",
                                 quick_search=False, save_all_results=True):
        """
        Run comprehensive fraud analysis with alert coverage
        
        Args:
            df (pandas.DataFrame): Input DataFrame with fraud data
            feature_columns (list): Features for clustering
            output_html_path (str): Path for main visualization
            quick_search (bool): Use quick parameter search
            save_all_results (bool): Save all intermediate results
            
        Returns:
            dict: Complete analysis results
        """
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info("Starting Enhanced Fraud Pipeline with Alert Coverage Analysis")
        self.logger.info(f"Dataset shape: {df.shape}")
        
        # Step 1: Validate and prepare data
        is_valid, missing_cols, suggestions = self.validate_input_data(df)
        
        if not is_valid:
            raise ValueError(f"Invalid input data. Missing required columns: {missing_cols}")
        
        if suggestions:
            self.logger.info("Data preparation suggestions:")
            for suggestion in suggestions:
                self.logger.info(f"  - {suggestion}")
        
        df_prepared = self.prepare_data_for_alert_analysis(df)
        
        # Step 2: Run clustering analysis
        self.logger.info("Running clustering analysis...")
        self.clustering_results = self.clustering_pipeline.run_optimized_analysis(
            df_prepared,
            feature_columns=feature_columns,
            output_html_path=output_html_path,
            quick_search=quick_search,
            save_tuning_results=save_all_results
        )
        
        # Step 3: Extract UMAP embedding from clustering pipeline
        if (self.clustering_results and 
            'analysis_results' in self.clustering_results and 
            self.clustering_results['analysis_results'] and
            'visualization' in self.clustering_results['analysis_results']):
            
            # Try to extract UMAP coordinates from the visualization pipeline
            try:
                # Get clustering dataframe
                clustering_df = self.clustering_results['analysis_results']['clustering_results']
                
                # Run UMAP on the same data to get coordinates
                from umap_visualization import UMAPVisualizer
                pca_columns = [col for col in clustering_df.columns if col.startswith('PC')]
                
                if len(pca_columns) > 0:
                    X_pca = clustering_df[pca_columns].values
                    umap_viz = UMAPVisualizer(n_components=3, random_state=42)
                    self.umap_embedding = umap_viz.fit_umap(X_pca)
                    self.logger.info("Generated UMAP embedding for alert analysis")
                
            except Exception as e:
                self.logger.warning(f"Could not generate UMAP embedding: {e}")
                self.umap_embedding = None
        
        # Step 4: Run alert coverage analysis
        self.logger.info("Running alert coverage analysis...")
        clustering_df = self.clustering_results['analysis_results']['clustering_results']
        
        self.alert_analysis_results = analyze_alert_coverage(
            clustering_df, 
            self.umap_embedding,
            output_prefix=f"alert_analysis_{timestamp}"
        )
        
        # Step 5: Create alert coverage visualizations
        self.logger.info("Creating alert coverage visualizations...")
        visualize_alert_coverage(
            self.alert_analysis_results,
            output_prefix=f"alert_coverage_{timestamp}"
        )
        
        # Step 6: Generate comprehensive report
        total_time = time.time() - start_time
        
        comprehensive_results = {
            'clustering_results': self.clustering_results,
            'alert_analysis': self.alert_analysis_results,
            'umap_embedding': self.umap_embedding,
            'total_runtime': total_time,
            'timestamp': timestamp,
            'input_data_shape': df.shape,
            'fraud_transactions': len(df_prepared[df_prepared['is_fraud'] == 1])
        }
        
        # Print comprehensive summary
        self._print_comprehensive_summary(comprehensive_results)
        
        # Save summary report if requested
        if save_all_results:
            self._save_comprehensive_report(comprehensive_results, timestamp)
        
        return comprehensive_results
    
    def find_worst_alert_pockets(self, top_n=10):
        """
        Identify the worst alert coverage pockets
        
        Args:
            top_n (int): Number of worst pockets to return
            
        Returns:
            dict: Analysis of worst alert coverage areas
        """
        if not self.alert_analysis_results:
            self.logger.error("No alert analysis results available. Run comprehensive analysis first.")
            return {}
        
        worst_pockets = {}
        
        # Worst clusters
        if 'cluster_analysis' in self.alert_analysis_results and len(self.alert_analysis_results['cluster_analysis']) > 0:
            cluster_df = self.alert_analysis_results['cluster_analysis']
            worst_clusters = cluster_df.head(top_n)
            
            worst_pockets['clusters'] = []
            for _, cluster in worst_clusters.iterrows():
                if cluster['alert_rate'] < 0.8 and cluster['size'] >= 10:  # Focus on substantial, under-alerted clusters
                    pocket = {
                        'type': 'cluster',
                        'id': cluster['cluster_id'],
                        'alert_rate': cluster['alert_rate'],
                        'missed_transactions': cluster['missed_count'],
                        'cluster_size': cluster['size'],
                        'alert_gap_score': cluster['alert_gap_score'],
                        'improvement_potential': cluster['missed_count']
                    }
                    
                    if 'primary_fraud_type' in cluster:
                        pocket['primary_fraud_type'] = cluster['primary_fraud_type']
                    
                    if 'avg_amount' in cluster:
                        pocket['avg_amount'] = cluster['avg_amount']
                    
                    worst_pockets['clusters'].append(pocket)
        
        # High-risk missed transactions
        if 'high_risk_missed' in self.alert_analysis_results and len(self.alert_analysis_results['high_risk_missed']) > 0:
            high_risk_df = self.alert_analysis_results['high_risk_missed'].head(top_n)
            
            worst_pockets['high_risk_transactions'] = []
            for _, txn in high_risk_df.iterrows():
                transaction = {
                    'risk_score': txn['risk_score'],
                    'cluster_id': txn['cluster_label'],
                    'amount': txn.get('amt', 'unknown'),
                    'fraud_type': txn.get('return_reason', 'unknown')
                }
                worst_pockets['high_risk_transactions'].append(transaction)
        
        # UMAP grid analysis (if available)
        if 'umap_analysis' in self.alert_analysis_results and len(self.alert_analysis_results['umap_analysis']) > 0:
            umap_df = self.alert_analysis_results['umap_analysis']
            worst_umap_areas = umap_df.head(min(5, top_n))  # Top 5 UMAP areas
            
            worst_pockets['umap_areas'] = []
            for _, area in worst_umap_areas.iterrows():
                if area['alert_rate'] < 0.7 and area['count'] >= 5:
                    umap_pocket = {
                        'type': 'umap_grid',
                        'location': f"({area['x_center']:.2f}, {area['y_center']:.2f})",
                        'alert_rate': area['alert_rate'],
                        'transaction_count': area['count'],
                        'missed_count': area['missed_count']
                    }
                    worst_pockets['umap_areas'].append(umap_pocket)
        
        return worst_pockets
    
    def get_alert_improvement_recommendations(self):
        """
        Get specific recommendations for improving alert coverage
        
        Returns:
            list: Prioritized list of improvement recommendations
        """
        if not self.alert_analysis_results:
            return []
        
        recommendations = self.alert_analysis_results.get('recommendations', {})
        worst_pockets = self.find_worst_alert_pockets()
        
        # Combine and prioritize recommendations
        all_recommendations = []
        
        # Add cluster-specific recommendations
        for category in ['cluster_based', 'feature_based', 'overall']:
            if category in recommendations:
                all_recommendations.extend(recommendations[category])
        
        # Add pocket-specific recommendations
        if 'clusters' in worst_pockets:
            for pocket in worst_pockets['clusters'][:3]:  # Top 3 worst clusters
                rec = {
                    'type': 'cluster_improvement',
                    'priority': 'high' if pocket['alert_rate'] < 0.5 else 'medium',
                    'recommendation': f"Focus on Cluster {pocket['id']}: {pocket['missed_transactions']} missed transactions "
                                    f"with {pocket['alert_rate']:.1%} alert rate",
                    'expected_improvement': pocket['missed_transactions'],
                    'cluster_id': pocket['id']
                }
                
                if 'primary_fraud_type' in pocket:
                    rec['recommendation'] += f" (mainly {pocket['primary_fraud_type']})"
                
                all_recommendations.append(rec)
        
        # Sort by priority and potential impact
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        all_recommendations.sort(key=lambda x: (
            priority_order.get(x.get('priority', 'low'), 1),
            x.get('expected_improvement', 0)
        ), reverse=True)
        
        return all_recommendations[:10]  # Top 10 recommendations
    
    def _print_comprehensive_summary(self, results):
        """
        Print comprehensive analysis summary
        
        Args:
            results (dict): Complete analysis results
        """
        print("\n" + "=" * 80)
        print("ENHANCED FRAUD CLUSTERING WITH ALERT COVERAGE ANALYSIS")
        print("=" * 80)
        
        # Basic stats
        clustering = results['clustering_results']
        alert_analysis = results['alert_analysis']
        
        print(f"Dataset: {results['input_data_shape'][0]:,} total transactions")
        print(f"Fraud Transactions: {results['fraud_transactions']:,}")
        print(f"Analysis Runtime: {results['total_runtime']:.2f} seconds")
        
        # Clustering results
        if clustering and 'analysis_results' in clustering:
            cluster_df = clustering['analysis_results']['clustering_results']
            cluster_summary = cluster_df['cluster_label'].value_counts()
            n_clusters = len(cluster_summary[cluster_summary.index != -1])
            n_noise = cluster_summary.get(-1, 0)
            
            print(f"\nClustering Results:")
            print(f"  Clusters Found: {n_clusters}")
            print(f"  Noise Points: {n_noise:,} ({n_noise/len(cluster_df)*100:.1f}%)")
        
        # Alert coverage results
        if alert_analysis and 'summary' in alert_analysis:
            summary = alert_analysis['summary']
            print(f"\nAlert Coverage Analysis:")
            print(f"  Overall Alert Rate: {summary['overall_alert_rate']:.1%}")
            print(f"  Total Missed: {summary['total_missed']:,}")
            
            if summary['worst_cluster_alert_rate']:
                print(f"  Worst Cluster Alert Rate: {summary['worst_cluster_alert_rate']:.1%}")
        
        # Top improvement opportunities
        recommendations = self.get_alert_improvement_recommendations()
        if recommendations:
            print(f"\nTop Improvement Opportunities:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec['recommendation']}")
        
        # File outputs
        print(f"\nGenerated Files (timestamp: {results['timestamp']}):")
        print(f"  - enhanced_clusters.html (main clustering visualization)")
        print(f"  - alert_analysis_{results['timestamp']}_*.csv (alert analysis data)")
        print(f"  - alert_coverage_{results['timestamp']}_*.png/html (alert visualizations)")
        
        print("=" * 80)
    
    def _save_comprehensive_report(self, results, timestamp):
        """
        Save comprehensive analysis report
        
        Args:
            results (dict): Analysis results
            timestamp (str): Timestamp for file naming
        """
        report_path = f"comprehensive_fraud_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED FRAUD CLUSTERING WITH ALERT COVERAGE ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic information
            f.write("DATASET SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Transactions: {results['input_data_shape'][0]:,}\n")
            f.write(f"Fraud Transactions: {results['fraud_transactions']:,}\n")
            f.write(f"Analysis Runtime: {results['total_runtime']:.2f} seconds\n\n")
            
            # Alert coverage summary
            if results['alert_analysis'] and 'summary' in results['alert_analysis']:
                summary = results['alert_analysis']['summary']
                f.write("ALERT COVERAGE SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Alert Rate: {summary['overall_alert_rate']:.1%}\n")
                f.write(f"Total Missed: {summary['total_missed']:,}\n")
                f.write(f"Clusters Analyzed: {summary['clusters_analyzed']}\n")
                if summary['worst_cluster_alert_rate']:
                    f.write(f"Worst Cluster Alert Rate: {summary['worst_cluster_alert_rate']:.1%}\n")
                f.write("\n")
            
            # Improvement recommendations
            recommendations = self.get_alert_improvement_recommendations()
            if recommendations:
                f.write("IMPROVEMENT RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec['recommendation']}\n")
                f.write("\n")
            
            # Worst pockets summary
            worst_pockets = self.find_worst_alert_pockets()
            if 'clusters' in worst_pockets and worst_pockets['clusters']:
                f.write("WORST ALERT COVERAGE POCKETS\n")
                f.write("-" * 30 + "\n")
                for i, pocket in enumerate(worst_pockets['clusters'][:5], 1):
                    f.write(f"{i}. Cluster {pocket['id']}: {pocket['alert_rate']:.1%} alert rate, "
                           f"{pocket['missed_transactions']} missed transactions\n")
        
        self.logger.info(f"Comprehensive report saved to: {report_path}")


def analyze_fraud_with_alert_coverage(df, output_html="enhanced_fraud_analysis.html", 
                                    quick_search=True, verbose=True):
    """
    Convenience function for complete fraud analysis with alert coverage
    
    Args:
        df (pandas.DataFrame): DataFrame with fraud data
        output_html (str): Main visualization output path
        quick_search (bool): Use quick parameter search
        verbose (bool): Enable verbose logging
        
    Returns:
        dict: Complete analysis results
    """
    pipeline = EnhancedFraudPipelineWithAlerts(verbose=verbose, enable_tuning=True)
    
    results = pipeline.run_comprehensive_analysis(
        df,
        output_html_path=output_html,
        quick_search=quick_search,
        save_all_results=True
    )
    
    return results, pipeline


if __name__ == "__main__":
    print("Enhanced Fraud Clustering Pipeline with Alert Coverage Analysis")
    print("=" * 70)
    print("This pipeline combines advanced clustering with alert coverage analysis")
    print("to identify weak 'pockets' in fraud detection.")
    print("\nFeatures:")
    print("- Automatic parameter tuning for optimal clustering")
    print("- Alert coverage analysis by cluster and density")
    print("- UMAP-based spatial alert gap detection")
    print("- High-risk missed transaction identification")
    print("- Comprehensive improvement recommendations")
    print("- Rich visualizations and reports")
    print("\nUsage:")
    print("results, pipeline = analyze_fraud_with_alert_coverage(your_fraud_df)")
    print("recommendations = pipeline.get_alert_improvement_recommendations()")
    print("worst_pockets = pipeline.find_worst_alert_pockets()")