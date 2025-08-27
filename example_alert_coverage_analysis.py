"""
Example: Alert Coverage Analysis for Work Dataset
This script demonstrates how to find weak "pockets" in fraud alerting using your work dataset.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
from enhanced_pipeline_with_alerts import EnhancedFraudPipelineWithAlerts, analyze_fraud_with_alert_coverage
from alert_coverage_analysis import analyze_alert_coverage
from alert_coverage_visualization import visualize_alert_coverage

def load_work_dataset():
    """
    Load your work fraud dataset
    
    Returns:
        pandas.DataFrame: Your fraud dataset
    """
    # Try to load your actual work dataset
    possible_files = ['fraudTrain.csv', 'fraud_sample.csv', 'fraud_data.csv']
    
    for filename in possible_files:
        try:
            df = pd.read_csv(filename)
            print(f"Loaded work dataset: {filename}")
            print(f"Shape: {df.shape}")
            if 'is_fraud' in df.columns:
                print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.1%})")
            return df
        except FileNotFoundError:
            continue
    
    # If no real data, create realistic sample
    print("No work dataset found, creating realistic sample...")
    return create_realistic_work_sample()

def create_realistic_work_sample():
    """
    Create a realistic work fraud sample with alert patterns
    
    Returns:
        pandas.DataFrame: Sample dataset mimicking work data
    """
    np.random.seed(42)
    
    n_transactions = 5000
    fraud_rate = 0.06  # 6% fraud rate (realistic)
    
    # Generate base transaction data
    data = {
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_transactions)],
        'amt': np.random.lognormal(mean=3.2, sigma=1.8, size=n_transactions),
        'lat': np.random.uniform(25, 49, n_transactions),  
        'long': np.random.uniform(-125, -66, n_transactions),
        'city_pop': np.random.randint(1000, 2000000, n_transactions),
        'unix_time': np.random.randint(1640995200, 1672531200, n_transactions),  # 2022 timestamps
        'merch_lat': np.random.uniform(25, 49, n_transactions),
        'merch_long': np.random.uniform(-125, -66, n_transactions),
        'zip': np.random.randint(10000, 99999, n_transactions),
        'category': np.random.choice(['grocery', 'gas', 'retail', 'restaurant', 'online', 'misc'], n_transactions)
    }
    
    # Create fraud labels with patterns
    n_fraud = int(n_transactions * fraud_rate)
    fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
    
    data['is_fraud'] = 0
    data['is_fraud'] = [1 if i in fraud_indices else 0 for i in range(n_transactions)]
    
    df = pd.DataFrame(data)
    
    # Create distinct fraud patterns for different clusters
    fraud_df = df[df['is_fraud'] == 1].copy()
    
    # Pattern 1: High-value card theft (30% of fraud)
    pattern1_size = int(len(fraud_df) * 0.3)
    pattern1_indices = fraud_df.index[:pattern1_size]
    df.loc[pattern1_indices, 'amt'] *= np.random.uniform(3.0, 8.0, pattern1_size)
    df.loc[pattern1_indices, 'return_reason'] = 'card_theft'
    
    # Pattern 2: Identity theft with geographic clustering (25% of fraud)
    pattern2_size = int(len(fraud_df) * 0.25)
    pattern2_indices = fraud_df.index[pattern1_size:pattern1_size + pattern2_size]
    df.loc[pattern2_indices, 'lat'] = np.random.normal(40.7, 0.8, pattern2_size)  # NYC area
    df.loc[pattern2_indices, 'long'] = np.random.normal(-74.0, 0.8, pattern2_size)
    df.loc[pattern2_indices, 'return_reason'] = 'identity_theft'
    
    # Pattern 3: Account takeover with time patterns (20% of fraud)
    pattern3_size = int(len(fraud_df) * 0.2)
    pattern3_indices = fraud_df.index[pattern1_size + pattern2_size:pattern1_size + pattern2_size + pattern3_size]
    # Late night transactions
    late_times = np.random.randint(1640995200, 1640995200 + 86400*30, pattern3_size)  # January 2022
    df.loc[pattern3_indices, 'unix_time'] = late_times
    df.loc[pattern3_indices, 'return_reason'] = 'account_takeover'
    
    # Pattern 4: Synthetic identity (15% of fraud)
    pattern4_size = int(len(fraud_df) * 0.15)
    pattern4_indices = fraud_df.index[pattern1_size + pattern2_size + pattern3_size:pattern1_size + pattern2_size + pattern3_size + pattern4_size]
    df.loc[pattern4_indices, 'city_pop'] = np.random.randint(50000, 500000, pattern4_size)  # Mid-size cities
    df.loc[pattern4_indices, 'return_reason'] = 'synthetic_identity'
    
    # Remaining fraud gets first_party_fraud
    remaining_indices = fraud_df.index[pattern1_size + pattern2_size + pattern3_size + pattern4_size:]
    df.loc[remaining_indices, 'return_reason'] = 'first_party_fraud'
    
    # Add realistic alert patterns (simulate your work system's alerting)
    fraud_mask = df['is_fraud'] == 1
    
    # Base alert probability
    alert_prob = np.full(fraud_mask.sum(), 0.75)  # 75% base alert rate
    
    # Alerting biases (simulate your system's strengths/weaknesses)
    fraud_data = df[fraud_mask].copy()
    
    # Higher amounts more likely to be alerted (your system catches these well)
    amount_percentile = fraud_data['amt'].rank(pct=True)
    alert_prob += (amount_percentile - 0.5) * 0.3
    
    # Some fraud types harder to catch (simulate system blind spots)
    type_adjustments = {
        'card_theft': 0.1,      # Easy to catch
        'identity_theft': -0.15, # Harder to catch (weak pocket!)
        'account_takeover': 0.05,
        'synthetic_identity': -0.2,  # Very hard to catch (major weak pocket!)
        'first_party_fraud': -0.1
    }
    
    for fraud_type, adjustment in type_adjustments.items():
        type_mask = fraud_data['return_reason'] == fraud_type
        alert_prob[type_mask] += adjustment
    
    # Geographic bias (some regions have worse coverage)
    # West coast has lower alert rates (simulate regional system differences)
    west_coast_mask = (fraud_data['long'] < -115)
    alert_prob[west_coast_mask] -= 0.1
    
    # Time-based bias (late night transactions often missed)
    # Convert unix time to hour
    hours = pd.to_datetime(fraud_data['unix_time'], unit='s').dt.hour
    late_night_mask = (hours >= 22) | (hours <= 5)
    alert_prob[late_night_mask] -= 0.15
    
    # Ensure probabilities are valid
    alert_prob = np.clip(alert_prob, 0.1, 0.95)
    
    # Generate actual alerts
    alerts = np.random.binomial(1, alert_prob, len(alert_prob))
    
    # Add to dataframe
    df['alerted'] = 0
    df.loc[fraud_mask, 'alerted'] = alerts
    
    # Set return_reason to empty for non-fraud
    df.loc[~fraud_mask, 'return_reason'] = ''
    
    final_alert_rate = df[df['is_fraud'] == 1]['alerted'].mean()
    print(f"Simulated work dataset created:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Fraud transactions: {fraud_mask.sum():,} ({fraud_mask.mean():.1%})")
    print(f"  Overall alert rate: {final_alert_rate:.1%}")
    print(f"  Weak pockets included: synthetic_identity, identity_theft, west coast, late night")
    
    return df

def example_1_identify_weak_pockets():
    """
    Example 1: Complete analysis to identify weak pockets in alerting
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: IDENTIFY WEAK POCKETS IN FRAUD ALERTING")
    print("=" * 70)
    
    # Load your work dataset
    df = load_work_dataset()
    
    # Run complete enhanced analysis
    print("\nRunning enhanced fraud analysis with alert coverage...")
    results, pipeline = analyze_fraud_with_alert_coverage(
        df,
        output_html="work_fraud_analysis.html",
        quick_search=True,  # Use False for more thorough analysis
        verbose=True
    )
    
    # Get specific weak pocket analysis
    print("\n" + "=" * 50)
    print("WEAK POCKET ANALYSIS")
    print("=" * 50)
    
    worst_pockets = pipeline.find_worst_alert_pockets(top_n=5)
    
    if 'clusters' in worst_pockets and worst_pockets['clusters']:
        print("\nWorst Alert Coverage Clusters:")
        for i, pocket in enumerate(worst_pockets['clusters'], 1):
            print(f"\n{i}. Cluster {pocket['id']}:")
            print(f"   Alert Rate: {pocket['alert_rate']:.1%}")
            print(f"   Missed Transactions: {pocket['missed_transactions']}")
            print(f"   Cluster Size: {pocket['cluster_size']}")
            if 'primary_fraud_type' in pocket:
                print(f"   Primary Fraud Type: {pocket['primary_fraud_type']}")
            if 'avg_amount' in pocket:
                print(f"   Average Amount: ${pocket['avg_amount']:.2f}")
    
    # Get improvement recommendations
    print("\n" + "=" * 50)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = pipeline.get_alert_improvement_recommendations()
    
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. [{rec.get('priority', 'MEDIUM')}] {rec['recommendation']}")
        if 'expected_improvement' in rec:
            print(f"   Potential Impact: {rec['expected_improvement']} additional alerts")
    
    print(f"\nAnalysis complete! Check these files:")
    print(f"- work_fraud_analysis.html (main visualization)")
    print(f"- alert_analysis_*.csv (detailed data)")
    print(f"- alert_coverage_*.png/html (specialized visualizations)")
    
    return results, pipeline

def example_2_deep_dive_specific_cluster():
    """
    Example 2: Deep dive into a specific problematic cluster
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DEEP DIVE INTO SPECIFIC WEAK CLUSTER")
    print("=" * 70)
    
    # First run basic analysis to identify clusters
    df = load_work_dataset()
    results, pipeline = analyze_fraud_with_alert_coverage(df, quick_search=True, verbose=False)
    
    # Get worst cluster
    worst_pockets = pipeline.find_worst_alert_pockets(top_n=1)
    
    if not worst_pockets.get('clusters'):
        print("No problematic clusters found.")
        return
    
    worst_cluster = worst_pockets['clusters'][0]
    cluster_id = worst_cluster['id']
    
    print(f"Deep diving into Cluster {cluster_id}:")
    print(f"  Alert Rate: {worst_cluster['alert_rate']:.1%}")
    print(f"  Missed Transactions: {worst_cluster['missed_transactions']}")
    
    # Get detailed cluster data
    clustering_df = results['clustering_results']['analysis_results']['clustering_results']
    cluster_data = clustering_df[clustering_df['cluster_label'] == cluster_id].copy()
    
    print(f"\nCluster {cluster_id} detailed analysis:")
    print(f"  Total transactions: {len(cluster_data)}")
    print(f"  Alerted: {cluster_data['alerted'].sum()}")
    print(f"  Missed: {(cluster_data['alerted'] == 0).sum()}")
    
    # Analyze characteristics of missed vs alerted in this cluster
    missed = cluster_data[cluster_data['alerted'] == 0]
    alerted = cluster_data[cluster_data['alerted'] == 1]
    
    if len(missed) > 0 and len(alerted) > 0:
        print(f"\nCharacteristic differences in Cluster {cluster_id}:")
        
        if 'amt' in cluster_data.columns:
            print(f"  Average amount - Missed: ${missed['amt'].mean():.2f}, Alerted: ${alerted['amt'].mean():.2f}")
        
        if 'return_reason' in cluster_data.columns:
            print(f"  Fraud types in missed transactions:")
            missed_types = missed['return_reason'].value_counts()
            for fraud_type, count in missed_types.items():
                print(f"    {fraud_type}: {count} transactions")
        
        # Geographic analysis if coordinates available
        if 'lat' in cluster_data.columns and 'long' in cluster_data.columns:
            lat_diff = abs(missed['lat'].mean() - alerted['lat'].mean())
            long_diff = abs(missed['long'].mean() - alerted['long'].mean())
            print(f"  Geographic difference - Lat: {lat_diff:.3f}, Long: {long_diff:.3f}")
    
    # Save detailed cluster analysis
    cluster_filename = f"cluster_{cluster_id}_detailed_analysis.csv"
    cluster_data.to_csv(cluster_filename, index=False)
    print(f"\nDetailed cluster data saved to: {cluster_filename}")
    
    return cluster_data

def example_3_monitor_alert_coverage():
    """
    Example 3: Create monitoring dashboard for alert coverage
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: ALERT COVERAGE MONITORING")
    print("=" * 70)
    
    df = load_work_dataset()
    
    # Run analysis
    results, pipeline = analyze_fraud_with_alert_coverage(df, verbose=False)
    
    # Create monitoring metrics
    alert_analysis = results['alert_analysis']
    
    if 'cluster_analysis' in alert_analysis:
        cluster_df = alert_analysis['cluster_analysis']
        
        # Key metrics for monitoring
        monitoring_metrics = {
            'overall_alert_rate': alert_analysis['summary']['overall_alert_rate'],
            'total_missed': alert_analysis['summary']['total_missed'],
            'worst_cluster_alert_rate': cluster_df['alert_rate'].min(),
            'clusters_under_70_percent': (cluster_df['alert_rate'] < 0.7).sum(),
            'high_risk_missed_count': len(alert_analysis.get('high_risk_missed', [])),
        }
        
        print("ALERT COVERAGE MONITORING DASHBOARD")
        print("-" * 40)
        print(f"Overall Alert Rate: {monitoring_metrics['overall_alert_rate']:.1%}")
        print(f"Total Missed: {monitoring_metrics['total_missed']:,}")
        print(f"Worst Cluster Alert Rate: {monitoring_metrics['worst_cluster_alert_rate']:.1%}")
        print(f"Clusters Under 70%: {monitoring_metrics['clusters_under_70_percent']}")
        print(f"High-Risk Missed: {monitoring_metrics['high_risk_missed_count']}")
        
        # Alert thresholds
        alerts = []
        if monitoring_metrics['overall_alert_rate'] < 0.75:
            alerts.append("🚨 Overall alert rate below 75%")
        
        if monitoring_metrics['worst_cluster_alert_rate'] < 0.5:
            alerts.append("🚨 Cluster with very low alert rate detected")
        
        if monitoring_metrics['clusters_under_70_percent'] > 3:
            alerts.append("🚨 Multiple clusters with poor coverage")
        
        if monitoring_metrics['high_risk_missed_count'] > 20:
            alerts.append("🚨 High number of high-risk missed transactions")
        
        if alerts:
            print("\nALERTS:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("\n✅ All alert coverage metrics within acceptable range")
        
        # Save monitoring report
        monitoring_df = pd.DataFrame([monitoring_metrics])
        monitoring_df['timestamp'] = pd.Timestamp.now()
        monitoring_df.to_csv("alert_coverage_monitoring.csv", index=False)
        
        print(f"\nMonitoring data saved to: alert_coverage_monitoring.csv")
    
    return monitoring_metrics

def main():
    """
    Run all alert coverage analysis examples
    """
    print("ALERT COVERAGE ANALYSIS FOR WORK DATASET")
    print("=" * 70)
    print("This script helps you find weak 'pockets' in your fraud alerting system")
    print("by analyzing clustering patterns and alert coverage.")
    
    try:
        # Example 1: Complete weak pocket identification
        example_1_identify_weak_pockets()
        
        # Example 2: Deep dive into worst cluster
        example_2_deep_dive_specific_cluster()
        
        # Example 3: Monitoring dashboard
        example_3_monitor_alert_coverage()
        
        print("\n" + "=" * 70)
        print("ALERT COVERAGE ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nKey outputs generated:")
        print("- work_fraud_analysis.html (interactive clustering visualization)")
        print("- alert_coverage_*.png/html (specialized alert gap visualizations)")
        print("- alert_analysis_*.csv (detailed data for further analysis)")
        print("- cluster_*_detailed_analysis.csv (specific cluster deep dive)")
        print("- alert_coverage_monitoring.csv (monitoring metrics)")
        
        print("\nNext steps:")
        print("1. Open work_fraud_analysis.html and color by 'alerted' to see gaps")
        print("2. Review alert_coverage_*.html for interactive gap analysis")
        print("3. Focus improvement efforts on the identified weak pockets")
        print("4. Use monitoring metrics to track improvement over time")
        
    except Exception as e:
        print(f"\nError running alert coverage analysis: {e}")
        print("Ensure you have the required dependencies and data files.")

if __name__ == "__main__":
    main()