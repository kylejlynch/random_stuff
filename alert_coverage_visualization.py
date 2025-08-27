"""
Alert Coverage Visualization Module
Creates specialized visualizations for identifying and analyzing alert coverage gaps.
"""

# Import compatibility module first
from compatibility import ensure_sklearn_compatibility
ensure_sklearn_compatibility()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class AlertCoverageVisualizer:
    """
    Visualizer for alert coverage analysis results
    """
    
    def __init__(self, analysis_results):
        """
        Initialize visualizer with analysis results
        
        Args:
            analysis_results (dict): Results from AlertCoverageAnalyzer
        """
        self.results = analysis_results
        self.fraud_df = None
        
        # Extract fraud dataframe from density analysis if available
        if 'density_analysis' in analysis_results and len(analysis_results['density_analysis']) > 0:
            self.fraud_df = analysis_results['density_analysis']
    
    def create_cluster_alert_heatmap(self):
        """
        Create heatmap showing alert performance by cluster
        
        Returns:
            matplotlib.figure.Figure: Heatmap figure
        """
        if 'cluster_analysis' not in self.results or len(self.results['cluster_analysis']) == 0:
            return None
        
        cluster_df = self.results['cluster_analysis']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare data for heatmaps
        cluster_df_viz = cluster_df[cluster_df['cluster_id'] != -1].copy()  # Exclude noise for main heatmaps
        
        if len(cluster_df_viz) == 0:
            return None
        
        # Heatmap 1: Alert Rate by Cluster Size
        scatter_colors = cluster_df_viz['alert_rate']
        scatter = axes[0].scatter(cluster_df_viz['size'], cluster_df_viz['alert_rate'], 
                                 c=cluster_df_viz['alert_gap_score'], cmap='Reds', 
                                 s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        axes[0].set_xlabel('Cluster Size')
        axes[0].set_ylabel('Alert Rate')
        axes[0].set_title('Alert Rate vs Cluster Size\n(Color = Alert Gap Score)')
        axes[0].grid(True, alpha=0.3)
        
        # Add cluster ID labels
        for _, row in cluster_df_viz.iterrows():
            axes[0].annotate(f"C{int(row['cluster_id'])}", 
                           (row['size'], row['alert_rate']), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, ax=axes[0], label='Alert Gap Score')
        
        # Heatmap 2: Missed Transactions by Cluster
        bars = axes[1].bar(range(len(cluster_df_viz)), cluster_df_viz['missed_count'], 
                          color=plt.cm.Reds(cluster_df_viz['alert_rate']), 
                          edgecolor='black', linewidth=1)
        
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Missed Transactions')
        axes[1].set_title('Missed Transactions by Cluster')
        axes[1].set_xticks(range(len(cluster_df_viz)))
        axes[1].set_xticklabels([f"C{int(cid)}" for cid in cluster_df_viz['cluster_id']], rotation=45)
        
        # Add value labels on bars
        for bar, missed in zip(bars, cluster_df_viz['missed_count']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{int(missed)}', ha='center', va='bottom', fontsize=8)
        
        # Heatmap 3: Alert Performance Summary
        if 'avg_amount' in cluster_df_viz.columns:
            # Amount vs Alert Rate
            scatter2 = axes[2].scatter(cluster_df_viz['avg_amount'], cluster_df_viz['alert_rate'],
                                     c=cluster_df_viz['size'], cmap='viridis', 
                                     s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            axes[2].set_xlabel('Average Transaction Amount')
            axes[2].set_ylabel('Alert Rate')
            axes[2].set_title('Alert Rate vs Average Amount\n(Size = Cluster Size)')
            axes[2].grid(True, alpha=0.3)
            
            plt.colorbar(scatter2, ax=axes[2], label='Cluster Size')
            
            # Add cluster ID labels
            for _, row in cluster_df_viz.iterrows():
                axes[2].annotate(f"C{int(row['cluster_id'])}", 
                               (row['avg_amount'], row['alert_rate']), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
        else:
            # Simple alert rate distribution
            axes[2].hist(cluster_df_viz['alert_rate'], bins=10, color='skyblue', 
                        alpha=0.7, edgecolor='black')
            axes[2].axvline(cluster_df_viz['alert_rate'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {cluster_df_viz["alert_rate"].mean():.2f}')
            axes[2].set_xlabel('Alert Rate')
            axes[2].set_ylabel('Number of Clusters')
            axes[2].set_title('Alert Rate Distribution')
            axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def create_umap_alert_coverage_plot(self):
        """
        Create UMAP visualization showing alert coverage gaps
        
        Returns:
            plotly.graph_objects.Figure: Interactive UMAP plot
        """
        if self.fraud_df is None or 'umap_x' not in self.fraud_df.columns:
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Alert Coverage (Red = Missed)', 'Alert Gap Score', 
                          'Cluster Distribution', 'Transaction Amount'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Common hover text
        hover_text = []
        for _, row in self.fraud_df.iterrows():
            text = (f"Alerted: {'Yes' if row['alerted'] else 'No'}<br>"
                   f"Cluster: {row['cluster_label']}<br>"
                   f"Alert Gap Score: {row.get('alert_gap_score', 0):.3f}")
            if 'amt' in row:
                text += f"<br>Amount: ${row['amt']:.2f}"
            if 'return_reason' in row:
                text += f"<br>Fraud Type: {row['return_reason']}"
            hover_text.append(text)
        
        # Plot 1: Alert Coverage (Red = Missed)
        colors = ['red' if not alerted else 'blue' for alerted in self.fraud_df['alerted']]
        fig.add_trace(
            go.Scatter(
                x=self.fraud_df['umap_x'],
                y=self.fraud_df['umap_y'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=4,
                    opacity=0.7
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Alert Status',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: Alert Gap Score
        fig.add_trace(
            go.Scatter(
                x=self.fraud_df['umap_x'],
                y=self.fraud_df['umap_y'],
                mode='markers',
                marker=dict(
                    color=self.fraud_df.get('alert_gap_score', 0),
                    colorscale='Reds',
                    size=4,
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Alert Gap Score", x=0.48)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Gap Score',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Cluster Distribution
        fig.add_trace(
            go.Scatter(
                x=self.fraud_df['umap_x'],
                y=self.fraud_df['umap_y'],
                mode='markers',
                marker=dict(
                    color=self.fraud_df['cluster_label'],
                    colorscale='viridis',
                    size=4,
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Cluster ID", x=1.02)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Clusters',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Transaction Amount (if available)
        if 'amt' in self.fraud_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.fraud_df['umap_x'],
                    y=self.fraud_df['umap_y'],
                    mode='markers',
                    marker=dict(
                        color=self.fraud_df['amt'],
                        colorscale='Plasma',
                        size=4,
                        opacity=0.7,
                        showscale=False
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name='Amount',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='UMAP Alert Coverage Analysis',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="UMAP Dimension 1", row=i, col=j)
                fig.update_yaxes(title_text="UMAP Dimension 2", row=i, col=j)
        
        return fig
    
    def create_alert_gap_density_plot(self):
        """
        Create density plot showing alert gaps in feature space
        
        Returns:
            matplotlib.figure.Figure: Density plot figure
        """
        if self.fraud_df is None or 'alert_gap_score' not in self.fraud_df.columns:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Alert Gap Analysis - Feature Space', fontsize=16)
        
        # Get high alert gap transactions
        high_gap = self.fraud_df[self.fraud_df['alert_gap_score'] > 0.1]
        normal_gap = self.fraud_df[self.fraud_df['alert_gap_score'] <= 0.1]
        
        # Plot 1: Alert Gap Score Distribution
        axes[0, 0].hist(self.fraud_df['alert_gap_score'], bins=30, alpha=0.7, 
                       color='lightcoral', edgecolor='black', label='All Transactions')
        axes[0, 0].axvline(self.fraud_df['alert_gap_score'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.fraud_df["alert_gap_score"].mean():.3f}')
        axes[0, 0].set_xlabel('Alert Gap Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Alert Gap Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Alert Gap vs Density Score
        if 'density_score' in self.fraud_df.columns:
            scatter = axes[0, 1].scatter(self.fraud_df['density_score'], self.fraud_df['alert_gap_score'],
                                       c=self.fraud_df['alerted'], cmap='RdYlBu', alpha=0.6)
            axes[0, 1].set_xlabel('Density Score')
            axes[0, 1].set_ylabel('Alert Gap Score')
            axes[0, 1].set_title('Alert Gap vs Density Score')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Alerted (1=Yes)')
        
        # Plot 3: High Gap Transactions by Cluster
        if len(high_gap) > 0:
            cluster_gap_counts = high_gap['cluster_label'].value_counts().head(10)
            bars = axes[1, 0].bar(range(len(cluster_gap_counts)), cluster_gap_counts.values,
                                 color='darkred', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('High Alert Gap Transactions')
            axes[1, 0].set_title('High Alert Gap Transactions by Cluster')
            axes[1, 0].set_xticks(range(len(cluster_gap_counts)))
            axes[1, 0].set_xticklabels([f'C{cid}' for cid in cluster_gap_counts.index], rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, cluster_gap_counts.values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Alert Rate by Amount Quartiles (if amount available)
        if 'amt' in self.fraud_df.columns:
            quartiles = pd.qcut(self.fraud_df['amt'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            alert_by_quartile = self.fraud_df.groupby(quartiles)['alerted'].agg(['mean', 'count'])
            
            bars = axes[1, 1].bar(alert_by_quartile.index, alert_by_quartile['mean'],
                                 color=['lightblue', 'skyblue', 'lightcoral', 'darkred'],
                                 alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Amount Quartile')
            axes[1, 1].set_ylabel('Alert Rate')
            axes[1, 1].set_title('Alert Rate by Amount Quartile')
            axes[1, 1].set_ylim(0, 1)
            
            # Add count labels
            for bar, (_, row) in zip(bars, alert_by_quartile.iterrows()):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{row["mean"]:.2f}\n(n={row["count"]})', 
                               ha='center', va='bottom', fontsize=8)
        else:
            # Alternative: Alert rate by cluster
            if 'cluster_analysis' in self.results and len(self.results['cluster_analysis']) > 0:
                cluster_df = self.results['cluster_analysis'].head(10)
                bars = axes[1, 1].bar(range(len(cluster_df)), cluster_df['alert_rate'],
                                     color=plt.cm.RdYlBu(cluster_df['alert_rate']),
                                     alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Cluster ID')
                axes[1, 1].set_ylabel('Alert Rate')
                axes[1, 1].set_title('Alert Rate by Cluster')
                axes[1, 1].set_xticks(range(len(cluster_df)))
                axes[1, 1].set_xticklabels([f'C{int(cid)}' for cid in cluster_df['cluster_id']], rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_high_risk_transactions_summary(self):
        """
        Create summary visualization of high-risk missed transactions
        
        Returns:
            matplotlib.figure.Figure: Summary figure
        """
        if 'high_risk_missed' not in self.results or len(self.results['high_risk_missed']) == 0:
            return None
        
        high_risk_df = self.results['high_risk_missed']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('High-Risk Missed Transactions Analysis', fontsize=16)
        
        # Plot 1: Risk Score Distribution
        axes[0, 0].hist(high_risk_df['risk_score'], bins=20, alpha=0.7, 
                       color='darkred', edgecolor='black')
        axes[0, 0].axvline(high_risk_df['risk_score'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {high_risk_df["risk_score"].mean():.3f}')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Risk Score Distribution (Missed Transactions)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Risk Score vs Transaction Amount
        if 'amt' in high_risk_df.columns:
            scatter = axes[0, 1].scatter(high_risk_df['amt'], high_risk_df['risk_score'],
                                       c=high_risk_df['cluster_label'], cmap='tab10', alpha=0.7)
            axes[0, 1].set_xlabel('Transaction Amount')
            axes[0, 1].set_ylabel('Risk Score')
            axes[0, 1].set_title('Risk Score vs Amount')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Cluster ID')
        
        # Plot 3: High-Risk Transactions by Cluster
        cluster_risk_counts = high_risk_df['cluster_label'].value_counts().head(8)
        bars = axes[1, 0].bar(range(len(cluster_risk_counts)), cluster_risk_counts.values,
                             color='maroon', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('High-Risk Missed Transactions')
        axes[1, 0].set_title('High-Risk Missed Transactions by Cluster')
        axes[1, 0].set_xticks(range(len(cluster_risk_counts)))
        axes[1, 0].set_xticklabels([f'C{cid}' for cid in cluster_risk_counts.index], rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, cluster_risk_counts.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Fraud Type Distribution (if available)
        if 'return_reason' in high_risk_df.columns:
            fraud_type_counts = high_risk_df['return_reason'].value_counts().head(6)
            colors = plt.cm.Set3(np.linspace(0, 1, len(fraud_type_counts)))
            
            wedges, texts, autotexts = axes[1, 1].pie(fraud_type_counts.values, 
                                                     labels=fraud_type_counts.index,
                                                     autopct='%1.1f%%', colors=colors)
            axes[1, 1].set_title('High-Risk Missed Transactions\nby Fraud Type')
        else:
            # Alternative: Risk score vs cluster size
            if 'cluster_analysis' in self.results:
                cluster_df = self.results['cluster_analysis']
                # Merge with high risk data
                high_risk_by_cluster = high_risk_df['cluster_label'].value_counts()
                merged_data = cluster_df.merge(
                    high_risk_by_cluster.to_frame('high_risk_count').reset_index().rename(columns={'cluster_label': 'cluster_id'}),
                    on='cluster_id', how='left'
                ).fillna(0)
                
                scatter = axes[1, 1].scatter(merged_data['size'], merged_data['high_risk_count'],
                                           c=merged_data['alert_rate'], cmap='RdYlBu', 
                                           s=80, alpha=0.7, edgecolors='black')
                axes[1, 1].set_xlabel('Cluster Size')
                axes[1, 1].set_ylabel('High-Risk Missed Count')
                axes[1, 1].set_title('High-Risk Missed vs Cluster Size')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Alert Rate')
        
        plt.tight_layout()
        return fig
    
    def save_all_visualizations(self, output_prefix="alert_coverage"):
        """
        Save all alert coverage visualizations
        
        Args:
            output_prefix (str): Prefix for output files
        """
        saved_files = []
        
        # Save cluster heatmap
        cluster_fig = self.create_cluster_alert_heatmap()
        if cluster_fig:
            cluster_fig.savefig(f"{output_prefix}_cluster_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(cluster_fig)
            saved_files.append(f"{output_prefix}_cluster_heatmap.png")
        
        # Save UMAP interactive plot
        umap_fig = self.create_umap_alert_coverage_plot()
        if umap_fig:
            pyo.plot(umap_fig, filename=f"{output_prefix}_umap_interactive.html", auto_open=False)
            saved_files.append(f"{output_prefix}_umap_interactive.html")
        
        # Save density plot
        density_fig = self.create_alert_gap_density_plot()
        if density_fig:
            density_fig.savefig(f"{output_prefix}_density_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(density_fig)
            saved_files.append(f"{output_prefix}_density_analysis.png")
        
        # Save high-risk summary
        risk_fig = self.create_high_risk_transactions_summary()
        if risk_fig:
            risk_fig.savefig(f"{output_prefix}_high_risk_summary.png", dpi=300, bbox_inches='tight')
            plt.close(risk_fig)
            saved_files.append(f"{output_prefix}_high_risk_summary.png")
        
        if saved_files:
            print(f"\nAlert coverage visualizations saved:")
            for file in saved_files:
                print(f"- {file}")
        
        return saved_files


def visualize_alert_coverage(analysis_results, output_prefix="alert_coverage_analysis"):
    """
    Convenience function to create all alert coverage visualizations
    
    Args:
        analysis_results (dict): Results from alert coverage analysis
        output_prefix (str): Prefix for output files
        
    Returns:
        AlertCoverageVisualizer: The visualizer instance
    """
    visualizer = AlertCoverageVisualizer(analysis_results)
    visualizer.save_all_visualizations(output_prefix)
    
    return visualizer


if __name__ == "__main__":
    print("Alert Coverage Visualization Module")
    print("=" * 50)
    print("Creates specialized visualizations for alert coverage analysis.")
    print("\nGenerated visualizations:")
    print("1. Cluster Alert Heatmap - Shows alert performance by cluster")
    print("2. UMAP Interactive Plot - Shows alert gaps in embedding space")
    print("3. Density Analysis Plot - Shows alert gaps in feature space")
    print("4. High-Risk Summary - Analyzes highest risk missed transactions")
    print("\nUsage:")
    print("results = analyze_alert_coverage(clustering_df, umap_embedding)")
    print("visualizer = visualize_alert_coverage(results)")