"""
Visualization Module for Parameter Tuning Results
Creates plots and interactive visualizations for parameter search results.
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

class ParameterSearchVisualizer:
    """
    Visualizer for parameter search results
    """
    
    def __init__(self, search_results):
        """
        Initialize visualizer with search results
        
        Args:
            search_results (dict): Results from parameter_tuning.run_full_parameter_search
        """
        self.search_results = search_results
        self.df_results = self._prepare_dataframe()
    
    def _prepare_dataframe(self):
        """
        Convert search results to DataFrame for easier plotting
        
        Returns:
            pandas.DataFrame: Results as DataFrame
        """
        if not self.search_results['all_results']:
            return pd.DataFrame()
        
        rows = []
        for result in self.search_results['all_results']:
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
        
        return pd.DataFrame(rows)
    
    def create_score_comparison_plot(self):
        """
        Create matplotlib plot comparing different scoring metrics
        
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if self.df_results.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Search Results - Score Comparisons', fontsize=16)
        
        # Top 20 results for readability
        top_results = self.df_results.head(20)
        
        # Composite Score vs PCA Components
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(top_results['pca_components'], top_results['composite_score'], 
                              c=top_results['n_clusters'], cmap='viridis', alpha=0.7, s=50)
        ax1.set_xlabel('PCA Components')
        ax1.set_ylabel('Composite Score')
        ax1.set_title('Composite Score vs PCA Components')
        plt.colorbar(scatter1, ax=ax1, label='Number of Clusters')
        
        # Composite Score vs Min Cluster Size
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(top_results['min_cluster_size'], top_results['composite_score'],
                              c=top_results['noise_ratio'], cmap='plasma', alpha=0.7, s=50)
        ax2.set_xlabel('Min Cluster Size')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Composite Score vs Min Cluster Size')
        plt.colorbar(scatter2, ax=ax2, label='Noise Ratio')
        
        # Number of Clusters vs Noise Ratio
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(top_results['n_clusters'], top_results['noise_ratio'],
                              c=top_results['composite_score'], cmap='coolwarm', alpha=0.7, s=50)
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Noise Ratio')
        ax3.set_title('Clusters vs Noise Trade-off')
        plt.colorbar(scatter3, ax=ax3, label='Composite Score')
        
        # Silhouette Score Distribution
        ax4 = axes[1, 1]
        valid_silhouette = top_results['silhouette_score'].dropna()
        if len(valid_silhouette) > 0:
            ax4.hist(valid_silhouette, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(valid_silhouette.mean(), color='red', linestyle='--', 
                       label=f'Mean: {valid_silhouette.mean():.3f}')
            ax4.set_xlabel('Silhouette Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Silhouette Score Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No valid silhouette scores', 
                    transform=ax4.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_parameter_explorer(self):
        """
        Create interactive Plotly visualization for exploring parameters
        
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        if self.df_results.empty:
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score vs PCA Components', 'Score vs Min Cluster Size',
                          'Clusters vs Noise Ratio', 'Runtime Analysis'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Prepare hover text
        hover_text = []
        for _, row in self.df_results.iterrows():
            text = (f"PCA: {row['pca_components']}<br>"
                   f"Min Cluster Size: {row['min_cluster_size']}<br>"
                   f"Min Samples: {row['min_samples']}<br>"
                   f"Metric: {row['metric']}<br>"
                   f"Clusters: {row['n_clusters']}<br>"
                   f"Noise Ratio: {row['noise_ratio']:.3f}<br>"
                   f"Score: {row['composite_score']:.3f}")
            hover_text.append(text)
        
        # Plot 1: Score vs PCA Components
        fig.add_trace(
            go.Scatter(
                x=self.df_results['pca_components'],
                y=self.df_results['composite_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df_results['n_clusters'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Clusters", x=0.48)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Score vs PCA'
            ),
            row=1, col=1
        )
        
        # Plot 2: Score vs Min Cluster Size
        fig.add_trace(
            go.Scatter(
                x=self.df_results['min_cluster_size'],
                y=self.df_results['composite_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df_results['noise_ratio'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Noise Ratio", x=1.02)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Score vs Min Cluster Size'
            ),
            row=1, col=2
        )
        
        # Plot 3: Clusters vs Noise Ratio
        fig.add_trace(
            go.Scatter(
                x=self.df_results['n_clusters'],
                y=self.df_results['noise_ratio'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.df_results['composite_score'],
                    colorscale='RdYlBu',
                    showscale=False
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Clusters vs Noise'
            ),
            row=2, col=1
        )
        
        # Plot 4: Runtime Analysis
        fig.add_trace(
            go.Scatter(
                x=self.df_results['composite_score'],
                y=self.df_results['runtime'],
                mode='markers',
                marker=dict(
                    size=self.df_results['pca_components'],
                    color=self.df_results['min_cluster_size'],
                    colorscale='Cividis',
                    showscale=False,
                    sizemode='diameter',
                    sizeref=0.3
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Score vs Runtime'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive Parameter Search Results Explorer',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="PCA Components", row=1, col=1)
        fig.update_yaxes(title_text="Composite Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Min Cluster Size", row=1, col=2)
        fig.update_yaxes(title_text="Composite Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Number of Clusters", row=2, col=1)
        fig.update_yaxes(title_text="Noise Ratio", row=2, col=1)
        
        fig.update_xaxes(title_text="Composite Score", row=2, col=2)
        fig.update_yaxes(title_text="Runtime (seconds)", row=2, col=2)
        
        return fig
    
    def create_heatmap_analysis(self):
        """
        Create heatmap showing parameter interactions
        
        Returns:
            matplotlib.figure.Figure: Heatmap figure
        """
        if self.df_results.empty:
            return None
        
        # Create pivot tables for different metric combinations
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: PCA Components vs Min Cluster Size (Composite Score)
        pivot1 = self.df_results.pivot_table(
            values='composite_score', 
            index='pca_components', 
            columns='min_cluster_size',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Composite Score Heatmap\n(PCA Components vs Min Cluster Size)')
        axes[0].set_xlabel('Min Cluster Size')
        axes[0].set_ylabel('PCA Components')
        
        # Heatmap 2: Min Cluster Size vs Min Samples (Number of Clusters)
        pivot2 = self.df_results.pivot_table(
            values='n_clusters',
            index='min_cluster_size',
            columns='min_samples',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='viridis', ax=axes[1])
        axes[1].set_title('Number of Clusters Heatmap\n(Min Cluster Size vs Min Samples)')
        axes[1].set_xlabel('Min Samples')
        axes[1].set_ylabel('Min Cluster Size')
        
        plt.tight_layout()
        return fig
    
    def create_top_parameters_summary(self, top_n=10):
        """
        Create summary table of top parameter combinations
        
        Args:
            top_n (int): Number of top results to show
            
        Returns:
            pandas.DataFrame: Top parameter combinations
        """
        if self.df_results.empty:
            return pd.DataFrame()
        
        # Select relevant columns for summary
        summary_cols = [
            'pca_components', 'min_cluster_size', 'min_samples', 'metric', 'alpha',
            'composite_score', 'n_clusters', 'noise_ratio', 'silhouette_score'
        ]
        
        top_results = self.df_results.head(top_n)[summary_cols].round(3)
        top_results.index = range(1, len(top_results) + 1)
        top_results.index.name = 'Rank'
        
        return top_results
    
    def save_all_visualizations(self, output_prefix="parameter_search"):
        """
        Save all visualizations to files
        
        Args:
            output_prefix (str): Prefix for output files
        """
        # Save matplotlib plots
        score_fig = self.create_score_comparison_plot()
        if score_fig:
            score_fig.savefig(f"{output_prefix}_score_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(score_fig)
        
        heatmap_fig = self.create_heatmap_analysis()
        if heatmap_fig:
            heatmap_fig.savefig(f"{output_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(heatmap_fig)
        
        # Save interactive plot
        interactive_fig = self.create_interactive_parameter_explorer()
        if interactive_fig:
            pyo.plot(interactive_fig, filename=f"{output_prefix}_interactive.html", auto_open=False)
        
        # Save summary table
        summary_table = self.create_top_parameters_summary()
        if not summary_table.empty:
            summary_table.to_csv(f"{output_prefix}_top_parameters.csv")
        
        print(f"Visualizations saved:")
        print(f"- {output_prefix}_score_comparison.png")
        print(f"- {output_prefix}_heatmap.png") 
        print(f"- {output_prefix}_interactive.html")
        print(f"- {output_prefix}_top_parameters.csv")


def visualize_parameter_search_results(search_results, output_prefix="parameter_analysis"):
    """
    Convenience function to create all visualizations
    
    Args:
        search_results (dict): Results from parameter search
        output_prefix (str): Prefix for output files
    """
    visualizer = ParameterSearchVisualizer(search_results)
    
    # Create and show summary
    print("Top 10 Parameter Combinations:")
    print("=" * 50)
    summary = visualizer.create_top_parameters_summary(10)
    print(summary.to_string())
    print("\n")
    
    # Save all visualizations
    visualizer.save_all_visualizations(output_prefix)
    
    return visualizer


if __name__ == "__main__":
    print("Parameter Search Visualization Module")
    print("=" * 50)
    print("Usage:")
    print("1. Run parameter search: results = parameter_tuning.run_full_parameter_search(...)")
    print("2. Visualize results: visualize_parameter_search_results(results)")
    print("3. Check generated files: *.png, *.html, *.csv")