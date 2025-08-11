"""
3D UMAP Visualization Module for Fraud Transaction Clustering
This module creates interactive 3D UMAP visualizations with dropdown menus for different coloring options.
"""

import pandas as pd
import numpy as np
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import logging
from logging_config import get_logger

class UMAPVisualizer:
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Initialize UMAP Visualizer
        
        Args:
            n_components (int): Number of dimensions for UMAP (should be 3 for 3D visualization)
            n_neighbors (int): Number of neighbors for UMAP
            min_dist (float): Minimum distance for UMAP
            random_state (int): Random state for reproducibility
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.umap_reducer = None
        self.embedding = None
        self.logger = get_logger(__name__)
        
    def load_clustering_data(self, clustering_file_path):
        """
        Load clustering results data from CSV file
        
        Args:
            clustering_file_path (str): Path to clustering results CSV file
            
        Returns:
            tuple: (PCA_features, metadata_dataframe, full_dataframe)
        """
        self.logger.info(f"Loading clustering data from {clustering_file_path}")
        df_clustering = pd.read_csv(clustering_file_path)
        
        # Separate PCA features from metadata
        pca_columns = [col for col in df_clustering.columns if col.startswith('PC')]
        metadata_columns = [col for col in df_clustering.columns if not col.startswith('PC')]
        
        X_pca = df_clustering[pca_columns].values
        metadata = df_clustering[metadata_columns].copy()
        
        self.logger.info(f"Loaded clustering data: {X_pca.shape[0]} samples, {X_pca.shape[1]} PCA components")
        self.logger.debug(f"Available metadata columns: {list(metadata.columns)}")
        
        return X_pca, metadata, df_clustering
    
    def fit_umap(self, X_pca):
        """
        Fit UMAP on PCA-transformed data
        
        Args:
            X_pca (numpy.array): PCA-transformed feature matrix
            
        Returns:
            numpy.array: UMAP embedding
        """
        self.logger.info("Fitting 3D UMAP")
        self.logger.info(f"UMAP parameters: n_neighbors={self.n_neighbors}, "
                        f"min_dist={self.min_dist}, n_components={self.n_components}")
        
        self.umap_reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )
        
        self.embedding = self.umap_reducer.fit_transform(X_pca)
        
        self.logger.info(f"UMAP embedding shape: {self.embedding.shape}")
        return self.embedding
    
    def prepare_color_options(self, df_full):
        """
        Prepare different coloring options for the visualization
        
        Args:
            df_full (pandas.DataFrame): Full dataframe with all columns
            
        Returns:
            dict: Dictionary of coloring options
        """
        color_options = {}
        
        # Check for required columns
        if 'alerted' in df_full.columns:
            color_options['alerted'] = df_full['alerted'].astype(str)
        
        if 'return_reason' in df_full.columns:
            color_options['return_reason'] = df_full['return_reason'].astype(str)
        
        if 'cluster_or_noise' in df_full.columns:
            color_options['cluster_or_noise'] = df_full['cluster_or_noise'].astype(str)
        
        # Add cluster labels as numeric
        if 'cluster_label' in df_full.columns:
            color_options['cluster_label'] = df_full['cluster_label'].astype(str)
        
        # Add transaction amount bins
        if 'amt' in df_full.columns:
            amt_bins = pd.qcut(df_full['amt'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            color_options['amount_tier'] = amt_bins.astype(str)
        
        # Add category if available
        if 'category' in df_full.columns:
            color_options['category'] = df_full['category'].astype(str)
        
        self.logger.info(f"Available color options: {list(color_options.keys())}")
        return color_options
    
    def create_3d_scatter(self, embedding, color_options, df_full, title="3D UMAP Visualization of Fraud Clusters"):
        """
        Create interactive 3D scatter plot with dropdown menus
        
        Args:
            embedding (numpy.array): UMAP 3D embedding
            color_options (dict): Dictionary of coloring options
            df_full (pandas.DataFrame): Full dataframe for hover information
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D plot
        """
        self.logger.info("Creating interactive 3D visualization")
        
        # Create base figure
        fig = go.Figure()
        
        # Get first color option as default
        default_color_key = list(color_options.keys())[0]
        default_colors = color_options[default_color_key]
        
        # Create color mapping
        unique_colors = default_colors.unique()
        color_palette = px.colors.qualitative.Set3
        if len(unique_colors) > len(color_palette):
            color_palette = px.colors.qualitative.Light24
        
        # Create hover text
        hover_text = []
        for i in range(len(df_full)):
            text = f"Point {i}<br>"
            if 'amt' in df_full.columns:
                text += f"Amount: ${df_full['amt'].iloc[i]:.2f}<br>"
            if 'cluster_label' in df_full.columns:
                text += f"Cluster: {df_full['cluster_label'].iloc[i]}<br>"
            if 'alerted' in df_full.columns:
                text += f"Alerted: {df_full['alerted'].iloc[i]}<br>"
            if 'return_reason' in df_full.columns:
                text += f"Fraud Type: {df_full['return_reason'].iloc[i]}<br>"
            if 'category' in df_full.columns:
                text += f"Category: {df_full['category'].iloc[i]}"
            hover_text.append(text)
        
        # Add traces for each unique value in default color option
        for i, unique_val in enumerate(unique_colors):
            mask = default_colors == unique_val
            color_idx = i % len(color_palette)
            
            fig.add_trace(go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1], 
                z=embedding[mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_palette[color_idx],
                    opacity=0.7
                ),
                name=str(unique_val),
                text=[hover_text[j] for j in range(len(hover_text)) if mask[j]],
                hovertemplate='%{text}<extra></extra>',
                visible=True
            ))
        
        # Now add traces for all other color options (initially hidden)
        trace_visibility = {}
        trace_count = len(unique_colors)
        
        for color_key in color_options.keys():
            if color_key == default_color_key:
                continue
                
            colors = color_options[color_key]
            unique_vals = colors.unique()
            trace_visibility[color_key] = []
            
            for i, unique_val in enumerate(unique_vals):
                mask = colors == unique_val
                color_idx = i % len(color_palette)
                
                fig.add_trace(go.Scatter3d(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    z=embedding[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_palette[color_idx],
                        opacity=0.7
                    ),
                    name=str(unique_val),
                    text=[hover_text[j] for j in range(len(hover_text)) if mask[j]],
                    hovertemplate='%{text}<extra></extra>',
                    visible=False
                ))
                trace_visibility[color_key].append(trace_count)
                trace_count += 1
        
        # Create dropdown buttons
        dropdown_buttons = []
        
        # Default option
        default_visibility = [True] * len(unique_colors) + [False] * (trace_count - len(unique_colors))
        dropdown_buttons.append(
            dict(
                args=[{"visible": default_visibility}],
                label=default_color_key.replace('_', ' ').title(),
                method="restyle"
            )
        )
        
        # Other options
        for color_key in color_options.keys():
            if color_key == default_color_key:
                continue
                
            visibility = [False] * trace_count
            for trace_idx in trace_visibility[color_key]:
                visibility[trace_idx] = True
            
            dropdown_buttons.append(
                dict(
                    args=[{"visible": visibility}],
                    label=color_key.replace('_', ' ').title(),
                    method="restyle"
                )
            )
        
        # Update layout with dropdown
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2", 
                zaxis_title="UMAP 3",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.4)",
                    borderwidth=1
                ),
            ],
            annotations=[
                dict(
                    text="Color by:",
                    x=0.05, y=1.08,
                    xref="paper", yref="paper",
                    align="left",
                    showarrow=False,
                    font=dict(size=12)
                )
            ],
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100)
        )
        
        return fig
    
    def save_html(self, fig, output_path):
        """
        Save interactive plot as HTML file
        
        Args:
            fig (plotly.graph_objects.Figure): Plotly figure
            output_path (str): Output HTML file path
        """
        self.logger.info(f"Saving interactive visualization to {output_path}")
        
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'fraud_clusters_3d',
                'height': 800,
                'width': 1000,
                'scale': 1
            },
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan3d', 'orbitRotation']
        }
        
        pyo.plot(fig, filename=output_path, auto_open=False, config=config)
        self.logger.info(f"Interactive HTML visualization saved to {output_path}")
    
    def run_full_visualization(self, clustering_file_path, output_html_path="fraud_clusters_3d.html"):
        """
        Run complete UMAP visualization pipeline
        
        Args:
            clustering_file_path (str): Path to clustering results CSV
            output_html_path (str): Output HTML file path
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot figure
        """
        self.logger.info("Starting 3D UMAP Visualization")
        
        # Load clustering data
        X_pca, metadata, df_full = self.load_clustering_data(clustering_file_path)
        
        # Fit UMAP
        embedding = self.fit_umap(X_pca)
        
        # Prepare color options
        color_options = self.prepare_color_options(df_full)
        
        # Create visualization
        fig = self.create_3d_scatter(embedding, color_options, df_full)
        
        # Save HTML
        self.save_html(fig, output_html_path)
        
        self.logger.info("3D UMAP Visualization Complete")
        return fig

if __name__ == "__main__":
    # Example usage
    visualizer = UMAPVisualizer(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    
    # Run visualization
    fig = visualizer.run_full_visualization(
        clustering_file_path="clustering_results.csv",
        output_html_path="fraud_clusters_3d.html"
    )
    
    logger = get_logger(__name__)
    logger.info("Visualization completed. Open fraud_clusters_3d.html in your browser to explore the clusters.")