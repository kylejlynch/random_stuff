"""
PCA Analysis Module for Fraud Transaction Clustering
This module performs PCA on fraudulent transactions and saves the results to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import logging
from logging_config import get_logger

class PCAAnalyzer:
    def __init__(self, n_components=10):
        """
        Initialize PCA Analyzer
        
        Args:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.logger = get_logger(__name__)
        
    def load_and_filter_data(self, data_path, fraud_only=True):
        """
        Load dataset and filter for fraudulent transactions only
        
        Args:
            data_path (str): Path to the fraud dataset
            fraud_only (bool): If True, keep only fraudulent transactions
            
        Returns:
            pandas.DataFrame: Filtered dataset
        """
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if fraud_only:
            df_fraud = df[df['is_fraud'] == 1].copy()
            self.logger.info(f"Filtered to {len(df_fraud)} fraudulent transactions from {len(df)} total")
        else:
            df_fraud = df.copy()
            
        # Add required columns for visualization
        # Create a simulated 'alerted' column (randomly assign alerts to some fraud cases)
        np.random.seed(42)
        df_fraud['alerted'] = np.random.choice([0, 1], size=len(df_fraud), p=[0.3, 0.7])
        
        # Create a simulated 'return_reason' column with different fraud types
        fraud_reasons = ['card_theft', 'identity_theft', 'account_takeover', 'synthetic_identity', 'first_party_fraud']
        df_fraud['return_reason'] = np.random.choice(fraud_reasons, size=len(df_fraud))
        
        return df_fraud
    
    def prepare_features(self, df, feature_columns=None):
        """
        Prepare numerical features for PCA
        
        Args:
            df (pandas.DataFrame): Input dataframe
            feature_columns (list): List of columns to use for PCA. If None, use default numerical columns
            
        Returns:
            numpy.array: Prepared feature matrix
        """
        if feature_columns is None:
            # Default numerical features that make sense for fraud analysis
            feature_columns = [
                'amt',           # Transaction amount
                'lat',           # Customer latitude
                'long',          # Customer longitude
                'city_pop',      # City population
                'unix_time',     # Transaction timestamp
                'merch_lat',     # Merchant latitude
                'merch_long',    # Merchant longitude
                'zip'            # ZIP code
            ]
        
        self.feature_names = feature_columns
        self.logger.info(f"Using features: {feature_columns}")
        
        # Select and prepare features
        X = df[feature_columns].copy()
        
        # Handle missing values
        self.logger.info("Handling missing values")
        X_imputed = self.imputer.fit_transform(X)
        
        # Standardize features
        self.logger.info("Standardizing features")
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit PCA and transform the data
        
        Args:
            X (numpy.array): Feature matrix
            
        Returns:
            numpy.array: PCA-transformed data
        """
        self.logger.info(f"Fitting PCA with {self.n_components} components")
        X_pca = self.pca.fit_transform(X)
        
        # Log explained variance
        explained_var_ratio = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        self.logger.info("Explained variance by component:")
        for i, (var, cum_var) in enumerate(zip(explained_var_ratio, cumulative_var)):
            self.logger.info(f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
        
        return X_pca
    
    def save_pca_results(self, X_pca, df_original, output_path):
        """
        Save PCA results to CSV file
        
        Args:
            X_pca (numpy.array): PCA-transformed data
            df_original (pandas.DataFrame): Original dataframe with metadata
            output_path (str): Path to save PCA results CSV
        """
        self.logger.info(f"Saving PCA results to {output_path}")
        
        # Create dataframe with PCA components
        pca_columns = [f'PC{i+1}' for i in range(self.n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_original.index)
        
        # Add important metadata columns for later use
        metadata_columns = ['alerted', 'return_reason', 'trans_num', 'amt', 'category']
        for col in metadata_columns:
            if col in df_original.columns:
                df_pca[col] = df_original[col].values
        
        # Save to CSV
        df_pca.to_csv(output_path, index=False)
        self.logger.info(f"PCA results saved with shape: {df_pca.shape}")
        
        return df_pca
    
    def run_full_analysis(self, data_path, feature_columns=None, output_path="pca_results.csv"):
        """
        Run complete PCA analysis pipeline
        
        Args:
            data_path (str): Path to input data
            feature_columns (list): Features to use for PCA
            output_path (str): Output CSV path
            
        Returns:
            pandas.DataFrame: PCA results dataframe
        """
        self.logger.info("Starting PCA Analysis")
        
        # Load and filter data
        df_fraud = self.load_and_filter_data(data_path)
        
        # Prepare features
        X_scaled = self.prepare_features(df_fraud, feature_columns)
        
        # Fit and transform
        X_pca = self.fit_transform(X_scaled)
        
        # Save results
        df_pca = self.save_pca_results(X_pca, df_fraud, output_path)
        
        self.logger.info("PCA Analysis Complete")
        return df_pca

if __name__ == "__main__":
    # Example usage
    pca_analyzer = PCAAnalyzer(n_components=10)
    
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
    
    # Run analysis
    df_pca = pca_analyzer.run_full_analysis(
        data_path="fraudTrain.csv",
        feature_columns=features,
        output_path="pca_results.csv"
    )
    
    logger = get_logger(__name__)
    logger.info("PCA analysis completed. Results saved to pca_results.csv")