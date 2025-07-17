"""
Feature Engineering Module for the Blood-Brain Barrier Permeability Prediction Project.

This module provides functionality for preprocessing molecular descriptors,
feature selection, and preparing the final feature matrix for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
import logging

from config import MODEL_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Class for preprocessing molecular descriptors and feature selection.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineering module."""
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.selected_features = None
        self.pca = None
        
    def scale_features(self, X, method='standard', fit=True):
        """
        Scale features using StandardScaler or MinMaxScaler.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix to scale.
        method : str, default='standard'
            Scaling method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
        fit : bool, default=True
            Whether to fit the scaler on the data or use a previously fitted scaler.
            
        Returns
        -------
        numpy.ndarray
            Scaled feature matrix.
        """
        if method not in ['standard', 'minmax']:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X_array = X.values
        else:
            X_array = X
            feature_names = None
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            X_scaled = self.scaler.fit_transform(X_array)
            logger.info(f"Fitted {method} scaler on {X_array.shape[1]} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            
            X_scaled = self.scaler.transform(X_array)
        
        # Return DataFrame if input was DataFrame
        if feature_names is not None:
            return pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
        
        return X_scaled
    
    def handle_missing_values(self, X, strategy='median', fit=True):
        """
        Handle missing values using imputation.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix with missing values.
        strategy : str, default='median'
            Imputation strategy: 'mean', 'median', 'most_frequent', or 'constant'.
        fit : bool, default=True
            Whether to fit the imputer on the data or use a previously fitted imputer.
            
        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Feature matrix with imputed values.
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X_array = X.values
        else:
            X_array = X
            feature_names = None
        
        if fit:
            self.imputer = SimpleImputer(strategy=strategy)
            X_imputed = self.imputer.fit_transform(X_array)
            logger.info(f"Fitted imputer with strategy '{strategy}'")
        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted. Call with fit=True first.")
            
            X_imputed = self.imputer.transform(X_array)
        
        # Return DataFrame if input was DataFrame
        if feature_names is not None:
            return pd.DataFrame(X_imputed, columns=feature_names, index=X.index)
        
        return X_imputed
    
    def select_features(self, X, y=None, method='variance', threshold=0.01, k=10, fit=True):
        """
        Select features based on variance or univariate statistical tests.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray, optional
            Target variable, required for 'univariate' method.
        method : str, default='variance'
            Feature selection method: 'variance' or 'univariate'.
        threshold : float, default=0.01
            Threshold for variance-based feature selection.
        k : int, default=10
            Number of top features to select for univariate selection.
        fit : bool, default=True
            Whether to fit the selector on the data or use a previously fitted selector.
            
        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Selected feature matrix.
        """
        if method not in ['variance', 'univariate']:
            raise ValueError("Method must be 'variance' or 'univariate'")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X_array = X.values
        else:
            X_array = X
            feature_names = None
        
        if fit:
            if method == 'variance':
                self.feature_selector = VarianceThreshold(threshold=threshold)
                X_selected = self.feature_selector.fit_transform(X_array)
                
                if feature_names is not None:
                    self.selected_features = feature_names[self.feature_selector.get_support()]
                
                logger.info(f"Selected {X_selected.shape[1]} features using variance threshold {threshold}")
                
            else:  # univariate
                if y is None:
                    raise ValueError("Target variable y is required for univariate feature selection")
                
                self.feature_selector = SelectKBest(f_classif, k=k)
                X_selected = self.feature_selector.fit_transform(X_array, y)
                
                if feature_names is not None:
                    self.selected_features = feature_names[self.feature_selector.get_support()]
                
                logger.info(f"Selected top {k} features using univariate statistical tests")
        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector not fitted. Call with fit=True first.")
            
            X_selected = self.feature_selector.transform(X_array)
        
        # Return DataFrame if input was DataFrame
        if feature_names is not None:
            if self.selected_features is not None:
                return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        return X_selected
    
    def create_feature_matrix(self, X, handle_missing=True, scale=True, select=True, y=None):
        """
        Create the final feature matrix for machine learning by applying preprocessing steps.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Raw feature matrix.
        handle_missing : bool, default=True
            Whether to handle missing values.
        scale : bool, default=True
            Whether to scale features.
        select : bool, default=True
            Whether to perform feature selection.
        y : pandas.Series, optional
            Target variable, required for univariate feature selection.
            
        Returns
        -------
        pandas.DataFrame
            Processed feature matrix ready for machine learning.
        """
        logger.info("Creating feature matrix for machine learning")
        
        X_processed = X.copy()
        
        # Handle missing values
        if handle_missing:
            X_processed = self.handle_missing_values(X_processed)
            logger.info("Handled missing values")
        
        # Scale features
        if scale:
            X_processed = self.scale_features(X_processed)
            logger.info("Scaled features")
        
        # Select features
        if select:
            X_processed = self.select_features(X_processed, y=y)
            logger.info(f"Selected {X_processed.shape[1]} features")
        
        return X_processed
    
    def apply_pca(self, X, n_components=2, fit=True):
        """
        Apply PCA for dimensionality reduction and visualization.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        n_components : int, default=2
            Number of principal components to keep.
        fit : bool, default=True
            Whether to fit PCA on the data or use a previously fitted PCA.
            
        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Transformed feature matrix with principal components.
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if fit:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X_array)
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            
            X_pca = self.pca.transform(X_array)
        
        # Create DataFrame with component names
        columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_pca, columns=columns, index=X.index)
        
        return pd.DataFrame(X_pca, columns=columns)