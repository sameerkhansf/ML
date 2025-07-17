"""
Interpretability Engine for the Blood-Brain Barrier Permeability Prediction Project.

This module provides functionality for model interpretability, feature importance analysis,
and SHAP-based explanations for BBB permeability predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import logging
from pathlib import Path

from config import SHAP_CONFIG, PLOT_CONFIG, PATHS

# Set up logging
logger = logging.getLogger(__name__)

class InterpretabilityEngine:
    """
    Class for providing model interpretability and feature analysis for BBB permeability prediction.
    """
    
    def __init__(self):
        """Initialize the InterpretabilityEngine."""
        self.feature_importance = None
        self.shap_values = None
        self.shap_explainer = None
        self.pca_model = None
        self.feature_names = None
        
    def calculate_feature_importance(self, model, X, y=None, method='built_in'):
        """
        Calculate feature importance from trained models.
        
        Parameters
        ----------
        model : object
            Trained model with feature importance or predict method.
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray, optional
            Target variable, required for permutation importance.
        method : str, default='built_in'
            Method for calculating importance: 'built_in' or 'permutation'.
            
        Returns
        -------
        pandas.Series
            Feature importance scores.
        """
        logger.info(f"Calculating feature importance using {method} method")
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if method == 'built_in':
            # Use built-in feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_scores = np.abs(model.coef_[0])
            else:
                logger.warning("Model doesn't have built-in feature importance. Using permutation importance.")
                return self.calculate_feature_importance(model, X, y, method='permutation')
                
        elif method == 'permutation':
            # Use permutation importance (model-agnostic)
            if y is None:
                raise ValueError("Target variable y is required for permutation importance")
            
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )
            importance_scores = perm_importance.importances_mean
            
        else:
            raise ValueError("Method must be 'built_in' or 'permutation'")
        
        # Create feature importance series
        self.feature_importance = pd.Series(
            importance_scores, 
            index=self.feature_names
        ).sort_values(ascending=False)
        
        logger.info(f"Calculated feature importance for {len(self.feature_importance)} features")
        
        return self.feature_importance
    
    def generate_shap_analysis(self, model, X, sample_size=None):
        """
        Generate SHAP analysis for model explanations.
        
        Parameters
        ----------
        model : object
            Trained model for SHAP analysis.
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        sample_size : int, optional
            Number of samples to use for SHAP analysis. If None, uses config value.
            
        Returns
        -------
        numpy.ndarray
            SHAP values for the samples.
        """
        logger.info("Generating SHAP analysis for model explanations")
        
        if sample_size is None:
            sample_size = SHAP_CONFIG['sample_size']
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Sample data for SHAP analysis to speed up computation
        if len(X_array) > sample_size:
            sample_indices = np.random.choice(len(X_array), sample_size, replace=False)
            X_sample = X_array[sample_indices]
        else:
            X_sample = X_array
        
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(model)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
                
                # For binary classification, use positive class SHAP values
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                    
            else:
                # Fallback to KernelExplainer for other models
                background = shap.sample(X_array, min(100, len(X_array)))
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
                
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                    
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Trying LinearExplainer.")
            try:
                # Try LinearExplainer for linear models
                self.shap_explainer = shap.LinearExplainer(model, X_sample)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
            except Exception as e2:
                logger.warning(f"LinearExplainer failed: {e2}. Using KernelExplainer.")
                # Final fallback to KernelExplainer
                background = shap.sample(X_array, min(50, len(X_array)))
                self.shap_explainer = shap.KernelExplainer(model.predict, background)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        logger.info(f"Generated SHAP values for {len(X_sample)} samples")
        
        return self.shap_values
    
    def analyze_chemical_space(self, X, y, n_components=2, save_path=None):
        """
        Analyze chemical space using PCA visualization colored by permeability.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix (molecular descriptors).
        y : pandas.Series or numpy.ndarray
            Target variable (BBB permeability).
        n_components : int, default=2
            Number of principal components for visualization.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Analyzing chemical space using PCA")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = None
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Standardize features for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        # Apply PCA
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
        
        # Create scatter plot colored by permeability
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], 
            c=y_array, 
            cmap='RdYlBu_r',
            alpha=0.6,
            s=50
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('BBB Permeability', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Non-Permeable', 'Permeable'])
        
        # Add labels and title
        explained_var = self.pca_model.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax.set_title('Chemical Space Analysis: PCA of Molecular Descriptors')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Chemical space plot saved to {save_path}")
        
        logger.info(f"PCA explains {explained_var.sum():.1%} of total variance")
        
        return fig
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance as a horizontal bar chart.
        
        Parameters
        ----------
        top_n : int, default=20
            Number of top features to display.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call calculate_feature_importance first.")
        
        logger.info(f"Plotting top {top_n} feature importances")
        
        # Get top features
        top_features = self.feature_importance.head(top_n)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        bars = ax.barh(range(len(top_features)), top_features.values, color='steelblue')
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Molecular Descriptors for BBB Permeability Prediction')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            ax.text(value + max(top_features.values) * 0.01, i, f'{value:.3f}', 
                   va='center', fontsize=9)
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_shap_summary(self, X=None, save_path=None):
        """
        Plot SHAP summary plot showing feature importance and impact.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Feature matrix used for SHAP analysis.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call generate_shap_analysis first.")
        
        logger.info("Creating SHAP summary plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=SHAP_CONFIG['plot_size'])
        
        # Create SHAP summary plot
        if X is not None:
            if isinstance(X, pd.DataFrame):
                X_display = X.iloc[:len(self.shap_values)]
            else:
                X_display = X[:len(self.shap_values)]
        else:
            X_display = None
        
        shap.summary_plot(
            self.shap_values, 
            X_display,
            feature_names=self.feature_names,
            max_display=SHAP_CONFIG['max_display'],
            show=False
        )
        
        plt.title('SHAP Feature Importance for BBB Permeability Prediction')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        return fig
    
    def explain_prediction(self, model, X_sample, sample_idx=0, save_path=None):
        """
        Explain individual prediction using SHAP waterfall plot.
        
        Parameters
        ----------
        model : object
            Trained model.
        X_sample : pandas.DataFrame or numpy.ndarray
            Sample(s) to explain.
        sample_idx : int, default=0
            Index of the sample to explain.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.shap_explainer is None:
            logger.info("SHAP explainer not found. Generating SHAP analysis first.")
            self.generate_shap_analysis(model, X_sample)
        
        logger.info(f"Explaining prediction for sample {sample_idx}")
        
        # Get SHAP values for the specific sample
        if isinstance(X_sample, pd.DataFrame):
            X_array = X_sample.values
        else:
            X_array = X_sample
        
        if sample_idx >= len(X_array):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        # Create explanation
        if hasattr(self.shap_explainer, 'expected_value'):
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1]  # For binary classification
        else:
            expected_value = 0
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(self.shap_values.shape) == 2:
            sample_shap_values = self.shap_values[sample_idx]
        else:
            sample_shap_values = self.shap_values
        
        # Create a simple waterfall-style explanation
        feature_values = X_array[sample_idx]
        
        # Get top contributing features
        abs_shap = np.abs(sample_shap_values)
        top_indices = np.argsort(abs_shap)[-10:][::-1]
        
        top_shap = sample_shap_values[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        top_values = feature_values[top_indices]
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'blue' for x in top_shap]
        bars = ax.barh(range(len(top_shap)), top_shap, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(top_shap)))
        ax.set_yticklabels([f"{feat} = {val:.2f}" for feat, val in zip(top_features, top_values)])
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.set_title(f'SHAP Explanation for Sample {sample_idx}')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_shap)):
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                   va='center', ha='left' if value >= 0 else 'right', fontsize=9)
        
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Individual prediction explanation saved to {save_path}")
        
        return fig
    
    def get_top_features(self, n=10):
        """
        Get the top n most important features.
        
        Parameters
        ----------
        n : int, default=10
            Number of top features to return.
            
        Returns
        -------
        pandas.Series
            Top n features with their importance scores.
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call calculate_feature_importance first.")
        
        return self.feature_importance.head(n)
    
    def generate_insights_report(self):
        """
        Generate a text report with key insights from the interpretability analysis.
        
        Returns
        -------
        str
            Formatted insights report.
        """
        if self.feature_importance is None:
            return "No feature importance analysis available. Run calculate_feature_importance first."
        
        report = []
        report.append("=== BBB Permeability Prediction: Key Insights ===\n")
        
        # Top features
        top_5 = self.get_top_features(5)
        report.append("Top 5 Most Important Molecular Descriptors:")
        for i, (feature, importance) in enumerate(top_5.items(), 1):
            report.append(f"{i}. {feature}: {importance:.4f}")
        
        report.append("")
        
        # Feature categories analysis
        lipophilicity_features = [f for f in self.feature_importance.index if 'LogP' in f or 'MolWt' in f]
        polarity_features = [f for f in self.feature_importance.index if 'TPSA' in f or 'PSA' in f]
        hb_features = [f for f in self.feature_importance.index if 'HBD' in f or 'HBA' in f or 'HDonor' in f or 'HAcceptor' in f]
        
        if lipophilicity_features:
            avg_lipophilicity_importance = self.feature_importance[lipophilicity_features].mean()
            report.append(f"Lipophilicity features average importance: {avg_lipophilicity_importance:.4f}")
        
        if polarity_features:
            avg_polarity_importance = self.feature_importance[polarity_features].mean()
            report.append(f"Polarity features average importance: {avg_polarity_importance:.4f}")
        
        if hb_features:
            avg_hb_importance = self.feature_importance[hb_features].mean()
            report.append(f"Hydrogen bonding features average importance: {avg_hb_importance:.4f}")
        
        report.append("")
        
        # PCA analysis if available
        if self.pca_model is not None:
            explained_var = self.pca_model.explained_variance_ratio_
            report.append(f"Chemical space analysis (PCA):")
            report.append(f"- First 2 components explain {explained_var[:2].sum():.1%} of molecular diversity")
            report.append(f"- PC1 explains {explained_var[0]:.1%}, PC2 explains {explained_var[1]:.1%}")
        
        report.append("")
        report.append("=== Recommendations for BBB-Permeable Drug Design ===")
        
        # Generate recommendations based on top features
        top_feature = top_5.index[0]
        if 'LogP' in top_feature:
            report.append("• Optimize lipophilicity (LogP) as it's the most predictive feature")
        elif 'TPSA' in top_feature:
            report.append("• Control polar surface area (TPSA) to improve BBB permeability")
        elif 'MolWt' in top_feature:
            report.append("• Consider molecular weight optimization for BBB penetration")
        
        report.append("• Balance lipophilicity and polarity for optimal BBB permeability")
        report.append("• Consider hydrogen bonding potential in molecular design")
        
        return "\n".join(report)