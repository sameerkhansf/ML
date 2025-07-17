"""
Visualization Suite for the Blood-Brain Barrier Permeability Prediction Project.

This module provides comprehensive visualization functions for model performance,
feature analysis, and chemical space exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

from config import PLOT_CONFIG, PATHS

# Set up logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use(PLOT_CONFIG['style'])
sns.set_palette(PLOT_CONFIG['color_palette'])

class VisualizationSuite:
    """
    Class for creating comprehensive visualizations for BBB permeability analysis.
    """
    
    def __init__(self):
        """Initialize the VisualizationSuite."""
        self.figures = {}
        
    def plot_model_performance(self, models_results, save_path=None):
        """
        Plot model performance with ROC curves and confusion matrices.
        
        Parameters
        ----------
        models_results : dict
            Dictionary with model names as keys and results dictionaries as values.
            Each result dict should contain 'y_true', 'y_pred', 'y_prob'.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating model performance visualization")
        
        n_models = len(models_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        
        for i, (model_name, results) in enumerate(models_results.items()):
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_prob = results['y_prob']
            
            # ROC Curve (top row)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            axes[0, i].plot(fpr, tpr, color=colors[i], lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, i].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            axes[0, i].set_xlim([0.0, 1.0])
            axes[0, i].set_ylim([0.0, 1.05])
            axes[0, i].set_xlabel('False Positive Rate')
            axes[0, i].set_ylabel('True Positive Rate')
            axes[0, i].set_title(f'{model_name} - ROC Curve')
            axes[0, i].legend(loc="lower right")
            axes[0, i].grid(True, alpha=0.3)
            
            # Confusion Matrix (bottom row)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Permeable', 'Permeable'],
                       yticklabels=['Non-Permeable', 'Permeable'],
                       ax=axes[1, i])
            axes[1, i].set_title(f'{model_name} - Confusion Matrix')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        self.figures['model_performance'] = fig
        return fig
    
    def plot_feature_importance(self, feature_importance, top_n=20, save_path=None):
        """
        Plot feature importance for model interpretability.
        
        Parameters
        ----------
        feature_importance : pandas.Series or dict
            Feature importance scores with feature names as index/keys.
        top_n : int, default=20
            Number of top features to display.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info(f"Creating feature importance plot for top {top_n} features")
        
        # Convert to pandas Series if needed
        if isinstance(feature_importance, dict):
            feature_importance = pd.Series(feature_importance)
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        
        bars = ax.barh(range(len(top_features)), top_features.values, 
                      color='steelblue', alpha=0.8)
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top Molecular Descriptors for BBB Permeability Prediction', 
                    fontsize=14, fontweight='bold')
        
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
        
        self.figures['feature_importance'] = fig
        return fig
    
    def plot_chemical_space(self, X, y, feature_names=None, save_path=None):
        """
        Plot chemical space using PCA colored by permeability.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix (molecular descriptors).
        y : pandas.Series or numpy.ndarray
            Target variable (BBB permeability).
        feature_names : list, optional
            Names of features for PCA component interpretation.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating chemical space visualization using PCA")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Standardize features for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
        
        # Create scatter plot colored by permeability
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], 
            c=y_array, 
            cmap='RdYlBu_r',
            alpha=0.7,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('BBB Permeability', rotation=270, labelpad=20, fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Non-Permeable', 'Permeable'])
        
        # Add labels and title
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
        ax.set_title('Chemical Space Analysis: PCA of Molecular Descriptors', 
                    fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add text box with variance explained
        textstr = f'Total variance explained: {explained_var.sum():.1%}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Chemical space plot saved to {save_path}")
        
        self.figures['chemical_space'] = fig
        return fig
    
    def plot_descriptor_distributions(self, data, descriptors=None, save_path=None):
        """
        Plot descriptor distributions comparing permeable vs non-permeable molecules.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing molecular descriptors and permeability labels.
        descriptors : list, optional
            List of descriptor names to plot. If None, plots key descriptors.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating descriptor distribution plots")
        
        # Default key descriptors if not specified
        if descriptors is None:
            key_descriptors = ['MolLogP', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
            descriptors = [d for d in key_descriptors if d in data.columns]
        
        # Ensure we have the permeability column
        if 'p_np' not in data.columns:
            raise ValueError("Data must contain 'p_np' column for permeability labels")
        
        n_descriptors = len(descriptors)
        n_cols = 3
        n_rows = (n_descriptors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, descriptor in enumerate(descriptors):
            if descriptor not in data.columns:
                logger.warning(f"Descriptor {descriptor} not found in data")
                continue
            
            ax = axes[i]
            
            # Create histograms for each class
            permeable = data[data['p_np'] == 1][descriptor].dropna()
            non_permeable = data[data['p_np'] == 0][descriptor].dropna()
            
            # Plot histograms
            ax.hist(non_permeable, bins=30, alpha=0.7, label='Non-Permeable', 
                   color='lightcoral', density=True)
            ax.hist(permeable, bins=30, alpha=0.7, label='Permeable', 
                   color='lightblue', density=True)
            
            # Customize plot
            ax.set_xlabel(descriptor, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Distribution of {descriptor}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_perm = permeable.mean()
            mean_non_perm = non_permeable.mean()
            ax.axvline(mean_perm, color='blue', linestyle='--', alpha=0.8, 
                      label=f'Permeable mean: {mean_perm:.2f}')
            ax.axvline(mean_non_perm, color='red', linestyle='--', alpha=0.8,
                      label=f'Non-permeable mean: {mean_non_perm:.2f}')
        
        # Hide unused subplots
        for i in range(len(descriptors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Descriptor distributions plot saved to {save_path}")
        
        self.figures['descriptor_distributions'] = fig
        return fig
    
    def plot_correlation_matrix(self, data, descriptors=None, save_path=None):
        """
        Plot correlation matrix heatmap for molecular descriptors.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing molecular descriptors.
        descriptors : list, optional
            List of descriptor names to include. If None, uses all numeric columns.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating correlation matrix heatmap")
        
        # Select descriptors
        if descriptors is None:
            # Use all numeric columns except target variable
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            descriptors = [col for col in numeric_cols if col != 'p_np']
        
        # Calculate correlation matrix
        corr_data = data[descriptors]
        correlation_matrix = corr_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f', ax=ax)
        
        ax.set_title('Molecular Descriptors Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        self.figures['correlation_matrix'] = fig
        return fig
    
    def plot_model_comparison(self, comparison_results, save_path=None):
        """
        Plot model comparison with multiple metrics.
        
        Parameters
        ----------
        comparison_results : pandas.DataFrame
            DataFrame with models as index and metrics as columns.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating model comparison plot")
        
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in key_metrics if m in comparison_results.columns]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        x = np.arange(len(comparison_results.index))
        width = 0.15
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
        
        for i, metric in enumerate(available_metrics):
            values = comparison_results[metric].values
            ax.bar(x + i*width, values, width, label=metric.upper(), 
                  color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(x[j] + i*width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(comparison_results.index, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        self.figures['model_comparison'] = fig
        return fig
    
    def create_analysis_dashboard(self, data, models_results, feature_importance, save_path=None):
        """
        Create a comprehensive analysis dashboard with multiple subplots.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing molecular descriptors and permeability labels.
        models_results : dict
            Dictionary with model results for performance plotting.
        feature_importance : pandas.Series
            Feature importance scores.
        save_path : str or Path, optional
            Path to save the plot.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Creating comprehensive analysis dashboard")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Class distribution
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = data['p_np'].value_counts()
        ax1.pie(class_counts.values, labels=['Non-Permeable', 'Permeable'], 
               autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
        ax1.set_title('BBB Permeability Distribution', fontweight='bold')
        
        # 2. Feature importance (top 10)
        ax2 = fig.add_subplot(gs[0, 1:])
        top_features = feature_importance.head(10)
        bars = ax2.barh(range(len(top_features)), top_features.values, color='steelblue')
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features.index)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 10 Most Important Features', fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Chemical space (PCA)
        ax3 = fig.add_subplot(gs[1, 0])
        # Get descriptor columns (exclude non-numeric and target)
        descriptor_cols = data.select_dtypes(include=[np.number]).columns
        descriptor_cols = [col for col in descriptor_cols if col != 'p_np']
        
        if len(descriptor_cols) > 0:
            X = data[descriptor_cols].fillna(0)
            y = data['p_np']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu_r', alpha=0.6)
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax3.set_title('Chemical Space (PCA)', fontweight='bold')
        
        # 4. Key descriptor distributions
        ax4 = fig.add_subplot(gs[1, 1])
        if 'MolLogP' in data.columns:
            permeable = data[data['p_np'] == 1]['MolLogP'].dropna()
            non_permeable = data[data['p_np'] == 0]['MolLogP'].dropna()
            ax4.hist(non_permeable, bins=20, alpha=0.7, label='Non-Permeable', color='lightcoral')
            ax4.hist(permeable, bins=20, alpha=0.7, label='Permeable', color='lightblue')
            ax4.set_xlabel('MolLogP')
            ax4.set_ylabel('Frequency')
            ax4.set_title('LogP Distribution', fontweight='bold')
            ax4.legend()
        
        # 5. Model performance comparison
        ax5 = fig.add_subplot(gs[1, 2])
        if models_results:
            model_names = list(models_results.keys())
            auc_scores = []
            for model_name, results in models_results.items():
                from sklearn.metrics import roc_auc_score
                auc_scores.append(roc_auc_score(results['y_true'], results['y_prob']))
            
            bars = ax5.bar(model_names, auc_scores, color='lightgreen', alpha=0.8)
            ax5.set_ylabel('AUC-ROC Score')
            ax5.set_title('Model Performance (AUC)', fontweight='bold')
            ax5.set_ylim(0, 1)
            plt.setp(ax5.get_xticklabels(), rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, auc_scores):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # 6. Correlation heatmap (subset)
        ax6 = fig.add_subplot(gs[2, :])
        key_descriptors = ['MolLogP', 'MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors']
        available_descriptors = [d for d in key_descriptors if d in data.columns]
        
        if len(available_descriptors) > 1:
            corr_matrix = data[available_descriptors].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax6, cbar_kws={"shrink": .8})
            ax6.set_title('Key Descriptors Correlation Matrix', fontweight='bold')
        
        plt.suptitle('BBB Permeability Prediction Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Analysis dashboard saved to {save_path}")
        
        self.figures['dashboard'] = fig
        return fig
    
    def save_all_figures(self, output_dir=None):
        """
        Save all generated figures to files.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save figures. If None, uses plots directory from config.
            
        Returns
        -------
        dict
            Dictionary mapping figure names to saved file paths.
        """
        if output_dir is None:
            output_dir = PATHS['plots']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for fig_name, fig in self.figures.items():
            file_path = output_dir / f"{fig_name}.{PLOT_CONFIG['save_format']}"
            fig.savefig(file_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
            saved_files[fig_name] = file_path
            logger.info(f"Saved {fig_name} to {file_path}")
        
        return saved_files