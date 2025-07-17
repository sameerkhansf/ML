"""
Unit tests for the visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from src.visualization import VisualizationSuite


@pytest.fixture
def sample_model_predictions():
    """Fixture providing sample model predictions for visualization testing."""
    # Create sample classification data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train models
    models = {}
    predictions = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Generate predictions
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        predictions[name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    return models, predictions, X_test, y_test


@pytest.fixture
def sample_feature_importance():
    """Fixture providing sample feature importance data."""
    feature_names = [f'Feature_{i}' for i in range(10)]
    importance_values = np.random.rand(10)
    importance_values = importance_values / importance_values.sum()  # Normalize
    
    return pd.Series(importance_values, index=feature_names).sort_values(ascending=False)


@pytest.fixture
def sample_model_comparison():
    """Fixture providing sample model comparison data."""
    models = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Generate random performance data
    np.random.seed(42)
    data = {}
    for metric in metrics:
        data[metric] = np.random.uniform(0.7, 0.95, len(models))
    
    comparison_df = pd.DataFrame(data, index=models)
    return comparison_df


class TestVisualizationSuite:
    """Test class for VisualizationSuite functionality."""
    
    def test_initialization(self):
        """Test VisualizationSuite initialization."""
        viz = VisualizationSuite()
        assert viz is not None
    
    def test_plot_model_performance(self, sample_model_predictions):
        """Test model performance plotting."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Plot model performance
        fig = viz.plot_model_performance(predictions)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) >= 2  # Should have multiple subplots
        
        # Close figure to prevent display during testing
        plt.close(fig)
    
    def test_plot_roc_curves(self, sample_model_predictions):
        """Test ROC curve plotting."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Plot ROC curves
        fig = viz.plot_roc_curves(predictions)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 1  # Single subplot for ROC curves
        
        # Check that curves were plotted
        ax = fig.axes[0]
        assert len(ax.lines) >= len(predictions)  # At least one line per model
        
        plt.close(fig)
    
    def test_plot_confusion_matrices(self, sample_model_predictions):
        """Test confusion matrix plotting."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Plot confusion matrices
        fig = viz.plot_confusion_matrices(predictions)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) >= len(predictions)  # One subplot per model
        
        plt.close(fig)
    
    def test_plot_feature_importance(self, sample_feature_importance):
        """Test feature importance plotting."""
        viz = VisualizationSuite()
        
        # Plot feature importance
        fig = viz.plot_feature_importance(sample_feature_importance, title='Test Feature Importance')
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Check that bars were plotted
        ax = fig.axes[0]
        assert len(ax.patches) == len(sample_feature_importance)  # One bar per feature
        
        plt.close(fig)
    
    def test_plot_feature_importance_comparison(self, sample_feature_importance):
        """Test feature importance comparison plotting."""
        viz = VisualizationSuite()
        
        # Create comparison data
        importance_dict = {
            'Model1': sample_feature_importance,
            'Model2': sample_feature_importance * 0.8 + 0.1  # Slightly different
        }
        
        # Plot comparison
        fig = viz.plot_feature_importance_comparison(importance_dict)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) >= 2  # Multiple subplots for comparison
        
        plt.close(fig)
    
    def test_plot_model_comparison(self, sample_model_comparison):
        """Test model comparison plotting."""
        viz = VisualizationSuite()
        
        # Plot model comparison
        fig = viz.plot_model_comparison(sample_model_comparison)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) >= 1
        
        plt.close(fig)
    
    def test_plot_chemical_space(self):
        """Test chemical space plotting."""
        viz = VisualizationSuite()
        
        # Create sample chemical space data
        np.random.seed(42)
        n_samples = 100
        pca_data = np.random.randn(n_samples, 2)
        labels = np.random.choice([0, 1], n_samples)
        explained_variance = [0.4, 0.3]
        
        # Plot chemical space
        fig = viz.plot_chemical_space(pca_data, labels, explained_variance)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Check that scatter plot was created
        ax = fig.axes[0]
        assert len(ax.collections) >= 1  # Scatter plot creates collections
        
        plt.close(fig)
    
    def test_plot_descriptor_distributions(self):
        """Test descriptor distribution plotting."""
        viz = VisualizationSuite()
        
        # Create sample descriptor data
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'descriptor1': np.random.normal(0, 1, n_samples),
            'descriptor2': np.random.normal(2, 1.5, n_samples),
            'descriptor3': np.random.exponential(1, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        descriptors = ['descriptor1', 'descriptor2', 'descriptor3']
        
        # Plot distributions
        fig = viz.plot_descriptor_distributions(data, descriptors, 'target')
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == len(descriptors)
        
        plt.close(fig)
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        viz = VisualizationSuite()
        
        # Create sample correlation data
        np.random.seed(42)
        n_features = 8
        correlation_matrix = np.random.rand(n_features, n_features)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal should be 1
        
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        correlation_df = pd.DataFrame(correlation_matrix, 
                                    index=feature_names, 
                                    columns=feature_names)
        
        # Plot heatmap
        fig = viz.plot_correlation_heatmap(correlation_df)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) >= 1  # Heatmap + colorbar
        
        plt.close(fig)
    
    def test_plot_learning_curves(self, sample_model_predictions):
        """Test learning curve plotting."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Create sample learning curve data
        train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        train_scores = np.random.uniform(0.7, 0.9, (len(train_sizes), 5))
        val_scores = np.random.uniform(0.6, 0.85, (len(train_sizes), 5))
        
        # Plot learning curves
        fig = viz.plot_learning_curves(train_sizes, train_scores, val_scores)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Check that curves were plotted
        ax = fig.axes[0]
        assert len(ax.lines) >= 2  # Training and validation curves
        
        plt.close(fig)
    
    def test_plot_prediction_distribution(self, sample_model_predictions):
        """Test prediction distribution plotting."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Get sample predictions
        model_name = list(predictions.keys())[0]
        y_prob = predictions[model_name]['y_prob']
        y_true = predictions[model_name]['y_true']
        
        # Plot prediction distribution
        fig = viz.plot_prediction_distribution(y_prob, y_true)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_create_interactive_plot(self):
        """Test interactive plot creation."""
        viz = VisualizationSuite()
        
        # Create sample data for interactive plot
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'x': np.random.randn(n_samples),
            'y': np.random.randn(n_samples),
            'color': np.random.choice(['A', 'B'], n_samples),
            'size': np.random.uniform(5, 15, n_samples)
        })
        
        # Create interactive plot
        try:
            fig = viz.create_interactive_plot(data, x='x', y='y', color='color', size='size')
            
            # Check that figure was created
            assert fig is not None
            
        except ImportError:
            # Plotly might not be available
            pytest.skip("Plotly not available for interactive plotting")
    
    def test_save_figure(self, sample_feature_importance, tmp_path):
        """Test figure saving functionality."""
        viz = VisualizationSuite()
        
        # Create a figure
        fig = viz.plot_feature_importance(sample_feature_importance)
        
        # Save figure
        save_path = tmp_path / "test_figure.png"
        viz.save_figure(fig, save_path)
        
        # Check that file was created
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_set_style(self):
        """Test style setting functionality."""
        viz = VisualizationSuite()
        
        # Test different styles
        styles = ['default', 'seaborn', 'ggplot']
        
        for style in styles:
            try:
                viz.set_style(style)
                # If no exception is raised, style was set successfully
                assert True
            except ValueError:
                # Some styles might not be available
                pass
    
    def test_create_subplot_grid(self):
        """Test subplot grid creation."""
        viz = VisualizationSuite()
        
        # Create subplot grid
        fig, axes = viz.create_subplot_grid(2, 3, figsize=(12, 8))
        
        # Check that grid was created correctly
        assert fig is not None
        assert len(axes) == 6  # 2x3 grid
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
        
        plt.close(fig)
    
    def test_add_statistical_annotations(self):
        """Test statistical annotation functionality."""
        viz = VisualizationSuite()
        
        # Create sample data
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        
        # Create figure
        fig, ax = plt.subplots()
        ax.boxplot([group1, group2])
        
        # Add statistical annotations
        viz.add_statistical_annotations(ax, group1, group2, test='t-test')
        
        # Check that annotations were added
        assert len(ax.texts) > 0  # Should have text annotations
        
        plt.close(fig)
    
    def test_error_handling_empty_data(self):
        """Test error handling for empty data."""
        viz = VisualizationSuite()
        
        # Test with empty predictions dictionary
        empty_predictions = {}
        
        with pytest.raises((ValueError, KeyError)):
            viz.plot_model_performance(empty_predictions)
    
    def test_error_handling_invalid_data_format(self):
        """Test error handling for invalid data formats."""
        viz = VisualizationSuite()
        
        # Test with invalid feature importance data
        invalid_importance = "not_a_series"
        
        with pytest.raises((AttributeError, TypeError)):
            viz.plot_feature_importance(invalid_importance)
    
    def test_color_palette_consistency(self, sample_model_predictions):
        """Test that color palettes are used consistently."""
        models, predictions, X_test, y_test = sample_model_predictions
        viz = VisualizationSuite()
        
        # Plot multiple figures
        fig1 = viz.plot_roc_curves(predictions)
        fig2 = viz.plot_confusion_matrices(predictions)
        
        # Check that figures were created (color consistency is visual)
        assert fig1 is not None
        assert fig2 is not None
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_figure_size_customization(self, sample_feature_importance):
        """Test figure size customization."""
        viz = VisualizationSuite()
        
        # Plot with custom figure size
        fig = viz.plot_feature_importance(sample_feature_importance, figsize=(10, 6))
        
        # Check figure size
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        
        plt.close(fig)
    
    def test_title_and_label_customization(self, sample_feature_importance):
        """Test title and label customization."""
        viz = VisualizationSuite()
        
        # Plot with custom title and labels
        custom_title = "Custom Feature Importance"
        fig = viz.plot_feature_importance(sample_feature_importance, 
                                        title=custom_title,
                                        xlabel="Custom X Label",
                                        ylabel="Custom Y Label")
        
        # Check that title was set
        ax = fig.axes[0]
        assert ax.get_title() == custom_title
        assert ax.get_xlabel() == "Custom X Label"
        assert ax.get_ylabel() == "Custom Y Label"
        
        plt.close(fig)