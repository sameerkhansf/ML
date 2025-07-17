"""
Unit tests for the machine learning models module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.models import ModelTrainer


@pytest.fixture
def sample_classification_data():
    """Fixture providing sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


class TestModelTrainer:
    """Test class for ModelTrainer functionality."""
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.best_model is None
        assert trainer.best_model_name is None
    
    def test_train_logistic_regression(self, sample_classification_data):
        """Test logistic regression training."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train model
        model, metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_train_random_forest(self, sample_classification_data):
        """Test random forest training."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train model
        model, metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')
        
        # Check feature importances
        assert len(model.feature_importances_) == X_train.shape[1]
        assert np.allclose(model.feature_importances_.sum(), 1.0)
        
        # Check metrics
        assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    def test_train_svm(self, sample_classification_data):
        """Test SVM training."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train model
        model, metrics = trainer.train_svm(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Check metrics
        assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    def test_train_xgboost(self, sample_classification_data):
        """Test XGBoost training."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train model
        model, metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')
        
        # Check feature importances
        assert len(model.feature_importances_) == X_train.shape[1]
        
        # Check metrics
        assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    def test_train_neural_network(self, sample_classification_data):
        """Test neural network training."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train model
        model, metrics = trainer.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Check metrics
        assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    def test_evaluate_model(self, sample_classification_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train a simple model
        model, _ = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        metrics = trainer.evaluate_model(model, X_test, y_test, 'Test Model')
        
        # Check all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        assert all(metric in metrics for metric in required_metrics)
        
        # Check metric ranges
        for metric in required_metrics:
            assert 0 <= metrics[metric] <= 1
    
    def test_cross_validate_models(self, sample_classification_data):
        """Test cross-validation of models."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train multiple models
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Cross-validate models
        cv_results = trainer.cross_validate_models(X_train, y_train, cv_folds=3)
        
        # Check results
        assert len(cv_results) == 2  # Two models trained
        
        for model_name, scores in cv_results.items():
            assert 'mean_score' in scores
            assert 'std_score' in scores
            assert 'scores' in scores
            assert len(scores['scores']) == 3  # 3-fold CV
            assert 0 <= scores['mean_score'] <= 1
            assert scores['std_score'] >= 0
    
    def test_compare_models(self, sample_classification_data):
        """Test model comparison."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train multiple models
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Compare models
        comparison = trainer.compare_models()
        
        # Check comparison results
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2  # Two models
        assert 'Model' in comparison.columns
        assert 'AUC' in comparison.columns
        assert 'Accuracy' in comparison.columns
        
        # Check that results are sorted by AUC (descending)
        auc_values = comparison['AUC'].values
        assert all(auc_values[i] >= auc_values[i+1] for i in range(len(auc_values)-1))
    
    def test_select_best_model(self, sample_classification_data):
        """Test best model selection."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train multiple models
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Select best model
        best_model, best_name, best_score = trainer.select_best_model(metric='auc')
        
        # Check best model selection
        assert best_model is not None
        assert best_name in trainer.models.keys()
        assert 0 <= best_score <= 1
        assert trainer.best_model == best_model
        assert trainer.best_model_name == best_name
    
    def test_save_and_load_model(self, sample_classification_data, tmp_path):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train a model
        model, _ = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        trainer.save_model(model, model_path)
        
        # Check that file was created
        assert model_path.exists()
        
        # Load model
        loaded_model = trainer.load_model(model_path)
        
        # Check that loaded model works
        assert loaded_model is not None
        predictions_original = model.predict(X_test)
        predictions_loaded = loaded_model.predict(X_test)
        
        # Predictions should be identical
        assert np.array_equal(predictions_original, predictions_loaded)
    
    def test_get_feature_importance(self, sample_classification_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train a tree-based model
        model, _ = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Get feature importance
        importance = trainer.get_feature_importance(model, feature_names=None)
        
        # Check importance
        assert isinstance(importance, pd.Series)
        assert len(importance) == X_train.shape[1]
        assert np.allclose(importance.sum(), 1.0)
        
        # Test with feature names
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        importance_named = trainer.get_feature_importance(model, feature_names)
        
        assert list(importance_named.index) == feature_names
    
    def test_predict_new_data(self, sample_classification_data):
        """Test prediction on new data."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train a model
        model, _ = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Make predictions
        predictions = trainer.predict(model, X_test)
        probabilities = trainer.predict_proba(model, X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Check probabilities
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_hyperparameter_tuning(self, sample_classification_data):
        """Test hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5]
        }
        
        # Train with hyperparameter tuning
        model, metrics = trainer.train_random_forest(
            X_train, y_train, X_test, y_test,
            param_grid=param_grid, cv_folds=3
        )
        
        # Check that model was trained with best parameters
        assert model is not None
        assert model.n_estimators in param_grid['n_estimators']
        assert model.max_depth in param_grid['max_depth']
    
    def test_class_imbalance_handling(self):
        """Test handling of imbalanced datasets."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        trainer = ModelTrainer()
        
        # Train model with class balancing
        model, metrics = trainer.train_random_forest(
            X_train, y_train, X_test, y_test,
            class_weight='balanced'
        )
        
        # Check that model handles imbalanced data
        assert model is not None
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Both precision and recall should be reasonable for imbalanced data
        assert metrics['precision'] > 0.1
        assert metrics['recall'] > 0.1
    
    def test_error_handling_invalid_data(self):
        """Test error handling for invalid data."""
        trainer = ModelTrainer()
        
        # Test with empty arrays
        with pytest.raises((ValueError, IndexError)):
            trainer.train_logistic_regression(np.array([]), np.array([]), 
                                            np.array([]), np.array([]))
        
        # Test with mismatched dimensions
        X_train = np.random.rand(10, 5)
        y_train = np.random.randint(0, 2, 8)  # Wrong size
        X_test = np.random.rand(5, 5)
        y_test = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError):
            trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    
    def test_model_persistence_in_trainer(self, sample_classification_data):
        """Test that models are stored in trainer."""
        X_train, X_test, y_train, y_test = sample_classification_data
        trainer = ModelTrainer()
        
        # Train models
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Check that models are stored
        assert 'Logistic Regression' in trainer.models
        assert 'Random Forest' in trainer.models
        assert 'Logistic Regression' in trainer.results
        assert 'Random Forest' in trainer.results
        
        # Check that we can access stored models
        lr_model = trainer.models['Logistic Regression']
        rf_model = trainer.models['Random Forest']
        
        assert lr_model is not None
        assert rf_model is not None
        assert hasattr(lr_model, 'predict')
        assert hasattr(rf_model, 'predict')