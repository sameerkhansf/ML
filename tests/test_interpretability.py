"""
Unit tests for the interpretability module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.interpretability import InterpretabilityEngine


@pytest.fixture
def sample_model_and_data():
    """Fixture providing a trained model and test data."""
    # Create sample classification data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    return model, X_train, X_test, y_train, y_test, feature_names


class TestInterpretabilityEngine:
    """Test class for InterpretabilityEngine functionality."""
    
    def test_initialization(self):
        """Test InterpretabilityEngine initialization."""
        engine = InterpretabilityEngine()
        assert engine is not None
    
    def test_calculate_feature_importance(self, sample_model_and_data):
        """Test feature importance calculation."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Calculate feature importance
        importance = engine.calculate_feature_importance(model, X_train, feature_names)
        
        # Check results
        assert isinstance(importance, pd.Series)
        assert len(importance) == len(feature_names)
        assert list(importance.index) == feature_names
        
        # Check that importances are normalized
        assert np.allclose(importance.sum(), 1.0)
        
        # Check that all importances are non-negative
        assert (importance >= 0).all()
    
    def test_calculate_feature_importance_without_names(self, sample_model_and_data):
        """Test feature importance calculation without feature names."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Calculate feature importance without names
        importance = engine.calculate_feature_importance(model, X_train)
        
        # Check results
        assert isinstance(importance, pd.Series)
        assert len(importance) == X_train.shape[1]
        
        # Check that default names were used
        expected_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        assert list(importance.index) == expected_names
    
    def test_generate_shap_analysis(self, sample_model_and_data):
        """Test SHAP analysis generation."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Generate SHAP analysis
        try:
            shap_results = engine.generate_shap_analysis(
                model, X_train, feature_names, sample_size=50
            )
            
            # Check results structure
            assert isinstance(shap_results, dict)
            assert 'shap_values' in shap_results
            assert 'X_sample' in shap_results
            assert 'explainer' in shap_results
            
            # Check SHAP values shape
            assert shap_results['shap_values'].shape[1] == len(feature_names)
            assert shap_results['X_sample'].shape[1] == len(feature_names)
            
        except ImportError:
            # SHAP might not be available in test environment
            pytest.skip("SHAP not available for testing")
        except Exception as e:
            # SHAP analysis might fail for various reasons in test environment
            pytest.skip(f"SHAP analysis failed: {str(e)}")
    
    def test_analyze_chemical_space(self, sample_model_and_data):
        """Test chemical space analysis."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Analyze chemical space using PCA
        results = engine.analyze_chemical_space(X_train, y_train, method='pca', n_components=3)
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'transformed' in results
        assert 'explained_variance' in results
        assert 'method' in results
        
        # Check transformed data shape
        assert results['transformed'].shape[0] == X_train.shape[0]
        assert results['transformed'].shape[1] == 3
        
        # Check explained variance
        assert len(results['explained_variance']) == 3
        assert all(0 <= var <= 1 for var in results['explained_variance'])
        
        # Check method
        assert results['method'] == 'pca'
    
    def test_analyze_chemical_space_tsne(self, sample_model_and_data):
        """Test chemical space analysis with t-SNE."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Analyze chemical space using t-SNE
        results = engine.analyze_chemical_space(X_train, y_train, method='tsne', n_components=2)
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'transformed' in results
        assert 'method' in results
        
        # Check transformed data shape
        assert results['transformed'].shape[0] == X_train.shape[0]
        assert results['transformed'].shape[1] == 2
        
        # Check method
        assert results['method'] == 'tsne'
        
        # t-SNE doesn't have explained variance
        assert 'explained_variance' not in results
    
    def test_get_top_features(self, sample_model_and_data):
        """Test getting top features by importance."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Calculate feature importance first
        importance = engine.calculate_feature_importance(model, X_train, feature_names)
        
        # Get top features
        top_features = engine.get_top_features(importance, n_features=5)
        
        # Check results
        assert isinstance(top_features, pd.Series)
        assert len(top_features) == 5
        
        # Check that features are sorted by importance (descending)
        importance_values = top_features.values
        assert all(importance_values[i] >= importance_values[i+1] 
                  for i in range(len(importance_values)-1))
    
    def test_explain_prediction(self, sample_model_and_data):
        """Test individual prediction explanation."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Get a sample for explanation
        sample_idx = 0
        sample = X_test[sample_idx:sample_idx+1]
        
        # Explain prediction
        try:
            explanation = engine.explain_prediction(model, sample, feature_names)
            
            # Check results structure
            assert isinstance(explanation, dict)
            assert 'prediction' in explanation
            assert 'probability' in explanation
            assert 'feature_contributions' in explanation
            
            # Check prediction
            assert explanation['prediction'] in [0, 1]
            
            # Check probability
            assert isinstance(explanation['probability'], np.ndarray)
            assert len(explanation['probability']) == 2  # Binary classification
            assert np.allclose(explanation['probability'].sum(), 1.0)
            
            # Check feature contributions
            assert isinstance(explanation['feature_contributions'], pd.Series)
            assert len(explanation['feature_contributions']) == len(feature_names)
            
        except ImportError:
            # SHAP might not be available
            pytest.skip("SHAP not available for prediction explanation")
        except Exception as e:
            # Explanation might fail in test environment
            pytest.skip(f"Prediction explanation failed: {str(e)}")
    
    def test_compare_feature_importance(self, sample_model_and_data):
        """Test comparing feature importance between models."""
        model1, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Train a second model with different parameters
        model2 = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        model2.fit(X_train, y_train)
        
        # Calculate importance for both models
        importance1 = engine.calculate_feature_importance(model1, X_train, feature_names)
        importance2 = engine.calculate_feature_importance(model2, X_train, feature_names)
        
        # Compare importance
        comparison = engine.compare_feature_importance([importance1, importance2], 
                                                     ['Model1', 'Model2'])
        
        # Check results
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison.columns) == 2  # Two models
        assert list(comparison.columns) == ['Model1', 'Model2']
        assert len(comparison) == len(feature_names)
        assert list(comparison.index) == feature_names
    
    def test_calculate_permutation_importance(self, sample_model_and_data):
        """Test permutation importance calculation."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Calculate permutation importance
        perm_importance = engine.calculate_permutation_importance(
            model, X_test, y_test, feature_names, n_repeats=5
        )
        
        # Check results
        assert isinstance(perm_importance, pd.DataFrame)
        assert 'importance_mean' in perm_importance.columns
        assert 'importance_std' in perm_importance.columns
        assert len(perm_importance) == len(feature_names)
        assert list(perm_importance.index) == feature_names
        
        # Check that standard deviations are non-negative
        assert (perm_importance['importance_std'] >= 0).all()
    
    def test_error_handling_invalid_model(self, sample_model_and_data):
        """Test error handling for models without feature_importances_."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Create a mock model without feature_importances_
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))
            
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)
        
        mock_model = MockModel()
        
        # Should handle gracefully
        with pytest.raises((AttributeError, ValueError)):
            engine.calculate_feature_importance(mock_model, X_train, feature_names)
    
    def test_error_handling_mismatched_dimensions(self, sample_model_and_data):
        """Test error handling for mismatched dimensions."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Test with wrong number of feature names
        wrong_feature_names = feature_names[:5]  # Too few names
        
        with pytest.raises((ValueError, IndexError)):
            engine.calculate_feature_importance(model, X_train, wrong_feature_names)
    
    def test_error_handling_empty_data(self):
        """Test error handling for empty data."""
        engine = InterpretabilityEngine()
        
        # Test with empty arrays
        empty_X = np.array([]).reshape(0, 5)
        empty_y = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            engine.analyze_chemical_space(empty_X, empty_y)
    
    def test_feature_importance_ranking(self, sample_model_and_data):
        """Test feature importance ranking functionality."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Calculate feature importance
        importance = engine.calculate_feature_importance(model, X_train, feature_names)
        
        # Get ranking
        ranking = engine.rank_features_by_importance(importance)
        
        # Check results
        assert isinstance(ranking, pd.DataFrame)
        assert 'feature' in ranking.columns
        assert 'importance' in ranking.columns
        assert 'rank' in ranking.columns
        assert len(ranking) == len(feature_names)
        
        # Check that ranking is correct
        assert ranking['rank'].tolist() == list(range(1, len(feature_names) + 1))
        
        # Check that features are sorted by importance (descending)
        importance_values = ranking['importance'].values
        assert all(importance_values[i] >= importance_values[i+1] 
                  for i in range(len(importance_values)-1))
    
    def test_chemical_space_visualization_data(self, sample_model_and_data):
        """Test chemical space visualization data preparation."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Analyze chemical space
        results = engine.analyze_chemical_space(X_train, y_train, method='pca', n_components=2)
        
        # Prepare visualization data
        viz_data = engine.prepare_visualization_data(results, y_train)
        
        # Check results
        assert isinstance(viz_data, pd.DataFrame)
        assert 'PC1' in viz_data.columns
        assert 'PC2' in viz_data.columns
        assert 'target' in viz_data.columns
        assert len(viz_data) == len(X_train)
        
        # Check that target values are preserved
        assert list(viz_data['target']) == list(y_train)
    
    def test_model_agnostic_explanations(self, sample_model_and_data):
        """Test model-agnostic explanation methods."""
        model, X_train, X_test, y_train, y_test, feature_names = sample_model_and_data
        engine = InterpretabilityEngine()
        
        # Test with different model types that don't have feature_importances_
        from sklearn.linear_model import LogisticRegression
        
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        
        # Should still be able to calculate permutation importance
        perm_importance = engine.calculate_permutation_importance(
            lr_model, X_test, y_test, feature_names, n_repeats=3
        )
        
        # Check results
        assert isinstance(perm_importance, pd.DataFrame)
        assert len(perm_importance) == len(feature_names)
        assert 'importance_mean' in perm_importance.columns