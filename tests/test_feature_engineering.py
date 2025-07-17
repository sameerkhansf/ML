"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import FeatureEngineering


@pytest.fixture
def sample_feature_data():
    """Fixture providing sample feature data for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'feature3': np.random.normal(-2, 0.5, 100),
        'feature4': np.random.normal(0, 1, 100) * 0.1,  # Low variance
        'feature5': np.random.normal(0, 1, 100),
    }
    # Make feature5 highly correlated with feature1
    data['feature5'] = data['feature1'] * 0.95 + np.random.normal(0, 0.1, 100)
    
    # Add some missing values
    data['feature2'][10:15] = np.nan
    data['feature3'][20:25] = np.nan
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_target():
    """Fixture providing sample target data."""
    np.random.seed(42)
    return np.random.choice([0, 1], size=100)


class TestFeatureEngineering:
    """Test class for FeatureEngineering functionality."""
    
    def test_initialization(self):
        """Test FeatureEngineering initialization."""
        fe = FeatureEngineering()
        assert fe.scaler is None
        assert fe.selected_features is None
        assert fe.feature_selector is None
    
    def test_handle_missing_values_median(self, sample_feature_data):
        """Test missing value handling with median strategy."""
        fe = FeatureEngineering()
        
        # Check that we have missing values
        assert sample_feature_data.isnull().sum().sum() > 0
        
        # Handle missing values
        result = fe.handle_missing_values(sample_feature_data, strategy='median')
        
        # Check that no missing values remain
        assert result.isnull().sum().sum() == 0
        
        # Check that imputer was fitted
        assert fe.imputer is not None
        
        # Check that values were imputed correctly
        original_median_f2 = sample_feature_data['feature2'].median()
        assert np.allclose(result.loc[10:14, 'feature2'], original_median_f2)
    
    def test_handle_missing_values_mean(self, sample_feature_data):
        """Test missing value handling with mean strategy."""
        fe = FeatureEngineering()
        
        result = fe.handle_missing_values(sample_feature_data, strategy='mean')
        
        # Check that no missing values remain
        assert result.isnull().sum().sum() == 0
        
        # Check that values were imputed with mean
        original_mean_f2 = sample_feature_data['feature2'].mean()
        assert np.allclose(result.loc[10:14, 'feature2'], original_mean_f2)
    
    def test_handle_missing_values_drop(self, sample_feature_data):
        """Test missing value handling with drop strategy."""
        fe = FeatureEngineering()
        
        original_length = len(sample_feature_data)
        result = fe.handle_missing_values(sample_feature_data, strategy='drop')
        
        # Check that rows with missing values were dropped
        assert len(result) < original_length
        assert result.isnull().sum().sum() == 0
    
    def test_scale_features_standard(self, sample_feature_data):
        """Test feature scaling with StandardScaler."""
        fe = FeatureEngineering()
        
        # Remove missing values first
        clean_data = sample_feature_data.dropna()
        
        # Scale features
        result = fe.scale_features(clean_data, method='standard')
        
        # Check that scaler was fitted
        assert fe.scaler is not None
        assert isinstance(fe.scaler, StandardScaler)
        
        # Check that features are standardized (mean ~0, std ~1)
        assert np.allclose(result.mean(), 0, atol=1e-10)
        assert np.allclose(result.std(), 1, atol=1e-10)
    
    def test_scale_features_minmax(self, sample_feature_data):
        """Test feature scaling with MinMaxScaler."""
        fe = FeatureEngineering()
        
        # Remove missing values first
        clean_data = sample_feature_data.dropna()
        
        # Scale features
        result = fe.scale_features(clean_data, method='minmax')
        
        # Check that features are scaled to [0, 1]
        assert result.min().min() >= 0
        assert result.max().max() <= 1
    
    def test_select_features_variance(self, sample_feature_data):
        """Test feature selection based on variance threshold."""
        fe = FeatureEngineering()
        
        # Remove missing values first
        clean_data = sample_feature_data.dropna()
        
        # Select features
        result = fe.select_features(clean_data, method='variance', threshold=0.01)
        
        # Check that low variance feature was removed
        assert 'feature4' not in result.columns
        assert len(result.columns) < len(clean_data.columns)
        
        # Check that selected features were stored
        assert fe.selected_features is not None
        assert len(fe.selected_features) == len(result.columns)
    
    def test_select_features_correlation(self, sample_feature_data):
        """Test feature selection based on correlation threshold."""
        fe = FeatureEngineering()
        
        # Remove missing values first
        clean_data = sample_feature_data.dropna()
        
        # Select features
        result = fe.select_features(clean_data, method='correlation', threshold=0.9)
        
        # Check that highly correlated features were handled
        # Either feature1 or feature5 should be removed (they're highly correlated)
        assert len(result.columns) < len(clean_data.columns)
        
        # Check correlation matrix of result
        corr_matrix = result.corr()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        max_corr = upper_triangle.abs().max().max()
        assert max_corr < 0.9
    
    def test_select_features_univariate(self, sample_feature_data, sample_target):
        """Test univariate feature selection."""
        fe = FeatureEngineering()
        
        # Remove missing values first
        clean_data = sample_feature_data.dropna()
        clean_target = sample_target[:len(clean_data)]
        
        # Select features
        result = fe.select_features(clean_data, method='univariate', 
                                  k=3, y=clean_target)
        
        # Check that only k features were selected
        assert len(result.columns) == 3
        
        # Check that feature selector was fitted
        assert fe.feature_selector is not None
    
    def test_create_feature_matrix_complete_pipeline(self, sample_feature_data, sample_target):
        """Test complete feature engineering pipeline."""
        fe = FeatureEngineering()
        
        # Create feature matrix with all steps
        result = fe.create_feature_matrix(
            sample_feature_data,
            handle_missing=True,
            scale=True,
            select=True,
            y=sample_target
        )
        
        # Check that all steps were applied
        assert result.isnull().sum().sum() == 0  # No missing values
        assert np.allclose(result.mean(), 0, atol=1e-10)  # Standardized
        assert len(result.columns) <= len(sample_feature_data.columns)  # Feature selection
        
        # Check that all components were fitted
        assert fe.imputer is not None
        assert fe.scaler is not None
        assert fe.selected_features is not None
    
    def test_transform_new_data(self, sample_feature_data, sample_target):
        """Test transforming new data with fitted pipeline."""
        fe = FeatureEngineering()
        
        # Fit on training data
        train_result = fe.create_feature_matrix(
            sample_feature_data,
            handle_missing=True,
            scale=True,
            select=True,
            y=sample_target
        )
        
        # Create new test data
        np.random.seed(123)
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(5, 2, 20),
            'feature3': np.random.normal(-2, 0.5, 20),
            'feature4': np.random.normal(0, 1, 20) * 0.1,
            'feature5': np.random.normal(0, 1, 20),
        })
        
        # Add some missing values
        test_data.loc[5, 'feature2'] = np.nan
        
        # Transform test data
        test_result = fe.create_feature_matrix(
            test_data,
            handle_missing=False,  # Use fitted transformers
            scale=False,
            select=False
        )
        
        # Check that transformations were applied consistently
        assert test_result.isnull().sum().sum() == 0
        assert test_result.shape[1] == train_result.shape[1]
        assert list(test_result.columns) == list(train_result.columns)
    
    def test_get_feature_names(self, sample_feature_data):
        """Test getting feature names after selection."""
        fe = FeatureEngineering()
        
        # Remove missing values and select features
        clean_data = sample_feature_data.dropna()
        fe.select_features(clean_data, method='variance', threshold=0.01)
        
        # Get feature names
        feature_names = fe.get_feature_names()
        
        # Check that feature names match selected features
        assert feature_names == fe.selected_features
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_error_handling_invalid_strategy(self, sample_feature_data):
        """Test error handling for invalid strategies."""
        fe = FeatureEngineering()
        
        # Test invalid missing value strategy
        with pytest.raises(ValueError):
            fe.handle_missing_values(sample_feature_data, strategy='invalid')
        
        # Test invalid scaling method
        with pytest.raises(ValueError):
            fe.scale_features(sample_feature_data, method='invalid')
        
        # Test invalid selection method
        with pytest.raises(ValueError):
            fe.select_features(sample_feature_data, method='invalid')
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        fe = FeatureEngineering()
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        result = fe.handle_missing_values(empty_df)
        assert len(result) == 0
        
        result = fe.scale_features(empty_df)
        assert len(result) == 0
    
    def test_single_feature_handling(self):
        """Test handling of single feature DataFrames."""
        fe = FeatureEngineering()
        single_feature_df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        
        # Should handle single feature DataFrame
        result = fe.scale_features(single_feature_df)
        assert len(result.columns) == 1
        assert np.allclose(result.mean(), 0, atol=1e-10)
        assert np.allclose(result.std(), 1, atol=1e-10)