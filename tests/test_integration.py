"""
Integration tests for the complete BBB permeability prediction pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from rdkit import Chem

# Import all main modules
from src.data_handler import DataHandler
from src.descriptors import MolecularDescriptors
from src.feature_engineering import FeatureEngineering
from src.models import ModelTrainer
from src.interpretability import InterpretabilityEngine
from src.visualization import VisualizationSuite
from src.main_analysis import MainAnalysis


@pytest.fixture
def sample_bbbp_data():
    """Fixture providing sample BBBP-like data for integration testing."""
    # Create realistic SMILES strings with known BBB permeability patterns
    data = {
        'num': list(range(1, 21)),
        'name': [f'Compound_{i}' for i in range(1, 21)],
        'p_np': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'smiles': [
            'CCO',  # Ethanol - small, permeable
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin - larger, less permeable
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine - moderate
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CCC',  # Propane - small, permeable
            'CCCCCCCCCCCCCCCCCC',  # Long chain - large, less permeable
            'c1ccccc1',  # Benzene - aromatic, moderate
            'CC(C)(C)C1=CC=C(C=C1)O',  # BHT - bulky
            'CCN(CC)CC',  # Triethylamine - basic
            'CC(=O)O',  # Acetic acid - small acid
            'CCCCO',  # Butanol - alcohol
            'CC1=CC=CC=C1',  # Toluene - aromatic
            'CCN',  # Ethylamine - small base
            'CCCCC',  # Pentane - alkane
            'CC(C)O',  # Isopropanol
            'CCCCCCCC',  # Octane - longer alkane
            'c1ccc2ccccc2c1',  # Naphthalene - polycyclic aromatic
            'CC(C)C',  # Isobutane
            'CCCCCCCCCCCC',  # Dodecane - very long chain
            'CO'  # Methanol - very small alcohol
        ]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_bbbp_data):
    """Fixture providing a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_bbbp_data.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline_execution(self, temp_data_file):
        """Test complete pipeline from data loading to model training."""
        # 1. Data Loading and Validation
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        
        assert len(data) == 20
        assert 'smiles' in data.columns
        assert 'p_np' in data.columns
        
        # Validate SMILES
        data_with_mol = data_handler.validate_smiles(data)
        assert 'mol' in data_with_mol.columns
        
        # Check that most SMILES are valid
        valid_count = data_with_mol['mol'].notna().sum()
        assert valid_count >= 18  # Allow for a few invalid SMILES
        
        # 2. Descriptor Calculation
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        # Check that descriptors were calculated
        assert len(data_with_descriptors.columns) > len(data.columns)
        assert 'MolLogP' in data_with_descriptors.columns
        assert 'MolWt' in data_with_descriptors.columns
        
        # 3. Feature Engineering
        feature_eng = FeatureEngineering()
        
        # Get descriptor columns
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        # Apply feature engineering
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        assert X_processed.shape[0] == len(y)
        assert X_processed.shape[1] <= len(descriptor_cols)
        
        # 4. Model Training
        model_trainer = ModelTrainer()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train a simple model
        model, metrics = model_trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Check that model was trained successfully
        assert model is not None
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        # 5. Model Evaluation and Interpretation
        interp_engine = InterpretabilityEngine()
        
        # Calculate feature importance
        feature_names = feature_eng.get_feature_names()
        importance = interp_engine.calculate_feature_importance(
            model, X_train, feature_names
        )
        
        assert isinstance(importance, pd.Series)
        assert len(importance) == X_train.shape[1]
        
        # 6. Visualization
        viz_suite = VisualizationSuite()
        
        # Create model predictions for visualization
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        predictions = {
            'Random Forest': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        }
        
        # Test visualization
        fig = viz_suite.plot_model_performance(predictions)
        assert fig is not None
        
        # Close figure
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_main_analysis_workflow(self, temp_data_file):
        """Test the main analysis workflow orchestrator."""
        # Initialize main analysis
        main_analysis = MainAnalysis()
        
        # Run complete analysis
        results = main_analysis.run_complete_analysis(
            data_path=temp_data_file,
            output_dir=None  # Use temporary directory
        )
        
        # Check that results were generated
        assert isinstance(results, dict)
        assert 'data_summary' in results
        assert 'model_results' in results
        assert 'best_model' in results
        
        # Check data summary
        data_summary = results['data_summary']
        assert 'total_compounds' in data_summary
        assert data_summary['total_compounds'] == 20
        
        # Check model results
        model_results = results['model_results']
        assert len(model_results) > 0
        
        # Check best model
        best_model = results['best_model']
        assert best_model is not None
    
    def test_error_handling_invalid_smiles(self):
        """Test error handling with invalid SMILES strings."""
        # Create data with invalid SMILES
        invalid_data = pd.DataFrame({
            'num': [1, 2, 3],
            'name': ['Valid', 'Invalid', 'Another_Invalid'],
            'p_np': [1, 0, 1],
            'smiles': ['CCO', 'INVALID_SMILES', '']
        })
        
        # Test data handler
        data_handler = DataHandler()
        result = data_handler.validate_smiles(invalid_data)
        
        # Should handle invalid SMILES gracefully
        assert len(result) == 3
        assert result.loc[0, 'mol'] is not None  # Valid SMILES
        assert result.loc[1, 'mol'] is None      # Invalid SMILES
        assert result.loc[2, 'mol'] is None      # Empty SMILES
        
        # Test descriptor calculation
        descriptor_calc = MolecularDescriptors()
        result_with_descriptors = descriptor_calc.calculate_from_dataframe(
            result, mol_column='mol'
        )
        
        # Should handle None molecules gracefully
        assert len(result_with_descriptors) == 3
        # Invalid molecules should have NaN descriptors
        assert pd.isna(result_with_descriptors.loc[1, 'MolLogP'])
        assert pd.isna(result_with_descriptors.loc[2, 'MolLogP'])
    
    def test_small_dataset_handling(self):
        """Test handling of very small datasets."""
        # Create minimal dataset
        small_data = pd.DataFrame({
            'num': [1, 2, 3, 4],
            'name': ['A', 'B', 'C', 'D'],
            'p_np': [1, 0, 1, 0],
            'smiles': ['CCO', 'CC', 'CCC', 'CCCC']
        })
        
        # Test complete pipeline with small dataset
        data_handler = DataHandler()
        data_with_mol = data_handler.validate_smiles(small_data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        # Feature engineering
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        # Should handle small dataset
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=False, y=y  # No selection for small data
        )
        
        assert X_processed.shape[0] == 4
        assert X_processed.shape[1] > 0
    
    def test_model_persistence_and_loading(self, temp_data_file, tmp_path):
        """Test model saving and loading functionality."""
        # Train a model
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        data_with_mol = data_handler.validate_smiles(data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        # Split and train
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        model_trainer.save_model(model, model_path)
        
        # Load model
        loaded_model = model_trainer.load_model(model_path)
        
        # Test that loaded model works
        predictions_original = model.predict(X_test)
        predictions_loaded = loaded_model.predict(X_test)
        
        # Predictions should be identical
        assert np.array_equal(predictions_original, predictions_loaded)
    
    def test_cross_validation_integration(self, temp_data_file):
        """Test cross-validation integration in the pipeline."""
        # Prepare data
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        data_with_mol = data_handler.validate_smiles(data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        # Train multiple models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_trainer = ModelTrainer()
        
        # Train multiple models
        model_trainer.train_random_forest(X_train, y_train, X_test, y_test)
        model_trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Cross-validate models
        cv_results = model_trainer.cross_validate_models(X_train, y_train, cv_folds=3)
        
        # Check cross-validation results
        assert len(cv_results) == 2  # Two models
        for model_name, scores in cv_results.items():
            assert 'mean_score' in scores
            assert 'std_score' in scores
            assert 0 <= scores['mean_score'] <= 1
    
    def test_feature_importance_pipeline(self, temp_data_file):
        """Test feature importance analysis in the complete pipeline."""
        # Prepare data and train model
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        data_with_mol = data_handler.validate_smiles(data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Feature importance analysis
        interp_engine = InterpretabilityEngine()
        feature_names = feature_eng.get_feature_names()
        
        # Test different importance methods
        tree_importance = interp_engine.calculate_feature_importance(
            model, X_train, feature_names
        )
        
        perm_importance = interp_engine.calculate_permutation_importance(
            model, X_test, y_test, feature_names, n_repeats=3
        )
        
        # Check results
        assert isinstance(tree_importance, pd.Series)
        assert isinstance(perm_importance, pd.DataFrame)
        assert len(tree_importance) == len(feature_names)
        assert len(perm_importance) == len(feature_names)
    
    def test_visualization_pipeline(self, temp_data_file):
        """Test visualization integration in the pipeline."""
        # Prepare data and train models
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        data_with_mol = data_handler.validate_smiles(data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_trainer = ModelTrainer()
        
        # Train multiple models
        rf_model, rf_metrics = model_trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        lr_model, lr_metrics = model_trainer.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        
        # Create predictions for visualization
        predictions = {}
        for name, model in [('Random Forest', rf_model), ('Logistic Regression', lr_model)]:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            predictions[name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        
        # Test visualizations
        viz_suite = VisualizationSuite()
        
        # Model performance visualization
        perf_fig = viz_suite.plot_model_performance(predictions)
        assert perf_fig is not None
        
        # Feature importance visualization
        feature_names = feature_eng.get_feature_names()
        interp_engine = InterpretabilityEngine()
        importance = interp_engine.calculate_feature_importance(
            rf_model, X_train, feature_names
        )
        
        imp_fig = viz_suite.plot_feature_importance(importance)
        assert imp_fig is not None
        
        # Chemical space visualization
        space_results = interp_engine.analyze_chemical_space(
            X_processed, y, method='pca', n_components=2
        )
        
        space_fig = viz_suite.plot_chemical_space(
            space_results['transformed'], y, space_results['explained_variance']
        )
        assert space_fig is not None
        
        # Close figures
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def test_prediction_on_new_molecules(self, temp_data_file):
        """Test making predictions on new molecules."""
        # Train pipeline
        data_handler = DataHandler()
        data = data_handler.load_bbbp_data(temp_data_file)
        data_with_mol = data_handler.validate_smiles(data)
        
        descriptor_calc = MolecularDescriptors()
        data_with_descriptors = descriptor_calc.calculate_from_dataframe(
            data_with_mol, mol_column='mol'
        )
        
        feature_eng = FeatureEngineering()
        exclude_cols = ['num', 'name', 'smiles', 'p_np', 'mol']
        descriptor_cols = [col for col in data_with_descriptors.columns 
                          if col not in exclude_cols]
        
        X = data_with_descriptors[descriptor_cols].fillna(0)
        y = data_with_descriptors['p_np']
        
        X_processed = feature_eng.create_feature_matrix(
            X, handle_missing=True, scale=True, select=True, y=y
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Create new molecules for prediction
        new_molecules = pd.DataFrame({
            'smiles': ['CCCCCO', 'c1ccc(cc1)O', 'CCN(CC)CC'],
            'name': ['Pentanol', 'Phenol', 'Triethylamine']
        })
        
        # Calculate descriptors for new molecules
        new_with_descriptors = descriptor_calc.calculate_from_dataframe(
            new_molecules, smiles_column='smiles'
        )
        
        # Apply same feature engineering (without fitting)
        new_X = new_with_descriptors[descriptor_cols].fillna(0)
        new_X_processed = feature_eng.create_feature_matrix(
            new_X, handle_missing=False, scale=False, select=False
        )
        
        # Make predictions
        predictions = model.predict(new_X_processed)
        probabilities = model.predict_proba(new_X_processed)
        
        # Check predictions
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (3, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)