"""
Unit tests for the data handler module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from src.data_handler import DataHandler


class TestDataHandler:
    """Test class for DataHandler functionality."""
    
    def test_initialization(self):
        """Test DataHandler initialization."""
        handler = DataHandler()
        assert handler.data is None
        assert handler.clean_data is None
    
    def test_load_bbbp_data(self, tmp_path):
        """Test loading BBBP data from CSV."""
        # Create a temporary CSV file
        test_data = pd.DataFrame({
            'num': [1, 2, 3],
            'name': ['Compound1', 'Compound2', 'Compound3'],
            'p_np': [1, 0, 1],
            'smiles': ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
        })
        
        csv_path = tmp_path / "test_bbbp.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Test loading
        handler = DataHandler()
        loaded_data = handler.load_bbbp_data(csv_path)
        
        # Check that data was loaded correctly
        assert len(loaded_data) == 3
        assert all(col in loaded_data.columns for col in ['num', 'name', 'p_np', 'smiles'])
        assert handler.data is not None
    
    def test_validate_smiles_individual(self):
        """Test individual SMILES validation."""
        handler = DataHandler()
        
        # Test valid SMILES
        is_valid, mol = handler.validate_smiles('CCO')
        assert is_valid is True
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        
        # Test invalid SMILES
        is_valid, mol = handler.validate_smiles('INVALID_SMILES')
        assert is_valid is False
        assert mol is None
        
        # Test empty SMILES
        is_valid, mol = handler.validate_smiles('')
        assert is_valid is False
        assert mol is None
        
        # Test None SMILES
        is_valid, mol = handler.validate_smiles(None)
        assert is_valid is False
        assert mol is None
    
    def test_validate_smiles_dataframe(self, sample_dataframe):
        """Test SMILES validation on DataFrame."""
        handler = DataHandler()
        
        # Validate SMILES
        result = handler.validate_smiles(sample_dataframe)
        
        # Check that mol column was added
        assert 'mol' in result.columns
        
        # Check that valid SMILES were converted to molecules
        assert result.loc[0, 'mol'] is not None  # Ethanol
        assert result.loc[1, 'mol'] is not None  # Aspirin
        assert result.loc[2, 'mol'] is not None  # Caffeine
        
        # Check that invalid SMILES resulted in None
        assert result.loc[4, 'mol'] is None  # Invalid SMILES
    
    def test_check_missing_values(self, sample_dataframe):
        """Test missing value detection."""
        handler = DataHandler()
        
        # Add some missing values
        test_data = sample_dataframe.copy()
        test_data.loc[1, 'name'] = np.nan
        test_data.loc[2, 'p_np'] = np.nan
        
        # Check missing values
        missing_summary = handler.check_missing_values(test_data)
        
        # Check that missing values were detected
        assert isinstance(missing_summary, pd.Series)
        assert missing_summary['name'] == 1
        assert missing_summary['p_np'] == 1
        assert missing_summary['smiles'] == 0
    
    def test_detect_duplicates(self, sample_dataframe):
        """Test duplicate detection."""
        handler = DataHandler()
        
        # Add duplicate rows
        test_data = sample_dataframe.copy()
        duplicate_row = test_data.iloc[0].copy()
        test_data = pd.concat([test_data, duplicate_row.to_frame().T], ignore_index=True)
        
        # Detect duplicates
        duplicates = handler.detect_duplicates(test_data)
        
        # Check that duplicates were found
        assert len(duplicates) >= 2  # Original + duplicate
        assert duplicates.iloc[0]['smiles'] == duplicates.iloc[-1]['smiles']
    
    def test_handle_missing_values_drop_mol(self, sample_dataframe):
        """Test handling missing values by dropping rows with invalid molecules."""
        handler = DataHandler()
        
        # Add mol column with some None values
        test_data = sample_dataframe.copy()
        test_data['mol'] = [Chem.MolFromSmiles(s) for s in test_data['smiles']]
        
        # Handle missing values
        result = handler.handle_missing_values(test_data, strategy='drop_mol')
        
        # Check that rows with None molecules were dropped
        assert len(result) < len(test_data)
        assert result['mol'].notna().all()
    
    def test_handle_missing_values_drop_rows(self, sample_dataframe):
        """Test handling missing values by dropping rows."""
        handler = DataHandler()
        
        # Add some missing values
        test_data = sample_dataframe.copy()
        test_data.loc[1, 'name'] = np.nan
        
        # Handle missing values
        result = handler.handle_missing_values(test_data, strategy='drop_rows')
        
        # Check that rows with missing values were dropped
        assert len(result) < len(test_data)
        assert result.notna().all().all()
    
    def test_generate_data_summary(self, sample_dataframe):
        """Test data summary generation."""
        handler = DataHandler()
        
        # Add mol column
        test_data = handler.validate_smiles(sample_dataframe)
        
        # Generate summary
        summary = handler.generate_data_summary(test_data)
        
        # Check summary contents
        assert isinstance(summary, dict)
        assert 'total_compounds' in summary
        assert 'valid_molecules' in summary
        assert 'invalid_molecules' in summary
        assert 'class_distribution' in summary
        
        # Check values
        assert summary['total_compounds'] == len(test_data)
        assert summary['valid_molecules'] == test_data['mol'].notna().sum()
        assert summary['invalid_molecules'] == test_data['mol'].isna().sum()
    
    def test_plot_class_distribution(self, sample_dataframe):
        """Test class distribution plotting."""
        handler = DataHandler()
        
        # Plot class distribution
        fig = handler.plot_class_distribution(sample_dataframe)
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 2  # Bar plot and pie chart
    
    def test_visualize_molecules(self, sample_dataframe):
        """Test molecule visualization."""
        handler = DataHandler()
        
        # Add mol column
        test_data = handler.validate_smiles(sample_dataframe)
        
        # Visualize molecules
        perm_img, non_perm_img = handler.visualize_molecules(test_data, n_molecules=2, by_class=True)
        
        # Check that images were generated
        assert perm_img is not None or non_perm_img is not None
    
    def test_analyze_molecular_properties(self, sample_dataframe):
        """Test molecular properties analysis."""
        handler = DataHandler()
        
        # Add mol column
        test_data = handler.validate_smiles(sample_dataframe)
        
        # Analyze properties
        properties_stats = handler.analyze_molecular_properties(test_data)
        
        # Check that statistics were calculated
        assert isinstance(properties_stats, pd.DataFrame)
        assert 'Property' in properties_stats.columns
        assert len(properties_stats) > 0
    
    def test_get_train_test_split(self, sample_dataframe):
        """Test train-test splitting."""
        handler = DataHandler()
        
        # Add mol column
        test_data = handler.validate_smiles(sample_dataframe)
        handler.data = test_data
        
        # Create train-test split
        X_train, X_test, y_train, y_test = handler.get_train_test_split(test_size=0.4, random_state=42)
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(test_data)
        assert len(y_train) + len(y_test) == len(test_data)
        
        # Check that target variable is correct
        assert all(y in [0, 1] for y in y_train)
        assert all(y in [0, 1] for y in y_test)
        
        # Check stratification
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        original_ratio = test_data['p_np'].mean()
        
        # Ratios should be similar (within tolerance for small datasets)
        assert abs(train_ratio - original_ratio) < 0.2
        assert abs(test_ratio - original_ratio) < 0.2
    
    def test_export_results(self, sample_dataframe, tmp_path):
        """Test exporting results to CSV."""
        handler = DataHandler()
        
        # Export results
        file_path = tmp_path / "test_results.csv"
        result_path = handler.export_results(sample_dataframe, file_path)
        
        # Check that file was created
        assert result_path.exists()
        
        # Check that data was saved correctly
        loaded_data = pd.read_csv(result_path)
        assert len(loaded_data) == len(sample_dataframe)
        assert all(col in loaded_data.columns for col in sample_dataframe.columns)
    
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid file paths."""
        handler = DataHandler()
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            handler.load_bbbp_data("non_existent_file.csv")
    
    def test_error_handling_invalid_strategy(self, sample_dataframe):
        """Test error handling for invalid strategies."""
        handler = DataHandler()
        
        # Test invalid missing value strategy
        with pytest.raises(ValueError):
            handler.handle_missing_values(sample_dataframe, strategy='invalid_strategy')
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        handler = DataHandler()
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        result = handler.validate_smiles(empty_df)
        assert len(result) == 0
        
        summary = handler.generate_data_summary(empty_df)
        assert summary['total_compounds'] == 0
    
    def test_data_persistence(self, sample_dataframe):
        """Test that data is properly stored in handler."""
        handler = DataHandler()
        
        # Load data
        handler.data = sample_dataframe
        assert handler.data is not None
        assert len(handler.data) == len(sample_dataframe)
        
        # Process data
        clean_data = handler.validate_smiles(sample_dataframe)
        handler.clean_data = clean_data
        
        assert handler.clean_data is not None
        assert 'mol' in handler.clean_data.columns