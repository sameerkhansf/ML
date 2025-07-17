"""
Unit tests for data quality checks and preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from src.data_handler import DataHandler

def test_check_missing_values(sample_dataframe):
    """Test missing value detection."""
    # Create a copy with some missing values
    df = sample_dataframe.copy()
    df.loc[0, 'smiles'] = None
    df.loc[1, 'name'] = None
    
    # Initialize data handler
    handler = DataHandler()
    
    # Check missing values
    missing = handler.check_missing_values(df)
    
    # Verify results
    assert len(missing) == 2
    assert missing['smiles'] == 1
    assert missing['name'] == 1

def test_handle_missing_values(sample_dataframe):
    """Test missing value handling strategies."""
    # Create a copy with some missing values
    df = sample_dataframe.copy()
    df.loc[0, 'smiles'] = None
    df.loc[1, 'name'] = None
    
    # Add mol column
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
    
    # Initialize data handler
    handler = DataHandler()
    
    # Test 'drop' strategy
    result_drop = handler.handle_missing_values(df, strategy='drop')
    assert len(result_drop) == len(df) - 2
    
    # Test 'drop_mol' strategy
    result_drop_mol = handler.handle_missing_values(df, strategy='drop_mol')
    assert len(result_drop_mol) == len(df) - 1
    
    # Test 'keep' strategy
    result_keep = handler.handle_missing_values(df, strategy='keep')
    assert len(result_keep) == len(df)

def test_detect_duplicates():
    """Test duplicate detection."""
    # Create a dataframe with duplicates
    data = {
        'num': [1, 2, 3, 4, 5],
        'name': ['Compound1', 'Compound2', 'Compound1', 'Compound4', 'Compound5'],
        'smiles': ['C', 'CC', 'C', 'CCC', 'CCCC'],
        'p_np': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Initialize data handler
    handler = DataHandler()
    
    # Detect duplicates
    duplicates = handler.detect_duplicates(df, subset=['name', 'smiles'])
    
    # Verify results
    assert len(duplicates) == 2
    assert 1 in duplicates['num'].values
    assert 3 in duplicates['num'].values

def test_remove_duplicates():
    """Test duplicate removal."""
    # Create a dataframe with duplicates
    data = {
        'num': [1, 2, 3, 4, 5],
        'name': ['Compound1', 'Compound2', 'Compound1', 'Compound4', 'Compound5'],
        'smiles': ['C', 'CC', 'C', 'CCC', 'CCCC'],
        'p_np': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Initialize data handler
    handler = DataHandler()
    
    # Remove duplicates keeping first occurrence
    result_first = handler.remove_duplicates(df, subset=['name', 'smiles'], keep='first')
    assert len(result_first) == 4
    assert 1 in result_first['num'].values
    assert 3 not in result_first['num'].values
    
    # Remove duplicates keeping last occurrence
    result_last = handler.remove_duplicates(df, subset=['name', 'smiles'], keep='last')
    assert len(result_last) == 4
    assert 3 in result_last['num'].values
    assert 1 not in result_last['num'].values
    
    # Remove all duplicates
    result_false = handler.remove_duplicates(df, subset=['name', 'smiles'], keep=False)
    assert len(result_false) == 3
    assert 1 not in result_false['num'].values
    assert 3 not in result_false['num'].values

def test_generate_data_summary(sample_dataframe):
    """Test data summary generation."""
    # Initialize data handler
    handler = DataHandler()
    handler.data = sample_dataframe
    
    # Add mol column
    handler.data['mol'] = handler.data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    
    # Generate summary
    summary = handler.generate_data_summary()
    
    # Verify results
    assert 'total_compounds' in summary
    assert 'valid_molecules' in summary
    assert 'permeable_compounds' in summary
    assert 'non_permeable_compounds' in summary
    assert summary['total_compounds'] == len(sample_dataframe)
    assert summary['valid_molecules'] <= len(sample_dataframe)

def test_plot_class_distribution(sample_dataframe):
    """Test class distribution plotting."""
    # Initialize data handler
    handler = DataHandler()
    handler.data = sample_dataframe
    
    # Plot class distribution
    fig = handler.plot_class_distribution()
    
    # Verify figure was created
    assert isinstance(fig, plt.Figure)
    
    # Close figure to avoid memory leaks
    plt.close(fig)

def test_analyze_molecular_properties(sample_dataframe):
    """Test molecular property analysis."""
    # Initialize data handler
    handler = DataHandler()
    handler.data = sample_dataframe
    
    # Add mol column with valid molecules
    valid_smiles = ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    handler.data = handler.data.iloc[:3].copy()
    handler.data['smiles'] = valid_smiles
    handler.data['mol'] = handler.data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    
    # Analyze molecular properties
    stats = handler.analyze_molecular_properties()
    
    # Verify results
    assert isinstance(stats, pd.DataFrame)
    assert 'molecular_weight' in stats.columns.levels[0]
    assert 'heavy_atom_count' in stats.columns.levels[0]