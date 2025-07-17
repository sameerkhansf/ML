"""
Pytest configuration for the Blood-Brain Barrier Permeability Prediction Project.
"""

import pytest
import pandas as pd
from rdkit import Chem
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_smiles():
    """Fixture providing a list of sample SMILES strings."""
    return [
        "CCO",                  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=CC=C3",  # Terphenyl
        "INVALID_SMILES"        # Invalid SMILES
    ]

@pytest.fixture
def sample_molecules(sample_smiles):
    """Fixture providing a list of RDKit molecule objects."""
    return [Chem.MolFromSmiles(smiles) for smiles in sample_smiles]

@pytest.fixture
def sample_dataframe(sample_smiles):
    """Fixture providing a sample DataFrame with SMILES strings."""
    data = {
        'num': [1, 2, 3, 4, 5],
        'name': ['Ethanol', 'Aspirin', 'Caffeine', 'Terphenyl', 'Invalid'],
        'p_np': [1, 0, 1, 0, 0],
        'smiles': sample_smiles
    }
    return pd.DataFrame(data)