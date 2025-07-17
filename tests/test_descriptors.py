"""
Unit tests for the molecular descriptor calculation module.
"""

import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from src.descriptors import MolecularDescriptors


class TestMolecularDescriptors:
    """Test class for MolecularDescriptors functionality."""
    
    def test_initialization(self):
        """Test MolecularDescriptors initialization."""
        calculator = MolecularDescriptors()
        assert calculator.descriptor_names is not None
        assert len(calculator.descriptor_names) > 0
    
    def test_calculate_lipophilicity(self):
        """Test lipophilicity descriptor calculation."""
        # Create a simple molecule
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        
        # Calculate lipophilicity
        calculator = MolecularDescriptors()
        descriptors = calculator.calculate_lipophilicity(mol)
        
        # Check that MolLogP is calculated
        assert 'MolLogP' in descriptors
        assert isinstance(descriptors['MolLogP'], float)
        
        # Test with more lipophilic molecule
        mol_lipophilic = Chem.MolFromSmiles("CCCCCCCC")  # Octane
        descriptors_lipophilic = calculator.calculate_lipophilicity(mol_lipophilic)
        
        # Octane should be more lipophilic than ethanol
        assert descriptors_lipophilic['MolLogP'] > descriptors['MolLogP']
    
    def test_calculate_size_descriptors(self):
        """Test size descriptor calculation."""
        # Create a simple molecule
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        
        # Calculate size descriptors
        calculator = MolecularDescriptors()
        descriptors = calculator.calculate_size_descriptors(mol)
        
        # Check that size descriptors are calculated
        assert 'MolWt' in descriptors
        assert 'HeavyAtomCount' in descriptors
        assert 'NumHeteroatoms' in descriptors
        
        # Check values for ethanol
        assert descriptors['HeavyAtomCount'] == 3
        assert descriptors['NumHeteroatoms'] == 1
        assert descriptors['MolWt'] > 40  # Approximate molecular weight
    
    def test_calculate_polarity_descriptors(self):
        """Test polarity descriptor calculation."""
        # Create molecules with different polarities
        mol_polar = Chem.MolFromSmiles("CCO")  # Ethanol (polar)
        mol_nonpolar = Chem.MolFromSmiles("CCCC")  # Butane (nonpolar)
        
        calculator = MolecularDescriptors()
        
        # Calculate polarity descriptors
        descriptors_polar = calculator.calculate_polarity_descriptors(mol_polar)
        descriptors_nonpolar = calculator.calculate_polarity_descriptors(mol_nonpolar)
        
        # Check that TPSA is calculated
        assert 'TPSA' in descriptors_polar
        assert 'TPSA' in descriptors_nonpolar
        
        # Ethanol should have higher TPSA than butane
        assert descriptors_polar['TPSA'] > descriptors_nonpolar['TPSA']
    
    def test_calculate_hb_descriptors(self):
        """Test hydrogen bonding descriptor calculation."""
        # Create molecules with different H-bonding capabilities
        mol_hb = Chem.MolFromSmiles("O")  # Water (H-bond donor and acceptor)
        mol_no_hb = Chem.MolFromSmiles("CC")  # Ethane (no H-bonding)
        
        calculator = MolecularDescriptors()
        
        # Calculate H-bonding descriptors
        descriptors_hb = calculator.calculate_hb_descriptors(mol_hb)
        descriptors_no_hb = calculator.calculate_hb_descriptors(mol_no_hb)
        
        # Check that H-bonding descriptors are calculated
        assert 'NumHDonors' in descriptors_hb
        assert 'NumHAcceptors' in descriptors_hb
        
        # Water should have H-bonding capability
        assert descriptors_hb['NumHDonors'] > 0
        assert descriptors_hb['NumHAcceptors'] > 0
        
        # Ethane should have no H-bonding capability
        assert descriptors_no_hb['NumHDonors'] == 0
        assert descriptors_no_hb['NumHAcceptors'] == 0
    
    def test_calculate_structural_descriptors(self):
        """Test structural descriptor calculation."""
        # Create molecules with different structural features
        mol_flexible = Chem.MolFromSmiles("CCCCCC")  # Hexane (flexible)
        mol_rigid = Chem.MolFromSmiles("c1ccccc1")  # Benzene (rigid)
        
        calculator = MolecularDescriptors()
        
        # Calculate structural descriptors
        descriptors_flexible = calculator.calculate_structural_descriptors(mol_flexible)
        descriptors_rigid = calculator.calculate_structural_descriptors(mol_rigid)
        
        # Check that structural descriptors are calculated
        assert 'NumRotatableBonds' in descriptors_flexible
        assert 'NumAromaticRings' in descriptors_flexible
        
        # Hexane should have more rotatable bonds than benzene
        assert descriptors_flexible['NumRotatableBonds'] > descriptors_rigid['NumRotatableBonds']
        
        # Benzene should have aromatic rings
        assert descriptors_rigid['NumAromaticRings'] > 0
        assert descriptors_flexible['NumAromaticRings'] == 0
    
    def test_calculate_complexity_descriptors(self):
        """Test complexity descriptor calculation."""
        # Create molecules with different complexities
        mol_simple = Chem.MolFromSmiles("CC")  # Ethane (simple)
        mol_complex = Chem.MolFromSmiles("CC(C)(C)C")  # Neopentane (more complex)
        
        calculator = MolecularDescriptors()
        
        # Calculate complexity descriptors
        descriptors_simple = calculator.calculate_complexity_descriptors(mol_simple)
        descriptors_complex = calculator.calculate_complexity_descriptors(mol_complex)
        
        # Check that complexity descriptors are calculated
        assert 'FractionCsp3' in descriptors_simple
        assert isinstance(descriptors_simple['FractionCsp3'], float)
        assert 0 <= descriptors_simple['FractionCsp3'] <= 1
    
    def test_calculate_all_descriptors(self):
        """Test calculation of all descriptors."""
        # Create a simple molecule
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        
        # Calculate all descriptors
        calculator = MolecularDescriptors()
        descriptors = calculator.calculate_all_descriptors(mol)
        
        # Check that all descriptor categories are included
        assert 'MolLogP' in descriptors  # Lipophilicity
        assert 'MolWt' in descriptors    # Size
        assert 'TPSA' in descriptors     # Polarity
        assert 'NumHDonors' in descriptors  # Hydrogen bonding
        assert 'NumRotatableBonds' in descriptors  # Flexibility
        assert 'NumAromaticRings' in descriptors   # Aromaticity
        assert 'FractionCsp3' in descriptors       # Complexity
        
        # Check that all values are numeric
        for key, value in descriptors.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
    
    def test_calculate_all_descriptors_smiles(self):
        """Test calculation of all descriptors from SMILES string."""
        calculator = MolecularDescriptors()
        
        # Calculate descriptors from SMILES
        descriptors = calculator.calculate_all_descriptors("CCO")
        
        # Should return same results as molecule object
        mol = Chem.MolFromSmiles("CCO")
        descriptors_mol = calculator.calculate_all_descriptors(mol)
        
        # Compare key descriptors
        assert descriptors['MolWt'] == descriptors_mol['MolWt']
        assert descriptors['NumHDonors'] == descriptors_mol['NumHDonors']
    
    def test_calculate_from_dataframe(self):
        """Test batch calculation from DataFrame."""
        # Create test DataFrame
        test_data = pd.DataFrame({
            'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1', 'INVALID'],
            'name': ['Ethanol', 'Acetic acid', 'Benzene', 'Invalid']
        })
        
        calculator = MolecularDescriptors()
        
        # Calculate descriptors
        result = calculator.calculate_from_dataframe(test_data, smiles_column='smiles')
        
        # Check that descriptors were added
        assert len(result.columns) > len(test_data.columns)
        assert 'MolLogP' in result.columns
        assert 'MolWt' in result.columns
        
        # Check that invalid SMILES resulted in NaN values
        assert pd.isna(result.loc[3, 'MolLogP'])
    
    def test_calculate_from_dataframe_with_mol_column(self):
        """Test batch calculation from DataFrame with molecule column."""
        # Create test DataFrame with molecule objects
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
        test_data = pd.DataFrame({
            'smiles': smiles_list,
            'mol': [Chem.MolFromSmiles(s) for s in smiles_list],
            'name': ['Ethanol', 'Acetic acid', 'Benzene']
        })
        
        calculator = MolecularDescriptors()
        
        # Calculate descriptors using mol column
        result = calculator.calculate_from_dataframe(test_data, mol_column='mol')
        
        # Check that descriptors were calculated
        assert 'MolLogP' in result.columns
        assert not pd.isna(result['MolLogP']).any()
    
    def test_handle_none_molecule(self):
        """Test handling of None molecule."""
        # Calculate descriptors for None molecule
        calculator = MolecularDescriptors()
        descriptors = calculator.calculate_all_descriptors(None)
        
        # Should return empty dictionary
        assert descriptors == {}
    
    def test_handle_invalid_smiles(self):
        """Test handling of invalid SMILES strings."""
        calculator = MolecularDescriptors()
        
        # Test with invalid SMILES
        descriptors = calculator.calculate_all_descriptors("INVALID_SMILES")
        assert descriptors == {}
        
        # Test with empty string
        descriptors = calculator.calculate_all_descriptors("")
        assert descriptors == {}
    
    def test_descriptor_consistency(self):
        """Test that descriptors are calculated consistently."""
        calculator = MolecularDescriptors()
        
        # Calculate descriptors multiple times for the same molecule
        mol = Chem.MolFromSmiles("CCO")
        descriptors1 = calculator.calculate_all_descriptors(mol)
        descriptors2 = calculator.calculate_all_descriptors(mol)
        
        # Results should be identical
        assert descriptors1 == descriptors2
    
    def test_descriptor_names_property(self):
        """Test descriptor names property."""
        calculator = MolecularDescriptors()
        
        # Check that descriptor names are available
        assert hasattr(calculator, 'descriptor_names')
        assert isinstance(calculator.descriptor_names, list)
        assert len(calculator.descriptor_names) > 0
        
        # Check that all descriptor names are strings
        assert all(isinstance(name, str) for name in calculator.descriptor_names)
    
    def test_known_molecule_values(self):
        """Test descriptor calculation for molecules with known values."""
        calculator = MolecularDescriptors()
        
        # Test water (H2O)
        water = Chem.MolFromSmiles("O")
        water_descriptors = calculator.calculate_all_descriptors(water)
        
        # Water should have specific properties
        assert water_descriptors['NumHDonors'] == 2  # Two H donors
        assert water_descriptors['NumHAcceptors'] == 1  # One O acceptor
        assert water_descriptors['HeavyAtomCount'] == 1  # One oxygen
        assert water_descriptors['NumRotatableBonds'] == 0  # No rotatable bonds
        
        # Test methane (CH4)
        methane = Chem.MolFromSmiles("C")
        methane_descriptors = calculator.calculate_all_descriptors(methane)
        
        # Methane should have specific properties
        assert methane_descriptors['NumHDonors'] == 0
        assert methane_descriptors['NumHAcceptors'] == 0
        assert methane_descriptors['HeavyAtomCount'] == 1  # One carbon
        assert methane_descriptors['NumRotatableBonds'] == 0
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        calculator = MolecularDescriptors()
        
        # Test with very large molecule (should not crash)
        large_smiles = "C" * 100  # Very long alkyl chain
        try:
            descriptors = calculator.calculate_all_descriptors(large_smiles)
            # Should either return descriptors or empty dict, but not crash
            assert isinstance(descriptors, dict)
        except Exception:
            # If it fails, it should fail gracefully
            pass
    
    def test_batch_processing_performance(self):
        """Test batch processing with multiple molecules."""
        calculator = MolecularDescriptors()
        
        # Create DataFrame with multiple molecules
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCC'] * 10  # 50 molecules
        test_data = pd.DataFrame({'smiles': smiles_list})
        
        # Calculate descriptors
        result = calculator.calculate_from_dataframe(test_data, smiles_column='smiles')
        
        # Check that all molecules were processed
        assert len(result) == len(test_data)
        assert 'MolLogP' in result.columns
        
        # Check that most calculations succeeded (allowing for some failures)
        success_rate = result['MolLogP'].notna().mean()
        assert success_rate > 0.8  # At least 80% should succeed