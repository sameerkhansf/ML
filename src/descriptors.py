"""
Molecular Descriptor Calculator for the Blood-Brain Barrier Permeability Prediction Project.

This module provides functionality for calculating molecular descriptors relevant to
BBB permeability from SMILES strings using RDKit.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
from pathlib import Path

from config import DESCRIPTOR_CONFIG, DATA_FILES

# Set up logging
logger = logging.getLogger(__name__)

class MolecularDescriptors:
    """
    Class for calculating molecular descriptors relevant to BBB permeability.
    """
    
    def __init__(self):
        """Initialize the MolecularDescriptors calculator."""
        self.descriptor_names = []
        
    def calculate_lipophilicity(self, mol):
        """
        Calculate lipophilicity descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of lipophilicity descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['lipophilicity']}
        
        descriptors = {}
        try:
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
        except:
            descriptors['MolLogP'] = None
            
        return descriptors
    
    def calculate_size_descriptors(self, mol):
        """
        Calculate molecular size descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of size descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['size']}
        
        descriptors = {}
        try:
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        except:
            descriptors['MolWt'] = None
            descriptors['HeavyAtomCount'] = None
            descriptors['NumHeteroatoms'] = None
            
        return descriptors
    
    def calculate_polarity_descriptors(self, mol):
        """
        Calculate molecular polarity descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of polarity descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['polarity']}
        
        descriptors = {}
        try:
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
        except:
            descriptors['TPSA'] = None
            descriptors['LabuteASA'] = None
            
        return descriptors
    
    def calculate_hb_descriptors(self, mol):
        """
        Calculate hydrogen bonding descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of hydrogen bonding descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['hydrogen_bonding']}
        
        descriptors = {}
        try:
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        except:
            descriptors['NumHDonors'] = None
            descriptors['NumHAcceptors'] = None
            
        return descriptors
    
    def calculate_structural_descriptors(self, mol):
        """
        Calculate structural descriptors like rotatable bonds and aromatic rings.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of structural descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['flexibility'] + DESCRIPTOR_CONFIG['aromaticity']}
        
        descriptors = {}
        try:
            # Flexibility descriptors
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Aromaticity descriptors
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            descriptors['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
            descriptors['RingCount'] = Descriptors.RingCount(mol)
        except Exception as e:
            logger.warning(f"Error calculating structural descriptors: {str(e)}")
            descriptors['NumRotatableBonds'] = None
            descriptors['NumAromaticRings'] = None
            descriptors['NumSaturatedRings'] = None
            descriptors['NumAliphaticRings'] = None
            descriptors['NumAromaticHeterocycles'] = None
            descriptors['NumAromaticCarbocycles'] = None
            descriptors['RingCount'] = None
            
        return descriptors
    
    def calculate_complexity_descriptors(self, mol):
        """
        Calculate molecular complexity descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of complexity descriptors.
        """
        if mol is None:
            return {name: None for name in DESCRIPTOR_CONFIG['complexity']}
        
        descriptors = {}
        try:
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            descriptors['Ipc'] = Descriptors.Ipc(mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
        except Exception as e:
            logger.warning(f"Error calculating complexity descriptors: {str(e)}")
            descriptors['BertzCT'] = None
            descriptors['FractionCsp3'] = None
            descriptors['HallKierAlpha'] = None
            descriptors['Ipc'] = None
            descriptors['Kappa1'] = None
            descriptors['Kappa2'] = None
            descriptors['Kappa3'] = None
            
        return descriptors
    
    def calculate_lipinski_descriptors(self, mol):
        """
        Calculate Lipinski's Rule of Five descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of Lipinski descriptors.
        """
        if mol is None:
            return {'LipinskiHBA': None, 'LipinskiHBD': None, 'NumRotatableBonds': None, 'MolWt': None, 'MolLogP': None}
        
        descriptors = {}
        try:
            descriptors['LipinskiHBA'] = Lipinski.NumHAcceptors(mol)
            descriptors['LipinskiHBD'] = Lipinski.NumHDonors(mol)
            descriptors['NumRotatableBonds'] = Lipinski.NumRotatableBonds(mol)
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            
            # Calculate Lipinski violations
            violations = 0
            if descriptors['MolWt'] > 500: violations += 1
            if descriptors['MolLogP'] > 5: violations += 1
            if descriptors['LipinskiHBD'] > 5: violations += 1
            if descriptors['LipinskiHBA'] > 10: violations += 1
            
            descriptors['LipinskiViolations'] = violations
        except Exception as e:
            logger.warning(f"Error calculating Lipinski descriptors: {str(e)}")
            descriptors['LipinskiHBA'] = None
            descriptors['LipinskiHBD'] = None
            descriptors['NumRotatableBonds'] = None
            descriptors['MolWt'] = None
            descriptors['MolLogP'] = None
            descriptors['LipinskiViolations'] = None
            
        return descriptors
    
    def calculate_veber_descriptors(self, mol):
        """
        Calculate Veber rule descriptors for oral bioavailability.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of Veber descriptors.
        """
        if mol is None:
            return {'TPSA': None, 'NumRotatableBonds': None, 'VeberViolations': None}
        
        descriptors = {}
        try:
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Calculate Veber violations
            violations = 0
            if descriptors['TPSA'] > 140: violations += 1
            if descriptors['NumRotatableBonds'] > 10: violations += 1
            
            descriptors['VeberViolations'] = violations
        except Exception as e:
            logger.warning(f"Error calculating Veber descriptors: {str(e)}")
            descriptors['TPSA'] = None
            descriptors['NumRotatableBonds'] = None
            descriptors['VeberViolations'] = None
            
        return descriptors
    
    def calculate_all_descriptors(self, mol):
        """
        Calculate all molecular descriptors.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
            
        Returns
        -------
        dict
            Dictionary of all molecular descriptors.
        """
        if mol is None:
            logger.warning("Cannot calculate descriptors for None molecule")
            return {}
        
        all_descriptors = {}
        
        # Calculate all descriptor categories
        all_descriptors.update(self.calculate_lipophilicity(mol))
        all_descriptors.update(self.calculate_size_descriptors(mol))
        all_descriptors.update(self.calculate_polarity_descriptors(mol))
        all_descriptors.update(self.calculate_hb_descriptors(mol))
        all_descriptors.update(self.calculate_structural_descriptors(mol))
        all_descriptors.update(self.calculate_complexity_descriptors(mol))
        all_descriptors.update(self.calculate_lipinski_descriptors(mol))
        all_descriptors.update(self.calculate_veber_descriptors(mol))
        
        # Update descriptor names
        self.descriptor_names = list(all_descriptors.keys())
        
        return all_descriptors
    
    def calculate_batch_descriptors(self, molecules):
        """
        Calculate descriptors for a batch of molecules.
        
        Parameters
        ----------
        molecules : list or pandas.Series
            List of RDKit molecule objects or pandas Series containing molecules.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing calculated descriptors for each molecule.
        """
        results = []
        
        for i, mol in enumerate(molecules):
            try:
                descriptors = self.calculate_all_descriptors(mol)
                results.append(descriptors)
            except Exception as e:
                logger.error(f"Error calculating descriptors for molecule {i}: {str(e)}")
                results.append({})
        
        return pd.DataFrame(results)
    
    def calculate_from_dataframe(self, df, mol_column='mol', save_path=None):
        """
        Calculate descriptors for molecules in a DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing molecules.
        mol_column : str, default='mol'
            Name of the column containing RDKit molecule objects.
        save_path : str or Path, optional
            Path to save the descriptors. If None, uses the descriptors path from config.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with original data and calculated descriptors.
        """
        if mol_column not in df.columns:
            raise ValueError(f"Column '{mol_column}' not found in DataFrame")
        
        logger.info(f"Calculating descriptors for {len(df)} molecules")
        
        # Calculate descriptors
        descriptors_df = self.calculate_batch_descriptors(df[mol_column])
        
        # Combine with original data
        result_df = pd.concat([df, descriptors_df], axis=1)
        
        # Save if path provided
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(save_path, index=False)
            logger.info(f"Descriptors saved to {save_path}")
        elif DATA_FILES.get('descriptors') is not None:
            result_df.to_csv(DATA_FILES['descriptors'], index=False)
            logger.info(f"Descriptors saved to {DATA_FILES['descriptors']}")
        
        return result_df