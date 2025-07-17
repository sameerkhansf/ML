"""
Data Handler Module for the Blood-Brain Barrier Permeability Prediction Project.

This module provides functionality for loading, validating, and preprocessing
the BBBP dataset, including SMILES validation and train-test splitting.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import DATA_FILES, MODEL_CONFIG, PATHS

# Set up logging
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Class for handling BBBP dataset operations including loading, validation,
    and preprocessing.
    """
    
    def __init__(self):
        """Initialize the DataHandler."""
        self.data = None
        self.mol_column = 'mol'
        self.summary_stats = None
        self.class_distribution = None
        
    def load_bbbp_data(self, file_path=None):
        """
        Load the BBBP dataset from CSV and validate its structure.
        
        Parameters
        ----------
        file_path : str or Path, optional
            Path to the BBBP dataset CSV file. If None, uses the default path from config.
            
        Returns
        -------
        pandas.DataFrame
            The loaded BBBP dataset with RDKit molecule objects.
        """
        if file_path is None:
            file_path = DATA_FILES['bbbp_dataset']
        
        logger.info(f"Loading BBBP dataset from {file_path}")
        
        try:
            # Load the CSV file
            self.data = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['num', 'name', 'p_np', 'smiles']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in dataset: {missing_columns}")
            
            # Add RDKit molecule objects
            self.data = self.validate_smiles(self.data)
            
            logger.info(f"Successfully loaded BBBP dataset with {len(self.data)} compounds")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading BBBP dataset: {str(e)}")
            raise
    
    def validate_smiles(self, data):
        """
        Validate SMILES strings and add RDKit molecule objects.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing a 'smiles' column.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added 'mol' column containing RDKit molecule objects.
            Invalid SMILES will have None in the 'mol' column.
        """
        logger.info("Validating SMILES strings and creating molecule objects")
        
        # Create molecule objects
        data[self.mol_column] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        
        # Count invalid SMILES
        invalid_count = data[self.mol_column].isna().sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid SMILES strings")
            
        return data
    
    def get_train_test_split(self, test_size=None, random_state=None, stratify=True):
        """
        Create stratified train-test split for model training.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the test split.
            If None, uses the value from config.
        random_state : int, optional
            Random seed for reproducibility. If None, uses the value from config.
        stratify : bool, default=True
            Whether to use stratified sampling based on the target variable.
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test) - Training and test splits.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_bbbp_data() first.")
        
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
            
        if random_state is None:
            random_state = MODEL_CONFIG['random_state']
        
        # Remove rows with invalid molecules
        valid_data = self.data.dropna(subset=[self.mol_column])
        
        # Prepare features and target
        X = valid_data.drop(columns=['p_np'])
        y = valid_data['p_np']
        
        # Create stratified split
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Created train-test split: {len(X_train)} training samples, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def export_results(self, data, file_path=None):
        """
        Save predictions and analysis results to CSV.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing results to export.
        file_path : str or Path, optional
            Path to save the results. If None, uses the processed_data path from config.
            
        Returns
        -------
        Path
            Path to the saved file.
        """
        if file_path is None:
            file_path = DATA_FILES['processed_data']
            
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        data.to_csv(file_path, index=False)
        logger.info(f"Results exported to {file_path}")
        
        return file_path    d
ef check_missing_values(self, data=None):
        """
        Check for missing values in the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to check. If None, uses self.data.
            
        Returns
        -------
        pandas.Series
            Series with count of missing values for each column.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        missing_values = data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        
        if len(missing_values) > 0:
            logger.info(f"Found missing values in {len(missing_values)} columns:\n{missing_values}")
        else:
            logger.info("No missing values found in the dataset")
        
        return missing_values
    
    def handle_missing_values(self, data=None, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to process. If None, uses self.data.
        strategy : str, default='drop'
            Strategy for handling missing values:
            - 'drop': Drop rows with missing values
            - 'drop_mol': Drop rows with missing molecule objects only
            - 'keep': Keep rows with missing values
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with handled missing values.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        if strategy == 'drop':
            result = data.dropna()
            logger.info(f"Dropped {len(data) - len(result)} rows with missing values")
        elif strategy == 'drop_mol':
            result = data.dropna(subset=[self.mol_column])
            logger.info(f"Dropped {len(data) - len(result)} rows with missing molecule objects")
        elif strategy == 'keep':
            result = data.copy()
            logger.info("Kept all rows including those with missing values")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result
    
    def detect_duplicates(self, data=None, subset=None):
        """
        Detect duplicate entries in the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to check. If None, uses self.data.
        subset : list, optional
            List of columns to consider for duplicate detection.
            If None, uses all columns except 'num' and 'mol'.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing duplicate entries.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        if subset is None:
            # Exclude 'num' and 'mol' columns from duplicate detection
            subset = [col for col in data.columns if col not in ['num', self.mol_column]]
        
        # Find duplicates
        duplicates = data[data.duplicated(subset=subset, keep=False)]
        
        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate entries based on {subset}")
        else:
            logger.info("No duplicate entries found")
        
        return duplicates
    
    def remove_duplicates(self, data=None, subset=None, keep='first'):
        """
        Remove duplicate entries from the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to process. If None, uses self.data.
        subset : list, optional
            List of columns to consider for duplicate detection.
            If None, uses all columns except 'num' and 'mol'.
        keep : {'first', 'last', False}, default='first'
            Which duplicates to keep:
            - 'first': Keep first occurrence of duplicates
            - 'last': Keep last occurrence of duplicates
            - False: Drop all duplicates
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with duplicates removed.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        if subset is None:
            # Exclude 'num' and 'mol' columns from duplicate detection
            subset = [col for col in data.columns if col not in ['num', self.mol_column]]
        
        # Remove duplicates
        result = data.drop_duplicates(subset=subset, keep=keep)
        
        logger.info(f"Removed {len(data) - len(result)} duplicate entries")
        
        return result
    
    def generate_data_summary(self, data=None):
        """
        Generate summary statistics for the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to analyze. If None, uses self.data.
            
        Returns
        -------
        dict
            Dictionary containing summary statistics.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        # Basic statistics
        total_compounds = len(data)
        valid_molecules = data[self.mol_column].notna().sum()
        invalid_molecules = total_compounds - valid_molecules
        
        # Class distribution
        class_counts = data['p_np'].value_counts()
        permeable = class_counts.get(1, 0)
        non_permeable = class_counts.get(0, 0)
        
        # Calculate class balance
        if total_compounds > 0:
            permeable_pct = permeable / total_compounds * 100
            non_permeable_pct = non_permeable / total_compounds * 100
        else:
            permeable_pct = 0
            non_permeable_pct = 0
        
        # Store summary statistics
        self.summary_stats = {
            'total_compounds': total_compounds,
            'valid_molecules': valid_molecules,
            'invalid_molecules': invalid_molecules,
            'permeable_compounds': permeable,
            'non_permeable_compounds': non_permeable,
            'permeable_percentage': permeable_pct,
            'non_permeable_percentage': non_permeable_pct
        }
        
        # Store class distribution
        self.class_distribution = {
            'permeable': permeable,
            'non_permeable': non_permeable
        }
        
        logger.info(f"Dataset summary: {total_compounds} compounds, "
                   f"{valid_molecules} valid molecules, "
                   f"{permeable} permeable ({permeable_pct:.1f}%), "
                   f"{non_permeable} non-permeable ({non_permeable_pct:.1f}%)")
        
        return self.summary_stats
    
    def plot_class_distribution(self, data=None, save_path=None):
        """
        Plot the class distribution of the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to analyze. If None, uses self.data.
        save_path : str or Path, optional
            Path to save the plot. If None, the plot is displayed but not saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        # Generate summary if not already done
        if self.class_distribution is None:
            self.generate_data_summary(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot class distribution
        sns.countplot(x='p_np', data=data, ax=ax, palette=['#ff9999', '#66b3ff'])
        
        # Add labels and title
        ax.set_xlabel('BBB Permeability')
        ax.set_ylabel('Count')
        ax.set_title('Blood-Brain Barrier Permeability Class Distribution')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Permeable (0)', 'Permeable (1)'])
        
        # Add count and percentage annotations
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            total = len(data)
            percentage = height / total * 100
            ax.text(p.get_x() + p.get_width()/2., height + 5,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha="center", fontsize=10)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        return fig
    
    def plot_missing_values(self, data=None, save_path=None):
        """
        Plot missing values in the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to analyze. If None, uses self.data.
        save_path : str or Path, optional
            Path to save the plot. If None, the plot is displayed but not saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        # Calculate missing values
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            logger.info("No missing values to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot missing values
        missing.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#ff7f0e')
        
        # Add labels and title
        ax.set_xlabel('Columns')
        ax.set_ylabel('Count')
        ax.set_title('Missing Values by Column')
        
        # Add count and percentage annotations
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            total = len(data)
            percentage = height / total * 100
            ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha="center", fontsize=9)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Missing values plot saved to {save_path}")
        
        return fig
    
    def analyze_molecular_properties(self, data=None):
        """
        Analyze basic molecular properties of the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to analyze. If None, uses self.data.
            
        Returns
        -------
        dict
            Dictionary containing molecular property statistics.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        # Filter valid molecules
        valid_data = data.dropna(subset=[self.mol_column])
        
        # Calculate molecular properties
        properties = {}
        
        # Molecular weight
        properties['molecular_weight'] = [Chem.Descriptors.MolWt(mol) for mol in valid_data[self.mol_column]]
        
        # Heavy atom count
        properties['heavy_atom_count'] = [Chem.Descriptors.HeavyAtomCount(mol) for mol in valid_data[self.mol_column]]
        
        # Ring count
        properties['ring_count'] = [Chem.Descriptors.RingCount(mol) for mol in valid_data[self.mol_column]]
        
        # Aromatic ring count
        properties['aromatic_ring_count'] = [Chem.Descriptors.NumAromaticRings(mol) for mol in valid_data[self.mol_column]]
        
        # Convert to DataFrame
        properties_df = pd.DataFrame(properties)
        
        # Add permeability class
        properties_df['p_np'] = valid_data['p_np'].values
        
        # Calculate statistics by class
        stats = properties_df.groupby('p_np').agg(['mean', 'std', 'min', 'max'])
        
        logger.info("Calculated molecular property statistics by permeability class")
        
        return stats
    
    def visualize_molecules(self, data=None, n_molecules=5, by_class=True, save_path=None):
        """
        Visualize representative molecules from the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            DataFrame to analyze. If None, uses self.data.
        n_molecules : int, default=5
            Number of molecules to visualize per class.
        by_class : bool, default=True
            Whether to visualize molecules separately by permeability class.
        save_path : str or Path, optional
            Path to save the visualization. If None, the visualization is displayed but not saved.
            
        Returns
        -------
        PIL.Image or tuple of PIL.Image
            The generated molecular visualizations.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_bbbp_data() first.")
            data = self.data
        
        # Filter valid molecules
        valid_data = data.dropna(subset=[self.mol_column])
        
        if by_class:
            # Separate by permeability class
            permeable = valid_data[valid_data['p_np'] == 1]
            non_permeable = valid_data[valid_data['p_np'] == 0]
            
            # Sample molecules from each class
            if len(permeable) > 0:
                permeable_sample = permeable.sample(min(n_molecules, len(permeable)))
                permeable_mols = permeable_sample[self.mol_column].tolist()
                permeable_names = permeable_sample['name'].tolist()
            else:
                permeable_mols = []
                permeable_names = []
            
            if len(non_permeable) > 0:
                non_permeable_sample = non_permeable.sample(min(n_molecules, len(non_permeable)))
                non_permeable_mols = non_permeable_sample[self.mol_column].tolist()
                non_permeable_names = non_permeable_sample['name'].tolist()
            else:
                non_permeable_mols = []
                non_permeable_names = []
            
            # Generate 2D coordinates for visualization
            for mol in permeable_mols + non_permeable_mols:
                AllChem.Compute2DCoords(mol)
            
            # Create visualizations
            if permeable_mols:
                permeable_img = Draw.MolsToGridImage(
                    permeable_mols,
                    molsPerRow=min(3, len(permeable_mols)),
                    subImgSize=(200, 200),
                    legends=permeable_names,
                    useSVG=True
                )
            else:
                permeable_img = None
            
            if non_permeable_mols:
                non_permeable_img = Draw.MolsToGridImage(
                    non_permeable_mols,
                    molsPerRow=min(3, len(non_permeable_mols)),
                    subImgSize=(200, 200),
                    legends=non_permeable_names,
                    useSVG=True
                )
            else:
                non_permeable_img = None
            
            # Save visualizations if path provided
            if save_path:
                if permeable_img:
                    permeable_path = str(save_path).replace('.svg', '_permeable.svg')
                    with open(permeable_path, 'w') as f:
                        f.write(permeable_img)
                    logger.info(f"Permeable molecules visualization saved to {permeable_path}")
                
                if non_permeable_img:
                    non_permeable_path = str(save_path).replace('.svg', '_non_permeable.svg')
                    with open(non_permeable_path, 'w') as f:
                        f.write(non_permeable_img)
                    logger.info(f"Non-permeable molecules visualization saved to {non_permeable_path}")
            
            return (permeable_img, non_permeable_img)
        
        else:
            # Sample molecules from the entire dataset
            sample = valid_data.sample(min(n_molecules, len(valid_data)))
            mols = sample[self.mol_column].tolist()
            names = sample['name'].tolist()
            
            # Generate 2D coordinates for visualization
            for mol in mols:
                AllChem.Compute2DCoords(mol)
            
            # Create visualization
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=min(3, len(mols)),
                subImgSize=(200, 200),
                legends=names,
                useSVG=True
            )
            
            # Save visualization if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(img)
                logger.info(f"Molecules visualization saved to {save_path}")
            
            return img