"""
Configuration file for the Blood-Brain Barrier Permeability Prediction Project.

This file contains all the configuration parameters, file paths, and settings
used throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory paths
PATHS = {
    'data': PROJECT_ROOT / 'data',
    'src': PROJECT_ROOT / 'src', 
    'notebooks': PROJECT_ROOT / 'notebooks',
    'results': PROJECT_ROOT / 'results',
    'tests': PROJECT_ROOT / 'tests',
    'models': PROJECT_ROOT / 'results' / 'models',
    'plots': PROJECT_ROOT / 'results' / 'plots',
    'reports': PROJECT_ROOT / 'results' / 'reports'
}

# Data file paths
DATA_FILES = {
    'bbbp_dataset': PATHS['data'] / 'BBBP.csv',
    'processed_data': PATHS['results'] / 'processed_bbbp.csv',
    'descriptors': PATHS['results'] / 'molecular_descriptors.csv'
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring_metric': 'roc_auc',
    'n_jobs': -1  # Use all available cores
}

# Molecular descriptor configuration
DESCRIPTOR_CONFIG = {
    'lipophilicity': ['MolLogP', 'MolWt'],
    'size': ['MolWt', 'HeavyAtomCount', 'NumHeteroatoms'],
    'polarity': ['TPSA', 'LabuteASA'],
    'hydrogen_bonding': ['NumHDonors', 'NumHAcceptors'],
    'flexibility': ['NumRotatableBonds'],
    'aromaticity': ['NumAromaticRings', 'NumSaturatedRings'],
    'complexity': ['BertzCT', 'FractionCsp3']
}

# Machine learning algorithms and their hyperparameters
ML_ALGORITHMS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    },
    'neural_network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }
}

# Visualization configuration
PLOT_CONFIG = {
    'figure_size': (10, 8),
    'dpi': 300,
    'style': 'default',
    'color_palette': 'husl',
    'save_format': 'png'
}

# SHAP analysis configuration
SHAP_CONFIG = {
    'max_display': 20,
    'sample_size': 100,  # For SHAP analysis to speed up computation
    'plot_size': (12, 8)
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PATHS['results'] / 'analysis.log'
}

# Create necessary directories
def create_directories():
    """Create all necessary directories if they don't exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories in results
    (PATHS['results'] / 'models').mkdir(exist_ok=True)
    (PATHS['results'] / 'plots').mkdir(exist_ok=True)
    (PATHS['results'] / 'reports').mkdir(exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    for name, path in PATHS.items():
        print(f"{name}: {path}")