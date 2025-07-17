# Blood-Brain Barrier Permeability Prediction Project

A computational chemistry research initiative focused on understanding and predicting molecular permeability across the blood-brain barrier (BBB) using machine learning.

## Project Overview

The blood-brain barrier represents one of the most significant challenges in central nervous system drug development. This project aims to develop machine learning models that can predict BBB permeability from molecular structure, providing insights into the key molecular descriptors and structural features that influence brain drug delivery.

## Features

- Prediction of BBB permeability from SMILES strings
- Calculation and analysis of molecular descriptors relevant to BBB permeability
- Machine learning model development with multiple algorithms
- Structure-activity relationship analysis
- Comprehensive visualizations and interpretability analysis

## Project Structure

```
├── config.py                  # Configuration parameters and paths
├── data/                      # Data directory
│   └── BBBP.csv               # Blood-Brain Barrier Permeability dataset
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_cleaning.ipynb        # Data loading and preprocessing
│   ├── 02_descriptor_summary.ipynb   # Molecular descriptor analysis
│   ├── 03_model_training.ipynb       # Model training and evaluation
│   └── 04_interpretation.ipynb       # Model interpretation and insights
├── results/                   # Results directory
│   ├── models/                # Saved models
│   ├── plots/                 # Generated plots and visualizations
│   └── reports/               # Analysis reports
├── src/                       # Source code
│   ├── data_handler.py        # Data loading and preprocessing
│   ├── descriptors.py         # Molecular descriptor calculation
│   ├── feature_engineering.py # Feature preprocessing and selection
│   └── models.py              # Machine learning model training
└── tests/                     # Unit tests
```

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

```python
from src.data_handler import DataHandler

# Load and preprocess the BBBP dataset
data_handler = DataHandler()
data = data_handler.load_bbbp_data()

# Get train-test split
X_train, X_test, y_train, y_test = data_handler.get_train_test_split()
```

### Molecular Descriptor Calculation

```python
from src.descriptors import MolecularDescriptors

# Calculate molecular descriptors
descriptor_calculator = MolecularDescriptors()
descriptors_df = descriptor_calculator.calculate_from_dataframe(data)
```

### Model Training and Evaluation

```python
from src.models import ModelTrainer

# Train models
model_trainer = ModelTrainer()
model_trainer.train_random_forest(X_train, y_train)
model_trainer.train_xgboost(X_train, y_train)

# Evaluate models
cv_results = model_trainer.cross_validate_models(X_train, y_train)
comparison = model_trainer.compare_models(cv_results)

# Save best model
model_trainer.save_model()
```

## Dataset

The project uses the BBBP (Blood-Brain Barrier Permeability) dataset, which includes:
- Molecular structures as SMILES strings
- Binary permeability labels (permeable/non-permeable)
- Additional molecular information

## License

[MIT License](LICENSE)

## Acknowledgments

- RDKit for molecular informatics
- scikit-learn for machine learning algorithms
- SHAP for model interpretability