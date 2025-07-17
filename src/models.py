"""
Model Training Pipeline for the Blood-Brain Barrier Permeability Prediction Project.

This module provides functionality for training and evaluating machine learning models
for BBB permeability prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
import logging
from pathlib import Path

from config import MODEL_CONFIG, PATHS

# Set up logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for BBB permeability prediction.
    """
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
    
    def train_logistic_regression(self, X, y, params=None):
        """
        Train a logistic regression model.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        params : dict, optional
            Hyperparameters for logistic regression.
            
        Returns
        -------
        sklearn.linear_model.LogisticRegression
            Trained logistic regression model.
        """
        logger.info("Training logistic regression model")
        
        if params is None:
            params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': MODEL_CONFIG['random_state']
            }
        
        model = LogisticRegression(**params)
        model.fit(X, y)
        
        self.models['logistic_regression'] = model
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return model
    
    def train_random_forest(self, X, y, params=None, grid_search=False):
        """
        Train a random forest model with optional hyperparameter tuning.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        params : dict, optional
            Hyperparameters for random forest.
        grid_search : bool, default=False
            Whether to perform grid search for hyperparameter tuning.
            
        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            Trained random forest model.
        """
        logger.info("Training random forest model")
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': MODEL_CONFIG['random_state']
            }
        
        if grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = GridSearchCV(
                RandomForestClassifier(random_state=MODEL_CONFIG['random_state']),
                param_grid,
                cv=MODEL_CONFIG['cv_folds'],
                scoring=MODEL_CONFIG['scoring_metric'],
                n_jobs=MODEL_CONFIG['n_jobs']
            )
            
            model.fit(X, y)
            
            logger.info(f"Best parameters from grid search: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = RandomForestClassifier(**params)
            model.fit(X, y)
        
        self.models['random_forest'] = model
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return model
    
    def train_svm(self, X, y, params=None, grid_search=False):
        """
        Train a support vector machine model with optional hyperparameter tuning.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        params : dict, optional
            Hyperparameters for SVM.
        grid_search : bool, default=False
            Whether to perform grid search for hyperparameter tuning.
            
        Returns
        -------
        sklearn.svm.SVC
            Trained SVM model.
        """
        logger.info("Training SVM model")
        
        if params is None:
            params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': MODEL_CONFIG['random_state']
            }
        
        if grid_search:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            
            model = GridSearchCV(
                SVC(probability=True, random_state=MODEL_CONFIG['random_state']),
                param_grid,
                cv=MODEL_CONFIG['cv_folds'],
                scoring=MODEL_CONFIG['scoring_metric'],
                n_jobs=MODEL_CONFIG['n_jobs']
            )
            
            model.fit(X, y)
            
            logger.info(f"Best parameters from grid search: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = SVC(**params)
            model.fit(X, y)
        
        self.models['svm'] = model
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return model
    
    def train_xgboost(self, X, y, params=None, grid_search=False):
        """
        Train an XGBoost model with optional hyperparameter tuning.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        params : dict, optional
            Hyperparameters for XGBoost.
        grid_search : bool, default=False
            Whether to perform grid search for hyperparameter tuning.
            
        Returns
        -------
        xgboost.XGBClassifier
            Trained XGBoost model.
        """
        logger.info("Training XGBoost model")
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': MODEL_CONFIG['random_state']
            }
        
        if grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            model = GridSearchCV(
                xgb.XGBClassifier(random_state=MODEL_CONFIG['random_state']),
                param_grid,
                cv=MODEL_CONFIG['cv_folds'],
                scoring=MODEL_CONFIG['scoring_metric'],
                n_jobs=MODEL_CONFIG['n_jobs']
            )
            
            model.fit(X, y)
            
            logger.info(f"Best parameters from grid search: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(X, y)
        
        self.models['xgboost'] = model
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return model
    
    def train_neural_network(self, X, y, params=None, grid_search=False):
        """
        Train a neural network model with optional hyperparameter tuning.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        params : dict, optional
            Hyperparameters for neural network.
        grid_search : bool, default=False
            Whether to perform grid search for hyperparameter tuning.
            
        Returns
        -------
        sklearn.neural_network.MLPClassifier
            Trained neural network model.
        """
        logger.info("Training neural network model")
        
        if params is None:
            params = {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': MODEL_CONFIG['random_state']
            }
        
        if grid_search:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
            
            model = GridSearchCV(
                MLPClassifier(random_state=MODEL_CONFIG['random_state']),
                param_grid,
                cv=MODEL_CONFIG['cv_folds'],
                scoring=MODEL_CONFIG['scoring_metric'],
                n_jobs=MODEL_CONFIG['n_jobs']
            )
            
            model.fit(X, y)
            
            logger.info(f"Best parameters from grid search: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = MLPClassifier(**params)
            model.fit(X, y)
        
        self.models['neural_network'] = model
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return model
    
    def evaluate_model(self, model, X, y, model_name=None):
        """
        Evaluate a model's performance on a test set.
        
        Parameters
        ----------
        model : object
            Trained model with predict and predict_proba methods.
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        model_name : str, optional
            Name of the model for logging.
            
        Returns
        -------
        dict
            Dictionary of performance metrics.
        """
        if model_name is None:
            model_name = type(model).__name__
        
        logger.info(f"Evaluating {model_name} model")
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        logger.info(f"{model_name} performance: Accuracy={metrics['accuracy']:.4f}, "
                   f"AUC-ROC={metrics['auc_roc']:.4f}")
        
        return metrics
    
    def cross_validate_models(self, X, y, cv=None):
        """
        Perform cross-validation for all trained models.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target variable.
        cv : int or cross-validation generator, optional
            Cross-validation strategy. If None, uses the cv_folds from config.
            
        Returns
        -------
        dict
            Dictionary of cross-validation results for each model.
        """
        if not self.models:
            raise ValueError("No models trained. Train models first.")
        
        if cv is None:
            cv = StratifiedKFold(n_splits=MODEL_CONFIG['cv_folds'], 
                                shuffle=True, 
                                random_state=MODEL_CONFIG['random_state'])
        
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name} model")
            
            scores = cross_validate(
                model, X, y, 
                cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                return_train_score=True
            )
            
            cv_results[name] = {
                'test_accuracy': scores['test_accuracy'].mean(),
                'test_precision': scores['test_precision'].mean(),
                'test_recall': scores['test_recall'].mean(),
                'test_f1': scores['test_f1'].mean(),
                'test_roc_auc': scores['test_roc_auc'].mean(),
                'train_accuracy': scores['train_accuracy'].mean(),
                'train_roc_auc': scores['train_roc_auc'].mean(),
                'cv_scores': scores
            }
            
            logger.info(f"{name} cross-validation: Accuracy={cv_results[name]['test_accuracy']:.4f}, "
                       f"AUC-ROC={cv_results[name]['test_roc_auc']:.4f}")
        
        return cv_results
    
    def compare_models(self, cv_results=None, X=None, y=None, metric='test_roc_auc'):
        """
        Compare models based on cross-validation results or by performing cross-validation.
        
        Parameters
        ----------
        cv_results : dict, optional
            Dictionary of cross-validation results from cross_validate_models.
        X : pandas.DataFrame or numpy.ndarray, optional
            Feature matrix, required if cv_results is None.
        y : pandas.Series or numpy.ndarray, optional
            Target variable, required if cv_results is None.
        metric : str, default='test_roc_auc'
            Metric to use for comparison.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with model comparison results.
        """
        if cv_results is None:
            if X is None or y is None:
                raise ValueError("Either cv_results or both X and y must be provided")
            
            cv_results = self.cross_validate_models(X, y)
        
        # Extract results for comparison
        comparison = {}
        for name, results in cv_results.items():
            comparison[name] = {
                'accuracy': results['test_accuracy'],
                'precision': results['test_precision'],
                'recall': results['test_recall'],
                'f1': results['test_f1'],
                'roc_auc': results['test_roc_auc'],
                'train_accuracy': results['train_accuracy'],
                'train_roc_auc': results['train_roc_auc']
            }
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison).T
        
        # Find best model
        best_model_name = comparison_df[metric.replace('test_', '')].idxmax()
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model based on {metric}: {best_model_name}")
        
        return comparison_df
    
    def save_model(self, model=None, model_name=None, file_path=None):
        """
        Save a trained model to disk.
        
        Parameters
        ----------
        model : object, optional
            Model to save. If None, uses the best model.
        model_name : str, optional
            Name of the model. If None and model is None, uses the best model name.
        file_path : str or Path, optional
            Path to save the model. If None, uses the models directory from config.
            
        Returns
        -------
        str
            Path to the saved model.
        """
        if model is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Call compare_models first.")
            model = self.best_model
            model_name = self.best_model_name
        
        if model_name is None:
            model_name = type(model).__name__
        
        if file_path is None:
            models_dir = PATHS['models']
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            file_path = models_dir / f"{model_name}.joblib"
        
        # Save model
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
        
        return file_path
    
    def load_model(self, file_path):
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the saved model.
            
        Returns
        -------
        object
            Loaded model.
        """
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        return model