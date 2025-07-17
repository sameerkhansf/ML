"""
Main Analysis Coordinator for the Blood-Brain Barrier Permeability Prediction Project.

This module orchestrates the complete analysis workflow from data loading to results generation,
integrating all components for comprehensive BBB permeability analysis.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from datetime import datetime
import json

from src.data_handler import DataHandler
from src.descriptors import MolecularDescriptors
from src.feature_engineering import FeatureEngineering
from src.models import ModelTrainer
from src.interpretability import InterpretabilityEngine
from src.visualization import VisualizationSuite
from config import DATA_FILES, PATHS, MODEL_CONFIG, LOGGING_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MainAnalysis:
    """
    Main analysis coordinator that orchestrates the complete BBB permeability prediction workflow.
    """
    
    def __init__(self):
        """Initialize the MainAnalysis coordinator."""
        self.data_handler = DataHandler()
        self.descriptor_calculator = MolecularDescriptors()
        self.feature_engineer = FeatureEngineering()
        self.model_trainer = ModelTrainer()
        self.interpretability_engine = InterpretabilityEngine()
        self.visualization_suite = VisualizationSuite()
        
        # Analysis results storage
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.model_results = {}
        self.feature_importance = None
        self.analysis_report = {}
        
        logger.info("MainAnalysis coordinator initialized")
    
    def run_complete_analysis(self, data_path=None, save_results=True):
        """
        Execute the complete analysis pipeline from data loading to results generation.
        
        Parameters
        ----------
        data_path : str or Path, optional
            Path to the BBBP dataset. If None, uses default from config.
        save_results : bool, default=True
            Whether to save analysis results to files.
            
        Returns
        -------
        dict
            Dictionary containing analysis results and summary.
        """
        logger.info("Starting complete BBB permeability analysis pipeline")
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data")
            self.raw_data = self.data_handler.load_bbbp_data(data_path)
            data_summary = self.data_handler.generate_data_summary()
            logger.info(f"Loaded {len(self.raw_data)} compounds")
            
            # Step 2: Calculate molecular descriptors
            logger.info("Step 2: Calculating molecular descriptors")
            self.processed_data = self.descriptor_calculator.calculate_from_dataframe(
                self.raw_data, mol_column='mol'
            )
            logger.info(f"Calculated descriptors for {len(self.processed_data)} compounds")
            
            # Step 3: Feature engineering and preprocessing
            logger.info("Step 3: Feature engineering and preprocessing")
            descriptor_columns = self.descriptor_calculator.descriptor_names
            available_descriptors = [col for col in descriptor_columns if col in self.processed_data.columns]
            
            if not available_descriptors:
                raise ValueError("No molecular descriptors found in processed data")
            
            X = self.processed_data[available_descriptors]
            y = self.processed_data['p_np']
            
            # Create train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_handler.get_train_test_split()
            
            # Extract descriptor columns from train/test sets
            X_train_descriptors = self.X_train[available_descriptors]
            X_test_descriptors = self.X_test[available_descriptors]
            
            # Apply feature engineering
            X_train_processed = self.feature_engineer.create_feature_matrix(
                X_train_descriptors, y=self.y_train
            )
            X_test_processed = self.feature_engineer.create_feature_matrix(
                X_test_descriptors, fit=False
            )
            
            logger.info(f"Feature matrix shape: {X_train_processed.shape}")
            
            # Step 4: Train machine learning models
            logger.info("Step 4: Training machine learning models")
            self._train_all_models(X_train_processed, self.y_train)
            
            # Step 5: Evaluate models
            logger.info("Step 5: Evaluating model performance")
            self._evaluate_all_models(X_test_processed, self.y_test)
            
            # Step 6: Model comparison and selection
            logger.info("Step 6: Comparing models and selecting best performer")
            cv_results = self.model_trainer.cross_validate_models(X_train_processed, self.y_train)
            comparison_df = self.model_trainer.compare_models(cv_results)
            
            # Step 7: Interpretability analysis
            logger.info("Step 7: Performing interpretability analysis")
            self._perform_interpretability_analysis(X_train_processed, self.y_train)
            
            # Step 8: Generate visualizations
            logger.info("Step 8: Generating comprehensive visualizations")
            self._generate_visualizations()
            
            # Step 9: Generate comprehensive report
            logger.info("Step 9: Generating analysis report")
            self.analysis_report = self._generate_comprehensive_report(
                data_summary, comparison_df, cv_results
            )
            
            # Step 10: Save results if requested
            if save_results:
                logger.info("Step 10: Saving analysis results")
                self.save_analysis_results()
            
            logger.info("Complete analysis pipeline finished successfully")
            return self.analysis_report
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
    
    def _train_all_models(self, X_train, y_train):
        """Train all machine learning models."""
        logger.info("Training multiple ML algorithms")
        
        # Train different models
        models_to_train = [
            ('logistic_regression', self.model_trainer.train_logistic_regression),
            ('random_forest', self.model_trainer.train_random_forest),
            ('svm', self.model_trainer.train_svm),
            ('xgboost', self.model_trainer.train_xgboost),
            ('neural_network', self.model_trainer.train_neural_network)
        ]
        
        for model_name, train_func in models_to_train:
            try:
                logger.info(f"Training {model_name}")
                model = train_func(X_train, y_train)
                self.trained_models[model_name] = model
                logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
    
    def _evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test set."""
        logger.info("Evaluating all trained models")
        
        for model_name, model in self.trained_models.items():
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Store results for visualization
                self.model_results[model_name] = {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
                
                # Evaluate performance
                metrics = self.model_trainer.evaluate_model(model, X_test, y_test, model_name)
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
    
    def _perform_interpretability_analysis(self, X_train, y_train):
        """Perform interpretability analysis on the best model."""
        logger.info("Performing interpretability analysis")
        
        try:
            # Get the best model
            best_model = self.model_trainer.best_model
            if best_model is None:
                logger.warning("No best model selected, using random forest for interpretability")
                best_model = self.trained_models.get('random_forest')
            
            if best_model is not None:
                # Calculate feature importance
                self.feature_importance = self.interpretability_engine.calculate_feature_importance(
                    best_model, X_train.columns if hasattr(X_train, 'columns') else None
                )
                
                # Generate SHAP analysis
                shap_values = self.interpretability_engine.generate_shap_analysis(
                    best_model, X_train, sample_size=100
                )
                
                logger.info("Interpretability analysis completed")
            else:
                logger.warning("No model available for interpretability analysis")
                
        except Exception as e:
            logger.warning(f"Error in interpretability analysis: {str(e)}")
    
    def _generate_visualizations(self):
        """Generate all analysis visualizations."""
        logger.info("Generating comprehensive visualizations")
        
        try:
            # Model performance plots
            if self.model_results:
                self.visualization_suite.plot_model_performance(
                    self.model_results,
                    save_path=PATHS['plots'] / 'model_performance.png'
                )
            
            # Feature importance plot
            if self.feature_importance is not None:
                self.visualization_suite.plot_feature_importance(
                    self.feature_importance,
                    save_path=PATHS['plots'] / 'feature_importance.png'
                )
            
            # Chemical space visualization
            if self.processed_data is not None:
                descriptor_columns = self.descriptor_calculator.descriptor_names
                available_descriptors = [col for col in descriptor_columns if col in self.processed_data.columns]
                
                if available_descriptors:
                    X_viz = self.processed_data[available_descriptors].fillna(0)
                    y_viz = self.processed_data['p_np']
                    
                    self.visualization_suite.plot_chemical_space(
                        X_viz, y_viz,
                        save_path=PATHS['plots'] / 'chemical_space.png'
                    )
            
            # Descriptor distributions
            if self.processed_data is not None:
                self.visualization_suite.plot_descriptor_distributions(
                    self.processed_data,
                    save_path=PATHS['plots'] / 'descriptor_distributions.png'
                )
            
            # Create comprehensive dashboard
            if all([self.processed_data is not None, self.model_results, self.feature_importance is not None]):
                self.visualization_suite.create_analysis_dashboard(
                    self.processed_data, self.model_results, self.feature_importance,
                    save_path=PATHS['plots'] / 'analysis_dashboard.png'
                )
            
            logger.info("All visualizations generated successfully")
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {str(e)}")
    
    def _generate_comprehensive_report(self, data_summary, comparison_df, cv_results):
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_summary': data_summary,
            'model_performance': {},
            'best_model': self.model_trainer.best_model_name,
            'feature_importance_top10': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # Add model performance results
        if not comparison_df.empty:
            report['model_performance'] = comparison_df.to_dict()
        
        # Add feature importance
        if self.feature_importance is not None:
            report['feature_importance_top10'] = self.feature_importance.head(10).to_dict()
        
        # Generate key insights
        report['key_insights'] = self._generate_key_insights(data_summary, comparison_df)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_key_insights(self, data_summary, comparison_df):
        """Generate key insights from the analysis."""
        insights = []
        
        # Dataset insights
        if data_summary:
            total_compounds = data_summary.get('total_compounds', 0)
            permeable_pct = data_summary.get('permeable_percentage', 0)
            insights.append(f"Dataset contains {total_compounds} compounds with {permeable_pct:.1f}% BBB-permeable")
        
        # Model performance insights
        if not comparison_df.empty and 'roc_auc' in comparison_df.columns:
            best_model = comparison_df['roc_auc'].idxmax()
            best_auc = comparison_df.loc[best_model, 'roc_auc']
            insights.append(f"Best performing model: {best_model} with AUC-ROC of {best_auc:.3f}")
        
        # Feature importance insights
        if self.feature_importance is not None:
            top_feature = self.feature_importance.index[0]
            top_importance = self.feature_importance.iloc[0]
            insights.append(f"Most important predictor: {top_feature} (importance: {top_importance:.3f})")
        
        return insights
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis results."""
        recommendations = [
            "Use the best-performing model for BBB permeability predictions",
            "Focus on optimizing the top molecular descriptors identified by feature importance analysis",
            "Consider molecular weight, lipophilicity, and polar surface area as key design parameters",
            "Validate predictions with experimental data before making drug design decisions",
            "Monitor model performance on new data and retrain if necessary"
        ]
        
        return recommendations
    
    def save_analysis_results(self, output_dir=None):
        """
        Save all analysis results to files.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save results. If None, uses results directory from config.
            
        Returns
        -------
        dict
            Dictionary mapping result types to saved file paths.
        """
        if output_dir is None:
            output_dir = PATHS['results']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save processed data
            if self.processed_data is not None:
                data_path = output_dir / 'processed_bbbp_data.csv'
                self.processed_data.to_csv(data_path, index=False)
                saved_files['processed_data'] = data_path
                logger.info(f"Saved processed data to {data_path}")
            
            # Save trained models
            models_dir = output_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in self.trained_models.items():
                model_path = models_dir / f'{model_name}.joblib'
                joblib.dump(model, model_path)
                saved_files[f'model_{model_name}'] = model_path
                logger.info(f"Saved {model_name} model to {model_path}")
            
            # Save feature importance
            if self.feature_importance is not None:
                importance_path = output_dir / 'feature_importance.csv'
                self.feature_importance.to_csv(importance_path, header=['importance'])
                saved_files['feature_importance'] = importance_path
                logger.info(f"Saved feature importance to {importance_path}")
            
            # Save analysis report
            if self.analysis_report:
                report_path = output_dir / 'analysis_report.json'
                with open(report_path, 'w') as f:
                    json.dump(self.analysis_report, f, indent=2, default=str)
                saved_files['analysis_report'] = report_path
                logger.info(f"Saved analysis report to {report_path}")
            
            # Save visualizations
            viz_files = self.visualization_suite.save_all_figures(PATHS['plots'])
            saved_files.update(viz_files)
            
            logger.info(f"All analysis results saved to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise
    
    def load_and_predict(self, smiles_list, model_name=None):
        """
        Load trained model and make predictions on new molecules.
        
        Parameters
        ----------
        smiles_list : list
            List of SMILES strings for prediction.
        model_name : str, optional
            Name of the model to use. If None, uses the best model.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with SMILES, predictions, and probabilities.
        """
        logger.info(f"Making predictions for {len(smiles_list)} molecules")
        
        try:
            # Use best model if not specified
            if model_name is None:
                model = self.model_trainer.best_model
                model_name = self.model_trainer.best_model_name
            else:
                model = self.trained_models.get(model_name)
            
            if model is None:
                raise ValueError(f"Model {model_name} not found or not trained")
            
            # Create DataFrame from SMILES
            pred_data = pd.DataFrame({'smiles': smiles_list})
            
            # Validate SMILES and create molecule objects
            pred_data = self.data_handler.validate_smiles(pred_data)
            
            # Calculate descriptors
            pred_data_with_descriptors = self.descriptor_calculator.calculate_from_dataframe(
                pred_data, mol_column='mol'
            )
            
            # Extract descriptor columns
            descriptor_columns = self.descriptor_calculator.descriptor_names
            available_descriptors = [col for col in descriptor_columns if col in pred_data_with_descriptors.columns]
            
            X_pred = pred_data_with_descriptors[available_descriptors]
            
            # Apply same preprocessing as training data
            X_pred_processed = self.feature_engineer.create_feature_matrix(
                X_pred, handle_missing=True, scale=True, select=True, fit=False
            )
            
            # Make predictions
            predictions = model.predict(X_pred_processed)
            probabilities = model.predict_proba(X_pred_processed)[:, 1]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'smiles': smiles_list,
                'predicted_permeability': predictions,
                'permeability_probability': probabilities,
                'prediction_label': ['Permeable' if p == 1 else 'Non-Permeable' for p in predictions]
            })
            
            logger.info(f"Predictions completed using {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def generate_summary_report(self):
        """
        Generate a concise summary report of the analysis.
        
        Returns
        -------
        str
            Formatted summary report.
        """
        if not self.analysis_report:
            return "No analysis results available. Run complete analysis first."
        
        report_lines = [
            "=" * 60,
            "BBB PERMEABILITY PREDICTION ANALYSIS SUMMARY",
            "=" * 60,
            "",
            f"Analysis Date: {self.analysis_report.get('analysis_timestamp', 'Unknown')}",
            "",
            "DATASET SUMMARY:",
            f"- Total Compounds: {self.analysis_report.get('dataset_summary', {}).get('total_compounds', 'N/A')}",
            f"- BBB Permeable: {self.analysis_report.get('dataset_summary', {}).get('permeable_percentage', 'N/A'):.1f}%",
            "",
            "MODEL PERFORMANCE:",
            f"- Best Model: {self.analysis_report.get('best_model', 'N/A')}",
        ]
        
        # Add model performance details
        model_perf = self.analysis_report.get('model_performance', {})
        if model_perf:
            best_model = self.analysis_report.get('best_model')
            if best_model and best_model in model_perf:
                perf = model_perf[best_model]
                report_lines.extend([
                    f"- Accuracy: {perf.get('accuracy', 'N/A'):.3f}",
                    f"- AUC-ROC: {perf.get('roc_auc', 'N/A'):.3f}",
                    f"- Precision: {perf.get('precision', 'N/A'):.3f}",
                    f"- Recall: {perf.get('recall', 'N/A'):.3f}",
                ])
        
        report_lines.extend([
            "",
            "TOP PREDICTIVE FEATURES:",
        ])
        
        # Add top features
        top_features = self.analysis_report.get('feature_importance_top10', {})
        for i, (feature, importance) in enumerate(list(top_features.items())[:5], 1):
            report_lines.append(f"{i}. {feature}: {importance:.3f}")
        
        report_lines.extend([
            "",
            "KEY INSIGHTS:",
        ])
        
        # Add insights
        insights = self.analysis_report.get('key_insights', [])
        for insight in insights:
            report_lines.append(f"- {insight}")
        
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        # Add recommendations
        recommendations = self.analysis_report.get('recommendations', [])
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


def main():
    """Main function to run the complete analysis."""
    # Create main analysis instance
    analyzer = MainAnalysis()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary report
    print(analyzer.generate_summary_report())
    
    return results


if __name__ == "__main__":
    main()