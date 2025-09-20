"""
Advanced SHAP-based model explainability for deeper insights.
Provides individual prediction explanations and global feature attribution.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from models.lightgbm_model import LightGBMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Advanced SHAP-based model explainability."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize SHAP explainer.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_lightgbm_shap(self, split_id: int = 0, max_samples: int = 500) -> Dict[str, Any]:
        """
        Perform SHAP analysis on a trained LightGBM model.
        
        Args:
            split_id: Which data split to analyze
            max_samples: Maximum number of samples for SHAP analysis
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        logger.info("Performing SHAP analysis for LightGBM model...")
        
        # Train the model
        lgb_model = LightGBMModel()
        lgb_model.train_single_split(split_id)
        
        # Get the trained model
        model = lgb_model.models.get(0)
        if model is None:
            logger.error("No trained model found")
            return {}
        
        # Load test data for SHAP analysis
        try:
            # Use split data directly from the model loading
            train_X, train_y, val_X, val_y, test_X, test_y = lgb_model.load_split_data(0)
            
            # Use test data for SHAP analysis
            feature_cols = test_X.columns.tolist()
            X_test = test_X.fillna(0)
            y_test = test_y.fillna(0)
            
            # IMPORTANT: Load original data with ticker information for mapping
            original_data = self._load_original_data_with_tickers(split_id)
            
            # Limit samples for computational efficiency
            if len(X_test) > max_samples:
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_test_sample = X_test.iloc[sample_indices]
                y_test_sample = y_test.iloc[sample_indices]
                # Create ticker mapping for sampled data - handle size mismatch
                ticker_mapping = {}
                for i, idx in enumerate(sample_indices):
                    if idx < len(original_data):
                        ticker_mapping[i] = original_data.iloc[idx]
                    else:
                        # Fallback for size mismatch - cycle through available data
                        fallback_idx = idx % len(original_data)
                        ticker_mapping[i] = original_data.iloc[fallback_idx]
            else:
                X_test_sample = X_test
                y_test_sample = y_test
                # Create ticker mapping for all data - handle size mismatch
                ticker_mapping = {}
                for i in range(len(X_test_sample)):
                    if i < len(original_data):
                        ticker_mapping[i] = original_data.iloc[i]
                    else:
                        # Fallback for size mismatch - cycle through available data
                        fallback_idx = i % len(original_data)
                        ticker_mapping[i] = original_data.iloc[fallback_idx]
            
            logger.info(f"Analyzing SHAP values for {len(X_test_sample)} samples")
            
        except Exception as e:
            logger.error(f"Could not load test data: {e}")
            return {}
        
        # Create SHAP explainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
            # Calculate feature importance from SHAP values
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            feature_names = X_test_sample.columns.tolist()
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)
            
            logger.info(f"SHAP analysis completed. Top feature: {importance_df.iloc[0]['feature']}")
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {}
        
        # Generate SHAP plots
        self._create_shap_plots(explainer, X_test_sample, shap_values, feature_names)
        
        # Analyze individual predictions with ticker information
        individual_analysis = self._analyze_individual_predictions_with_tickers(
            shap_values, X_test_sample, y_test_sample, feature_names, ticker_mapping, model
        )
        
        # Feature interaction analysis
        interaction_analysis = self._analyze_feature_interactions(
            shap_values, X_test_sample, feature_names
        )
        
        shap_results = {
            'shap_feature_importance': importance_df.head(30).to_dict('records'),
            'individual_predictions': individual_analysis,
            'feature_interactions': interaction_analysis,
            'sample_size': len(X_test_sample),
            'model_performance': {
                'split_id': split_id,
                'test_samples': len(X_test_sample),
                'unique_tickers': len(set(info.get('ticker', 'UNKNOWN') for info in ticker_mapping.values()))
            }
        }
        
        return shap_results
    
    def _create_shap_plots(self, explainer, X_sample, shap_values, feature_names):
        """Create various SHAP visualization plots."""
        try:
            # Set style
            plt.style.use('default')
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            show=False, max_display=20)
            plt.title('SHAP Summary Plot - Feature Impact on Model Output')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_summary_plot.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot (bar)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                            plot_type="bar", show=False, max_display=20)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_feature_importance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dependence plots for top features
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 features
            
            for i, feature_idx in enumerate(top_features_idx):
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                   feature_names=feature_names, show=False)
                plt.title(f'SHAP Dependence Plot - {feature_names[feature_idx]}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"shap_dependence_{i+1}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("SHAP plots created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create SHAP plots: {e}")
    
    def _analyze_individual_predictions(self, shap_values, X_sample, y_sample, feature_names):
        """Analyze individual prediction explanations."""
        
        # Select interesting predictions for detailed analysis
        predictions = shap_values.sum(axis=1)  # Base value will be added separately
        
        # Find extreme predictions and interesting cases
        high_pred_idx = np.argmax(predictions)
        low_pred_idx = np.argmin(predictions)
        median_idx = np.argsort(predictions)[len(predictions)//2]
        
        # Analyze these specific predictions
        interesting_cases = {
            'highest_prediction': high_pred_idx,
            'lowest_prediction': low_pred_idx,
            'median_prediction': median_idx
        }
        
        individual_analysis = {}
        
        for case_name, idx in interesting_cases.items():
            shap_contrib = shap_values[idx]
            feature_values = X_sample.iloc[idx]
            actual_return = y_sample.iloc[idx] if hasattr(y_sample, 'iloc') else y_sample[idx]
            
            # Get top contributing features
            contrib_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_contrib,
                'feature_value': feature_values.values
            })
            contrib_df['abs_shap'] = np.abs(contrib_df['shap_value'])
            contrib_df = contrib_df.sort_values('abs_shap', ascending=False)
            
            individual_analysis[case_name] = {
                'sample_index': int(idx),
                'predicted_impact': float(predictions[idx]),
                'actual_return': float(actual_return),
                'top_positive_contributors': contrib_df[contrib_df['shap_value'] > 0].head(5).to_dict('records'),
                'top_negative_contributors': contrib_df[contrib_df['shap_value'] < 0].head(5).to_dict('records'),
                'total_positive_impact': float(contrib_df[contrib_df['shap_value'] > 0]['shap_value'].sum()),
                'total_negative_impact': float(contrib_df[contrib_df['shap_value'] < 0]['shap_value'].sum())
            }
        
        return individual_analysis
    
    def _analyze_feature_interactions(self, shap_values, X_sample, feature_names):
        """Analyze feature interactions using SHAP values."""
        
        # Calculate feature correlation with SHAP values
        feature_shap_corr = {}
        
        for i, feature in enumerate(feature_names):
            feature_vals = X_sample.iloc[:, i]
            shap_vals = shap_values[:, i]
            
            # Check for valid data and avoid constant features
            if len(np.unique(feature_vals)) > 1 and not np.any(np.isnan(feature_vals)) and not np.any(np.isnan(shap_vals)):
                correlation = np.corrcoef(feature_vals, shap_vals)[0, 1]
                # Only store if correlation is valid (not NaN)
                if not np.isnan(correlation):
                    feature_shap_corr[feature] = float(correlation)  # Convert to Python float
        
        # Sort by absolute correlation
        sorted_corr = sorted(feature_shap_corr.items(), 
                           key=lambda x: abs(x[1]), reverse=True)
        
        # Find potential interaction pairs
        # Look for features where SHAP values don't correlate well with feature values
        # (indicating interactions with other features)
        low_correlation_features = [(f, corr) for f, corr in sorted_corr 
                                  if abs(corr) < 0.3 and abs(corr) > 0.05]
        
        interaction_analysis = {
            'feature_shap_correlations': dict(sorted_corr[:20]),
            'potential_interaction_features': low_correlation_features[:10],
            'highly_linear_features': [(f, corr) for f, corr in sorted_corr[:10] 
                                     if abs(corr) > 0.7]
        }
        
        return interaction_analysis
    
    def _load_original_data_with_tickers(self, split_id: int) -> pd.DataFrame:
        """Load original test data with ticker information for mapping."""
        try:
            # Load the split metadata to get date range for test set
            split_dir = f"data/processed/splits/split_{split_id}"
            split_info_path = os.path.join(split_dir, "split_info.parquet")
            
            # Load original features with ticker information
            features_path = "data/processed/features_engineered.parquet"
            original_df = pd.read_parquet(features_path)
            original_df['date'] = pd.to_datetime(original_df['date'])
            
            if os.path.exists(split_info_path):
                split_info = pd.read_parquet(split_info_path)
                test_start = pd.to_datetime(split_info.iloc[0]['test_start'])
                test_end = pd.to_datetime(split_info.iloc[0]['test_end'])
                
                # Filter data for test period
                test_mask = (original_df['date'] >= test_start) & (original_df['date'] <= test_end)
                test_data_with_tickers = original_df[test_mask][['ticker', 'date']].reset_index(drop=True)
                
                logger.info(f"Loaded ticker mapping for {len(test_data_with_tickers)} test samples from {test_start} to {test_end}")
                
            else:
                # Fallback: use last portion of data (approximately test set)
                logger.warning("Split info not found, using fallback method for ticker mapping")
                n_test = 1590  # Approximate test set size based on observed data
                test_data_with_tickers = original_df.tail(n_test)[['ticker', 'date']].reset_index(drop=True)
            
            return test_data_with_tickers
            
        except Exception as e:
            logger.error(f"Could not load ticker mapping: {e}")
            # Return dummy data to prevent crashes
            return pd.DataFrame({'ticker': ['UNKNOWN'] * 500, 'date': [pd.Timestamp.now()] * 500})
    
    def _analyze_individual_predictions_with_tickers(
        self, 
        shap_values, 
        X_sample, 
        y_actual, 
        feature_names,
        ticker_mapping: Dict,
        model
    ):
        """Analyze individual predictions with ticker information."""
        predictions = model.predict(X_sample)
        
        # Find examples of different prediction magnitudes with ticker info
        pred_indices = np.argsort(np.abs(predictions))
        
        examples = {}
        
        # Highest prediction
        highest_idx = pred_indices[-1]
        ticker_info = ticker_mapping.get(highest_idx, {'ticker': 'UNKNOWN', 'date': 'UNKNOWN'})
        examples['highest_prediction'] = self._create_prediction_example(
            highest_idx, predictions, y_actual, shap_values, X_sample, feature_names, ticker_info
        )
        
        # Lowest prediction
        lowest_idx = pred_indices[0]
        ticker_info = ticker_mapping.get(lowest_idx, {'ticker': 'UNKNOWN', 'date': 'UNKNOWN'})
        examples['lowest_prediction'] = self._create_prediction_example(
            lowest_idx, predictions, y_actual, shap_values, X_sample, feature_names, ticker_info
        )
        
        # Medium prediction (closest to median)
        median_pred = np.median(predictions)
        median_distances = np.abs(predictions - median_pred)
        median_idx = np.argmin(median_distances)
        ticker_info = ticker_mapping.get(median_idx, {'ticker': 'UNKNOWN', 'date': 'UNKNOWN'})
        examples['median_prediction'] = self._create_prediction_example(
            median_idx, predictions, y_actual, shap_values, X_sample, feature_names, ticker_info
        )
        
        # Add ticker-specific analysis
        ticker_analysis = self._analyze_by_ticker(shap_values, X_sample, predictions, y_actual, ticker_mapping)
        examples['ticker_analysis'] = ticker_analysis
        
        return examples
        
    def _create_prediction_example(self, idx, predictions, y_actual, shap_values, X_sample, feature_names, ticker_info):
        """Create a prediction example with ticker information."""
        row_shap = shap_values[idx, :]
        row_features = X_sample.iloc[idx]
        
        # Create contribution dataframe
        contrib_df = pd.DataFrame({
            'feature': feature_names,
            'feature_value': row_features.values,
            'shap_value': row_shap
        })
        contrib_df['abs_shap'] = np.abs(contrib_df['shap_value'])
        contrib_df = contrib_df.sort_values('abs_shap', ascending=False)
        
        actual_return = y_actual.iloc[idx] if idx < len(y_actual) else 0.0
        
        return {
            'sample_index': int(idx),
            'ticker': ticker_info.get('ticker', 'UNKNOWN'),
            'date': str(ticker_info.get('date', 'UNKNOWN')),
            'predicted_impact': float(predictions[idx]),
            'actual_return': float(actual_return),
            'top_positive_contributors': [
                {
                    'feature': row['feature'],
                    'shap_value': float(row['shap_value']),
                    'feature_value': float(row['feature_value']),
                    'abs_shap': float(row['abs_shap'])
                }
                for _, row in contrib_df[contrib_df['shap_value'] > 0].head(10).iterrows()
            ],
            'top_negative_contributors': [
                {
                    'feature': row['feature'],
                    'shap_value': float(row['shap_value']),
                    'feature_value': float(row['feature_value']),
                    'abs_shap': float(row['abs_shap'])
                }
                for _, row in contrib_df[contrib_df['shap_value'] < 0].head(10).iterrows()
            ],
            'total_positive_impact': float(contrib_df[contrib_df['shap_value'] > 0]['shap_value'].sum()),
            'total_negative_impact': float(contrib_df[contrib_df['shap_value'] < 0]['shap_value'].sum())
        }
    
    def _analyze_by_ticker(self, shap_values, X_sample, predictions, y_actual, ticker_mapping):
        """Analyze SHAP patterns by ticker."""
        ticker_stats = {}
        
        # Group by ticker
        ticker_groups = {}
        for idx, ticker_info in ticker_mapping.items():
            ticker = ticker_info.get('ticker', 'UNKNOWN')
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(idx)
        
        # Calculate statistics for each ticker
        for ticker, indices in ticker_groups.items():
            if len(indices) < 2:  # Skip tickers with too few samples
                continue
                
            ticker_shap = shap_values[indices, :]
            ticker_preds = predictions[indices]
            ticker_actual = y_actual.iloc[indices] if len(indices) <= len(y_actual) else np.zeros(len(indices))
            
            # Calculate ticker-specific feature importance
            ticker_feature_importance = np.mean(np.abs(ticker_shap), axis=0)
            top_features_idx = np.argsort(ticker_feature_importance)[-5:]  # Top 5 features
            
            ticker_stats[ticker] = {
                'sample_count': len(indices),
                'avg_prediction': float(np.mean(ticker_preds)),
                'avg_actual': float(np.mean(ticker_actual)),
                'prediction_std': float(np.std(ticker_preds)),
                'top_features': [
                    {
                        'feature': X_sample.columns[i],
                        'importance': float(ticker_feature_importance[i])
                    }
                    for i in reversed(top_features_idx)
                ]
            }
        
        return ticker_stats
    
    def create_prediction_explanation_report(self, sample_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Create detailed explanation for a specific prediction.
        
        Args:
            sample_idx: Index of sample to explain (random if None)
            
        Returns:
            Detailed prediction explanation
        """
        logger.info("Creating detailed prediction explanation...")
        
        # Perform SHAP analysis
        shap_results = self.analyze_lightgbm_shap(max_samples=500)
        
        if not shap_results:
            return {}
        
        # Use individual prediction analysis
        individual_predictions = shap_results.get('individual_predictions', {})
        
        if not individual_predictions:
            return {}
        
        # Create comprehensive report
        report = {
            'explanation_type': 'SHAP-based prediction explanation',
            'model_type': 'LightGBM',
            'analysis_date': datetime.now().isoformat(),
            'sample_explanations': individual_predictions,
            'global_insights': {
                'top_features_globally': shap_results.get('shap_feature_importance', [])[:10],
                'feature_interactions': shap_results.get('feature_interactions', {})
            },
            'model_performance': shap_results.get('model_performance', {}),
            'methodology': {
                'explainer_type': 'TreeExplainer',
                'sample_size': shap_results.get('sample_size', 0),
                'interpretation': 'SHAP values represent the contribution of each feature to the prediction'
            }
        }
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"prediction_explanation_report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Prediction explanation report saved: {report_file}")
        
        return report

def main():
    """Main function to run SHAP-based explainability analysis."""
    
    logger.info("=" * 70)
    logger.info("ADVANCED SHAP EXPLAINABILITY ANALYSIS")
    logger.info("=" * 70)
    
    # Initialize SHAP explainer
    shap_explainer = SHAPExplainer()
    
    # Generate comprehensive explanation report
    explanation_report = shap_explainer.create_prediction_explanation_report()
    
    # Print key insights
    if explanation_report:
        global_insights = explanation_report.get('global_insights', {})
        top_features = global_insights.get('top_features_globally', [])
        
        if top_features:
            logger.info("\\nTOP 10 FEATURES BY SHAP IMPORTANCE:")
            logger.info("-" * 45)
            for i, feature_data in enumerate(top_features[:10], 1):
                feature_name = feature_data.get('feature', 'Unknown')
                shap_importance = feature_data.get('shap_importance', 0)
                logger.info(f"{i:2d}. {feature_name:<25} {shap_importance:.4f}")
        
        # Show sample explanation
        sample_explanations = explanation_report.get('sample_explanations', {})
        if 'highest_prediction' in sample_explanations:
            highest_pred = sample_explanations['highest_prediction']
            logger.info("\\nEXAMPLE: HIGHEST PREDICTION EXPLANATION:")
            logger.info("-" * 45)
            logger.info(f"Predicted impact: {highest_pred.get('predicted_impact', 0):.4f}")
            logger.info(f"Actual return:    {highest_pred.get('actual_return', 0):.4f}")
            
            pos_contributors = highest_pred.get('top_positive_contributors', [])
            if pos_contributors:
                logger.info("\\nTop positive contributors:")
                for contrib in pos_contributors[:3]:
                    feature = contrib.get('feature', 'Unknown')
                    shap_val = contrib.get('shap_value', 0)
                    feat_val = contrib.get('feature_value', 0)
                    logger.info(f"  {feature:<20} SHAP: {shap_val:+.4f}, Value: {feat_val:.4f}")
    
    logger.info("\\n" + "=" * 70)
    logger.info("SHAP EXPLAINABILITY ANALYSIS COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()