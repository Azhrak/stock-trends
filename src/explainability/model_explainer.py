"""
Model explainability and validation tools.
Provides feature importance analysis, SHAP explanations, and model interpretation.
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

from models.lightgbm_model import LightGBMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Comprehensive model explainability toolkit."""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "reports"):
        """
        Initialize model explainer.
        
        Args:
            models_dir: Directory containing trained models
            output_dir: Directory for output reports
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_lightgbm_importance(self, num_splits: int = 5) -> Dict[str, Any]:
        """
        Analyze LightGBM feature importance across multiple splits.
        
        Args:
            num_splits: Number of splits to analyze
            
        Returns:
            Feature importance analysis results
        """
        logger.info("Analyzing LightGBM feature importance...")
        
        # Initialize and train LightGBM model
        lgb_model = LightGBMModel()
        
        # Train on multiple splits to get robust importance estimates
        results = lgb_model.train_all_splits(max_splits=num_splits, save_results=False)
        
        if not results:
            logger.error("No LightGBM results obtained")
            return {}
        
        # Get feature importance across all splits
        feature_importance_df = lgb_model.get_feature_importance(top_n=50)
        
        # Analyze feature categories
        importance_analysis = self._categorize_feature_importance(feature_importance_df)
        
        # Create visualizations
        self._plot_feature_importance(feature_importance_df, "lightgbm_feature_importance.png")
        self._plot_importance_by_category(importance_analysis, "lightgbm_importance_by_category.png")
        
        # Stability analysis
        stability_analysis = self._analyze_importance_stability(lgb_model.models)
        
        results = {
            'feature_importance': feature_importance_df.to_dict('records'),
            'category_analysis': importance_analysis,
            'stability_analysis': stability_analysis,
            'top_features': feature_importance_df.head(20)['feature'].tolist(),
            'model_performance': lgb_model.calculate_summary_metrics(results)
        }
        
        logger.info(f"Feature importance analysis completed. Top feature: {feature_importance_df.iloc[0]['feature']}")
        
        return results
    
    def _categorize_feature_importance(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize features by type and analyze importance by category."""
        
        # Define feature categories based on naming patterns
        categories = {
            'price': ['open', 'high', 'low', 'close', 'adj_close'],
            'volume': ['volume'],
            'technical': ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'atr_', 'adx_', 'cci_', 'williams_', 'stoch_'],
            'volatility': ['volatility_', 'vol_', 'realized_vol'],
            'momentum': ['momentum_', 'roc_', 'mom_'],
            'lagged': ['lag_', 'prev_'],
            'returns': ['return_', 'ret_'],
            'other': []
        }
        
        # Categorize each feature
        feature_categories = {}
        category_importance = {cat: [] for cat in categories.keys()}
        
        for _, row in feature_df.iterrows():
            feature = row['feature']
            importance = row['mean_importance']
            
            categorized = False
            for category, patterns in categories.items():
                if category == 'other':
                    continue
                    
                if any(pattern in feature.lower() for pattern in patterns):
                    feature_categories[feature] = category
                    category_importance[category].append(importance)
                    categorized = True
                    break
            
            if not categorized:
                feature_categories[feature] = 'other'
                category_importance['other'].append(importance)
        
        # Calculate category statistics
        category_stats = {}
        for category, importances in category_importance.items():
            if importances:
                category_stats[category] = {
                    'mean_importance': np.mean(importances),
                    'total_importance': np.sum(importances),
                    'feature_count': len(importances),
                    'top_features': [f for f, c in feature_categories.items() if c == category][:5]
                }
        
        return {
            'feature_categories': feature_categories,
            'category_stats': category_stats
        }
    
    def _analyze_importance_stability(self, models: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze stability of feature importance across splits."""
        
        if len(models) < 2:
            return {'stability_score': 0, 'message': 'Not enough models for stability analysis'}
        
        # Collect feature importance from all models
        all_importances = {}
        feature_names = None
        
        for split_id, model in models.items():
            if hasattr(model, 'feature_importance'):
                importance_values = model.feature_importance()
                if feature_names is None:
                    # Assume feature names are stored in the model or can be retrieved
                    feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                
                for i, importance in enumerate(importance_values):
                    feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    if feature_name not in all_importances:
                        all_importances[feature_name] = []
                    all_importances[feature_name].append(importance)
        
        if not all_importances:
            return {'stability_score': 0, 'message': 'No importance data available'}
        
        # Calculate stability metrics
        stability_scores = {}
        for feature, importances in all_importances.items():
            if len(importances) > 1:
                cv = np.std(importances) / np.mean(importances) if np.mean(importances) > 0 else 0
                stability_scores[feature] = 1 / (1 + cv)  # Higher score = more stable
        
        # Overall stability score
        overall_stability = np.mean(list(stability_scores.values())) if stability_scores else 0
        
        # Find most/least stable features
        if stability_scores:
            sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            most_stable = sorted_stability[:10]
            least_stable = sorted_stability[-10:]
        else:
            most_stable = []
            least_stable = []
        
        return {
            'overall_stability': overall_stability,
            'feature_stability': stability_scores,
            'most_stable_features': most_stable,
            'least_stable_features': least_stable
        }
    
    def _plot_feature_importance(self, feature_df: pd.DataFrame, filename: str):
        """Plot feature importance."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Top 20 features
            top_features = feature_df.head(20)
            
            plt.barh(range(len(top_features)), top_features['mean_importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Mean Importance')
            plt.title('Top 20 Feature Importance (LightGBM)')
            plt.gca().invert_yaxis()
            
            # Add error bars
            plt.errorbar(top_features['mean_importance'], range(len(top_features)), 
                        xerr=top_features['std_importance'], fmt='none', color='red', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved: {filename}")
            
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")
    
    def _plot_importance_by_category(self, analysis: Dict[str, Any], filename: str):
        """Plot importance by feature category."""
        try:
            category_stats = analysis['category_stats']
            
            if not category_stats:
                logger.warning("No category statistics available for plotting")
                return
            
            categories = list(category_stats.keys())
            mean_importances = [category_stats[cat]['mean_importance'] for cat in categories]
            total_importances = [category_stats[cat]['total_importance'] for cat in categories]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Mean importance by category
            ax1.bar(categories, mean_importances)
            ax1.set_title('Mean Feature Importance by Category')
            ax1.set_ylabel('Mean Importance')
            ax1.tick_params(axis='x', rotation=45)
            
            # Total importance by category
            ax2.bar(categories, total_importances)
            ax2.set_title('Total Feature Importance by Category')
            ax2.set_ylabel('Total Importance')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Category importance plot saved: {filename}")
            
        except Exception as e:
            logger.warning(f"Could not create category plot: {e}")
    
    def analyze_prediction_patterns(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze prediction patterns and model behavior.
        
        Args:
            model_results: Model training results
            
        Returns:
            Prediction pattern analysis
        """
        logger.info("Analyzing prediction patterns...")
        
        detailed_results = model_results.get('detailed_results', [])
        
        if not detailed_results:
            logger.error("No detailed results available")
            return {}
        
        # Collect all predictions and actuals
        all_predictions = []
        all_actuals = []
        split_info = []
        
        for split_result in detailed_results:
            split_id = split_result['split_id']
            
            # Get test predictions and actuals
            test_pred = split_result.get('predictions', {}).get('test', [])
            test_actual = split_result.get('actuals', {}).get('test', [])
            
            if test_pred and test_actual and len(test_pred) == len(test_actual):
                all_predictions.extend(test_pred)
                all_actuals.extend(test_actual)
                split_info.extend([split_id] * len(test_pred))
        
        if not all_predictions:
            logger.error("No valid predictions found")
            return {}
        
        pred_array = np.array(all_predictions)
        actual_array = np.array(all_actuals)
        
        # Basic statistics
        correlation = np.corrcoef(pred_array, actual_array)[0, 1]
        
        # Prediction distribution analysis
        pred_stats = {
            'mean': float(np.mean(pred_array)),
            'std': float(np.std(pred_array)),
            'min': float(np.min(pred_array)),
            'max': float(np.max(pred_array)),
            'skewness': float(self._calculate_skewness(pred_array)),
            'kurtosis': float(self._calculate_kurtosis(pred_array))
        }
        
        actual_stats = {
            'mean': float(np.mean(actual_array)),
            'std': float(np.std(actual_array)),
            'min': float(np.min(actual_array)),
            'max': float(np.max(actual_array)),
            'skewness': float(self._calculate_skewness(actual_array)),
            'kurtosis': float(self._calculate_kurtosis(actual_array))
        }
        
        # Directional accuracy analysis
        pred_directions = pred_array > 0
        actual_directions = actual_array > 0
        directional_accuracy = np.mean(pred_directions == actual_directions)
        
        # Confidence-based analysis
        confidence_analysis = self._analyze_prediction_confidence(pred_array, actual_array)
        
        # Error analysis
        errors = pred_array - actual_array
        error_analysis = {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'bias': float(np.mean(errors)),
            'error_std': float(np.std(errors))
        }
        
        # Create prediction plots
        self._plot_prediction_analysis(pred_array, actual_array, "prediction_analysis.png")
        
        results = {
            'correlation': float(correlation),
            'directional_accuracy': float(directional_accuracy),
            'prediction_stats': pred_stats,
            'actual_stats': actual_stats,
            'error_analysis': error_analysis,
            'confidence_analysis': confidence_analysis,
            'sample_size': len(all_predictions)
        }
        
        logger.info(f"Prediction analysis completed. Correlation: {correlation:.3f}, "
                   f"Directional accuracy: {directional_accuracy:.3f}")
        
        return results
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _analyze_prediction_confidence(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction performance by confidence level."""
        
        # Use prediction magnitude as proxy for confidence
        pred_abs = np.abs(predictions)
        
        # Define confidence quartiles
        quartiles = np.percentile(pred_abs, [25, 50, 75])
        
        confidence_levels = []
        for pred_mag in pred_abs:
            if pred_mag <= quartiles[0]:
                confidence_levels.append('low')
            elif pred_mag <= quartiles[1]:
                confidence_levels.append('medium_low')
            elif pred_mag <= quartiles[2]:
                confidence_levels.append('medium_high')
            else:
                confidence_levels.append('high')
        
        # Analyze performance by confidence level
        performance_by_confidence = {}
        
        for level in ['low', 'medium_low', 'medium_high', 'high']:
            mask = np.array(confidence_levels) == level
            
            if np.sum(mask) == 0:
                continue
            
            level_preds = predictions[mask]
            level_actuals = actuals[mask]
            
            level_mae = np.mean(np.abs(level_preds - level_actuals))
            level_corr = np.corrcoef(level_preds, level_actuals)[0, 1] if len(level_preds) > 1 else 0
            level_dir_acc = np.mean((level_preds > 0) == (level_actuals > 0))
            
            performance_by_confidence[level] = {
                'count': int(np.sum(mask)),
                'mae': float(level_mae),
                'correlation': float(level_corr),
                'directional_accuracy': float(level_dir_acc),
                'pred_range': [float(np.min(pred_abs[mask])), float(np.max(pred_abs[mask]))]
            }
        
        return performance_by_confidence
    
    def _plot_prediction_analysis(self, predictions: np.ndarray, actuals: np.ndarray, filename: str):
        """Create comprehensive prediction analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Scatter plot
            axes[0, 0].scatter(actuals, predictions, alpha=0.5)
            axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
            axes[0, 0].set_xlabel('Actual Returns')
            axes[0, 0].set_ylabel('Predicted Returns')
            axes[0, 0].set_title('Predictions vs Actuals')
            
            # Residuals plot
            residuals = predictions - actuals
            axes[0, 1].scatter(predictions, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Returns')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predictions')
            
            # Distribution comparison
            axes[1, 0].hist(actuals, bins=50, alpha=0.7, label='Actual', density=True)
            axes[1, 0].hist(predictions, bins=50, alpha=0.7, label='Predicted', density=True)
            axes[1, 0].set_xlabel('Returns')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Distribution Comparison')
            axes[1, 0].legend()
            
            # Cumulative returns simulation
            cum_actual = np.cumsum(actuals[:1000])  # First 1000 predictions
            cum_predicted = np.cumsum(predictions[:1000])
            axes[1, 1].plot(cum_actual, label='Actual Cumulative')
            axes[1, 1].plot(cum_predicted, label='Predicted Cumulative')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Cumulative Returns')
            axes[1, 1].set_title('Cumulative Returns Comparison')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prediction analysis plot saved: {filename}")
            
        except Exception as e:
            logger.warning(f"Could not create prediction plots: {e}")
    
    def generate_model_report(self, model_type: str = 'lightgbm') -> Dict[str, Any]:
        """
        Generate comprehensive model explainability report.
        
        Args:
            model_type: Type of model to analyze ('lightgbm', 'transformer')
            
        Returns:
            Comprehensive model report
        """
        logger.info(f"Generating comprehensive report for {model_type} model...")
        
        report = {
            'model_type': model_type,
            'analysis_date': datetime.now().isoformat(),
            'sections': {}
        }
        
        if model_type == 'lightgbm':
            # Feature importance analysis
            importance_analysis = self.analyze_lightgbm_importance()
            report['sections']['feature_importance'] = importance_analysis
            
            # Load model results for prediction analysis
            try:
                results_files = [f for f in os.listdir(self.models_dir) 
                               if f.startswith('lightgbm_results_') and f.endswith('.json')]
                
                if results_files:
                    latest_file = os.path.join(self.models_dir, sorted(results_files)[-1])
                    with open(latest_file, 'r') as f:
                        model_results = json.load(f)
                    
                    # Prediction pattern analysis
                    prediction_analysis = self.analyze_prediction_patterns(model_results)
                    report['sections']['prediction_patterns'] = prediction_analysis
                    
                    # Model performance summary
                    report['sections']['performance_summary'] = model_results.get('summary', {})
                
            except Exception as e:
                logger.error(f"Could not load model results: {e}")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"{model_type}_explainability_report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Model explainability report saved: {report_file}")
        
        return report

def main():
    """Main function to run model explainability analysis."""
    
    logger.info("=" * 70)
    logger.info("MODEL EXPLAINABILITY ANALYSIS")
    logger.info("=" * 70)
    
    # Initialize explainer
    explainer = ModelExplainer()
    
    # Generate LightGBM report
    lgb_report = explainer.generate_model_report('lightgbm')
    
    # Print key insights
    if 'feature_importance' in lgb_report['sections']:
        importance_section = lgb_report['sections']['feature_importance']
        top_features = importance_section.get('top_features', [])
        
        logger.info("\\nTOP 10 MOST IMPORTANT FEATURES:")
        logger.info("-" * 40)
        for i, feature in enumerate(top_features[:10], 1):
            logger.info(f"{i:2d}. {feature}")
    
    if 'prediction_patterns' in lgb_report['sections']:
        pattern_section = lgb_report['sections']['prediction_patterns']
        correlation = pattern_section.get('correlation', 0)
        dir_accuracy = pattern_section.get('directional_accuracy', 0)
        
        logger.info("\\nPREDICTION QUALITY:")
        logger.info("-" * 40)
        logger.info(f"Correlation with actuals: {correlation:.3f}")
        logger.info(f"Directional accuracy:     {dir_accuracy:.3f}")
    
    logger.info("\\n" + "=" * 70)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()