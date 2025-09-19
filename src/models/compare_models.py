"""
Compare performance of LightGBM and Temporal Transformer models.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_results(models_dir: str = "models") -> dict:
    """Load the latest results for each model type."""
    
    results = {}
    
    # Find LightGBM results
    lgb_files = [f for f in os.listdir(models_dir) if f.startswith('lightgbm_results_') and f.endswith('.json')]
    if lgb_files:
        lgb_file = os.path.join(models_dir, sorted(lgb_files)[-1])
        with open(lgb_file, 'r') as f:
            results['lightgbm'] = json.load(f)
        logger.info(f"Loaded LightGBM results from: {lgb_file}")
    
    # Find Transformer results
    transformer_files = [f for f in os.listdir(models_dir) if f.startswith('transformer_results_') and f.endswith('.json')]
    if transformer_files:
        transformer_file = os.path.join(models_dir, sorted(transformer_files)[-1])
        with open(transformer_file, 'r') as f:
            transformer_data = json.load(f)
        
        # Check if it's the old format (list) or new format (dict)
        if isinstance(transformer_data, list):
            # Convert old format to new format
            results['transformer'] = {
                'detailed_results': transformer_data,
                'summary': calculate_summary_from_detailed(transformer_data),
                'training_config': {'splits_trained': len(transformer_data), 'model_type': 'transformer'}
            }
        else:
            results['transformer'] = transformer_data
        
        logger.info(f"Loaded Transformer results from: {transformer_file}")
    
    return results

def calculate_summary_from_detailed(detailed_results: list) -> dict:
    """Calculate summary statistics from detailed results."""
    metrics = ['test_rmse', 'test_dir_accuracy', 'val_rmse', 'val_dir_accuracy', 'train_rmse']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in detailed_results if metric in r]
        if values:
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
    
    return summary

def compare_model_performance(results: dict):
    """Compare performance metrics between models."""
    
    logger.info("=" * 70)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 70)
    
    if 'lightgbm' not in results or 'transformer' not in results:
        logger.error("Both model results are required for comparison")
        return
    
    # Extract summaries
    lgb_summary = results['lightgbm'].get('summary', {})
    transformer_summary = results['transformer'].get('summary', {})
    
    # Define metrics to compare
    metrics = [
        ('test_rmse', 'Test RMSE', 'lower'),
        ('test_dir_accuracy', 'Test Direction Accuracy', 'higher'),
        ('val_rmse', 'Validation RMSE', 'lower'),
        ('val_dir_accuracy', 'Val Direction Accuracy', 'higher'),
    ]
    
    # Create comparison table
    logger.info(f"{'Metric':<25} {'LightGBM':<15} {'Transformer':<15} {'Winner':<12}")
    logger.info("-" * 70)
    
    for metric, display_name, direction in metrics:
        lgb_mean = lgb_summary.get(f'mean_{metric}', 0)
        lgb_std = lgb_summary.get(f'std_{metric}', 0)
        
        transformer_mean = transformer_summary.get(f'mean_{metric}', 0)
        transformer_std = transformer_summary.get(f'std_{metric}', 0)
        
        if lgb_mean == 0 and transformer_mean == 0:
            continue
        
        # Determine winner
        if direction == 'lower':
            winner = "LightGBM" if lgb_mean < transformer_mean else "Transformer"
            improvement = abs(lgb_mean - transformer_mean) / max(lgb_mean, transformer_mean) * 100
        else:
            winner = "LightGBM" if lgb_mean > transformer_mean else "Transformer"
            improvement = abs(lgb_mean - transformer_mean) / max(lgb_mean, transformer_mean) * 100
        
        lgb_str = f"{lgb_mean:.4f}±{lgb_std:.4f}"
        transformer_str = f"{transformer_mean:.4f}±{transformer_std:.4f}"
        winner_str = f"{winner} ({improvement:.1f}%)"
        
        logger.info(f"{display_name:<25} {lgb_str:<15} {transformer_str:<15} {winner_str:<12}")
    
    # Model characteristics comparison
    logger.info("\\n" + "=" * 70)
    logger.info("MODEL CHARACTERISTICS")
    logger.info("=" * 70)
    
    lgb_config = results['lightgbm'].get('training_config', {})
    transformer_config = results['transformer'].get('training_config', {})
    
    logger.info(f"{'Characteristic':<25} {'LightGBM':<15} {'Transformer':<15}")
    logger.info("-" * 55)
    logger.info(f"{'Splits Trained':<25} {lgb_config.get('splits_trained', 'N/A'):<15} {transformer_config.get('splits_trained', 'N/A'):<15}")
    logger.info(f"{'Model Type':<25} {'Gradient Boost':<15} {'Transformer':<15}")
    logger.info(f"{'Features Used':<25} {'Tabular':<15} {'Sequential':<15}")

def analyze_prediction_patterns(results: dict):
    """Analyze prediction patterns from detailed results."""
    
    logger.info("\\n" + "=" * 70)
    logger.info("PREDICTION ANALYSIS")
    logger.info("=" * 70)
    
    for model_name, model_results in results.items():
        detailed_results = model_results.get('detailed_results', [])
        
        if not detailed_results:
            continue
        
        logger.info(f"\\n{model_name.upper()} ANALYSIS:")
        
        # Collect all predictions and actuals
        all_predictions = []
        all_actuals = []
        
        for split_result in detailed_results:
            predictions = split_result.get('predictions', {})
            actuals = split_result.get('actuals', {})
            
            test_pred = predictions.get('test', [])
            test_actual = actuals.get('test', [])
            
            all_predictions.extend(test_pred)
            all_actuals.extend(test_actual)
        
        if all_predictions and all_actuals:
            pred_array = np.array(all_predictions)
            actual_array = np.array(all_actuals)
            
            # Calculate correlation
            correlation = np.corrcoef(pred_array, actual_array)[0, 1]
            
            # Calculate prediction statistics
            pred_mean = np.mean(pred_array)
            pred_std = np.std(pred_array)
            actual_mean = np.mean(actual_array)
            actual_std = np.std(actual_array)
            
            logger.info(f"  Prediction-Actual Correlation: {correlation:.4f}")
            logger.info(f"  Prediction Mean/Std:           {pred_mean:.4f} / {pred_std:.4f}")
            logger.info(f"  Actual Mean/Std:               {actual_mean:.4f} / {actual_std:.4f}")
            
            # Directional accuracy
            pred_direction = pred_array > 0
            actual_direction = actual_array > 0
            dir_accuracy = np.mean(pred_direction == actual_direction)
            
            logger.info(f"  Overall Direction Accuracy:    {dir_accuracy:.3f}")
            
            # Prediction extremes
            top_10_pct_threshold = np.percentile(np.abs(pred_array), 90)
            high_confidence_mask = np.abs(pred_array) >= top_10_pct_threshold
            
            if np.sum(high_confidence_mask) > 0:
                high_conf_dir_acc = np.mean(
                    pred_direction[high_confidence_mask] == actual_direction[high_confidence_mask]
                )
                logger.info(f"  High Confidence Dir Accuracy:  {high_conf_dir_acc:.3f}")

def save_comparison_report(results: dict, output_file: str = "models/model_comparison_report.json"):
    """Save detailed comparison report."""
    
    report = {
        'comparison_date': datetime.now().isoformat(),
        'models_compared': list(results.keys()),
        'summary': {}
    }
    
    # Extract key metrics for each model
    for model_name, model_results in results.items():
        summary = model_results.get('summary', {})
        config = model_results.get('training_config', {})
        
        report['summary'][model_name] = {
            'test_rmse': {
                'mean': summary.get('mean_test_rmse', 0),
                'std': summary.get('std_test_rmse', 0)
            },
            'test_dir_accuracy': {
                'mean': summary.get('mean_test_dir_accuracy', 0), 
                'std': summary.get('std_test_dir_accuracy', 0)
            },
            'splits_trained': config.get('splits_trained', 0),
            'model_type': model_name
        }
    
    # Determine overall winner
    if 'lightgbm' in results and 'transformer' in results:
        lgb_rmse = results['lightgbm']['summary'].get('mean_test_rmse', float('inf'))
        transformer_rmse = results['transformer']['summary'].get('mean_test_rmse', float('inf'))
        
        lgb_acc = results['lightgbm']['summary'].get('mean_test_dir_accuracy', 0)
        transformer_acc = results['transformer']['summary'].get('mean_test_dir_accuracy', 0)
        
        # Winner based on RMSE (lower is better)
        rmse_winner = "lightgbm" if lgb_rmse < transformer_rmse else "transformer"
        
        # Winner based on accuracy (higher is better)
        acc_winner = "lightgbm" if lgb_acc > transformer_acc else "transformer"
        
        report['winners'] = {
            'rmse': rmse_winner,
            'accuracy': acc_winner,
            'overall': rmse_winner if rmse_winner == acc_winner else 'tie'
        }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Comparison report saved to: {output_file}")

def main():
    """Main comparison function."""
    
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON ANALYSIS")
    logger.info("=" * 70)
    
    # Load results
    results = load_latest_results()
    
    if not results:
        logger.error("No model results found")
        return
    
    logger.info(f"Found results for models: {list(results.keys())}")
    
    # Run comparisons
    compare_model_performance(results)
    analyze_prediction_patterns(results)
    
    # Save comprehensive report
    save_comparison_report(results)
    
    logger.info("\\n" + "=" * 70)
    logger.info("COMPARISON COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()