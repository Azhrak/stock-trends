"""
Multi-model training utilities for baseline comparisons and research.

This module provides functionality to train multiple models in a single session
for comparison purposes, extracted from train_baselines.py.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from models.lightgbm_model import LightGBMModel
from models.temporal_transformer import TemporalTransformerModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_baseline_models(
    max_splits: int = 3,
    save_results: bool = True,
    models_to_train: List[str] = ['lightgbm', 'transformer']
) -> Dict[str, Any]:
    """
    Train baseline models on multiple splits for comparison.
    
    Args:
        max_splits: Maximum number of splits to train on
        save_results: Whether to save results to disk
        models_to_train: List of models to train ('lightgbm', 'transformer')
        
    Returns:
        Dictionary with training results for all models
    """
    logger.info(f"Starting baseline model training on {max_splits} splits")
    logger.info(f"Models to train: {models_to_train}")
    
    results = {}
    
    # Train LightGBM model
    if 'lightgbm' in models_to_train:
        logger.info("=" * 60)
        logger.info("TRAINING LIGHTGBM MODEL")
        logger.info("=" * 60)
        
        lgb_model = LightGBMModel()
        lgb_results = lgb_model.train_all_splits(
            max_splits=max_splits, 
            save_results=save_results
        )
        
        if lgb_results and 'detailed_results' in lgb_results:
            results['lightgbm'] = {
                'model_instance': lgb_model,
                'detailed_results': lgb_results['detailed_results'],
                'summary': lgb_results['summary']
            }
            
            # Show feature importance
            feature_importance = lgb_model.get_feature_importance(top_n=10)
            logger.info("\\nTOP 10 FEATURES (LightGBM):")
            logger.info("-" * 50)
            for _, row in feature_importance.iterrows():
                logger.info(f"{row['feature']:<30} {row['mean_importance']:>8.1f} ± {row['std_importance']:>6.1f}")
        else:
            logger.error("No LightGBM results obtained")
    
    # Train Temporal Transformer model  
    if 'transformer' in models_to_train:
        logger.info("=" * 60)
        logger.info("TRAINING TEMPORAL TRANSFORMER MODEL")
        logger.info("=" * 60)
        
        transformer_model = TemporalTransformerModel(sequence_length=26)
        transformer_results = transformer_model.train_all_splits(
            max_splits=max_splits, 
            save_results=save_results
        )
        
        if transformer_results and 'detailed_results' in transformer_results:
            results['transformer'] = {
                'model_instance': transformer_model,
                'detailed_results': transformer_results['detailed_results'],
                'summary': transformer_results['summary']
            }
            
            # Summary for transformer
            summary = transformer_results['summary']
            logger.info(f"\\nTransformer Training Summary:")
            logger.info(f"Mean Test RMSE: {summary.get('mean_test_rmse', 'N/A'):.4f}")
            logger.info(f"Mean Test Dir Accuracy: {summary.get('mean_test_dir_accuracy', 'N/A'):.3f}")
        else:
            logger.error("No Transformer results obtained")
    
    # Compare models if multiple were trained
    if len(results) > 1:
        logger.info("\\n" + "=" * 60)
        logger.info("COMPARING MODEL PERFORMANCE")
        logger.info("=" * 60)
        
        comparison = compare_models(results)
        results['comparison'] = comparison
    
    return results

def compare_models(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare performance of different models with ranking.
    
    Args:
        results: Dictionary with results for each model
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    # Extract key metrics for comparison
    metrics_to_compare = ['test_rmse', 'test_dir_accuracy', 'val_rmse']
    
    for metric in metrics_to_compare:
        metric_comparison = {}
        
        for model_name in results:
            if model_name == 'comparison':
                continue
                
            summary = results[model_name]['summary']
            mean_key = f'mean_{metric}'
            std_key = f'std_{metric}'
            
            if mean_key in summary and std_key in summary:
                metric_comparison[model_name] = {
                    'mean': summary[mean_key],
                    'std': summary[std_key]
                }
        
        comparison[metric] = metric_comparison
    
    # Log comparison with ranking
    for metric, model_results in comparison.items():
        logger.info(f"\\n{metric.upper()} RANKING:")
        logger.info("-" * 40)
        
        # Sort by mean performance
        if metric == 'test_rmse' or metric == 'val_rmse':
            # Lower is better for RMSE
            sorted_models = sorted(model_results.items(), key=lambda x: x[1]['mean'])
        else:
            # Higher is better for accuracy
            sorted_models = sorted(model_results.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for i, (model_name, stats) in enumerate(sorted_models):
            rank = i + 1
            logger.info(f"{rank}. {model_name:<15} {stats['mean']:>8.4f} ± {stats['std']:>6.4f}")
    
    return comparison

def save_combined_results(results: Dict[str, Any], output_dir: str = "models"):
    """
    Save combined results from all models.
    
    Args:
        results: Combined results dictionary
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for model_name, model_results in results.items():
        if model_name == 'comparison':
            serializable_results[model_name] = model_results
        else:
            serializable_results[model_name] = {
                'detailed_results': model_results['detailed_results'],
                'summary': model_results['summary']
                # Skip model_instance as it's not serializable
            }
    
    # Save combined results
    combined_file = os.path.join(output_dir, f"baseline_comparison_{timestamp}.json")
    with open(combined_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Saved combined results to: {combined_file}")

def quick_baseline_comparison(
    models_to_train: List[str] = ['lightgbm', 'transformer'],
    max_splits: int = 2
) -> Dict[str, Any]:
    """
    Quick baseline comparison for research purposes.
    
    Args:
        models_to_train: List of models to compare
        max_splits: Number of splits to train on (keep low for speed)
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("=" * 60)
    logger.info("QUICK BASELINE COMPARISON")
    logger.info("=" * 60)
    
    # Train models
    results = train_baseline_models(
        max_splits=max_splits,
        save_results=False,  # Don't save for quick comparisons
        models_to_train=models_to_train
    )
    
    # Save combined results
    save_combined_results(results)
    
    return results

def main():
    """Main training function for standalone use."""
    logger.info("Starting multi-model baseline training")
    
    # Configuration
    MAX_SPLITS = 3  # Train on first 3 splits for speed
    MODELS_TO_TRAIN = ['lightgbm', 'transformer']
    
    try:
        # Train models
        results = train_baseline_models(
            max_splits=MAX_SPLITS,
            save_results=True,
            models_to_train=MODELS_TO_TRAIN
        )
        
        # Save combined results
        save_combined_results(results)
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("MULTI-MODEL TRAINING COMPLETED")
        logger.info("=" * 60)
        
        for model_name in MODELS_TO_TRAIN:
            if model_name in results:
                summary = results[model_name]['summary']
                logger.info(f"\\n{model_name.upper()} FINAL RESULTS:")
                logger.info(f"  Mean Test RMSE: {summary.get('mean_test_rmse', 'N/A'):.4f}")
                logger.info(f"  Mean Test Dir Accuracy: {summary.get('mean_test_dir_accuracy', 'N/A'):.3f}")
        
    except Exception as e:
        logger.error(f"Multi-model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()