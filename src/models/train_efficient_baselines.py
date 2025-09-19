"""
Efficient baseline model training script.
Trains both LightGBM and simplified Temporal Transformer models.
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

def get_efficient_transformer_params():
    """Get efficient transformer parameters for CPU training."""
    model_params = {
        'd_model': 64,      # Smaller model dimension
        'nhead': 4,         # Fewer attention heads
        'num_layers': 2,    # Fewer transformer layers
        'dropout': 0.1
    }
    
    training_params = {
        'batch_size': 64,   # Larger batch size for efficiency
        'learning_rate': 1e-3,  # Higher learning rate
        'num_epochs': 30,   # Reasonable number of epochs
        'patience': 8,      # Early stopping patience
        'weight_decay': 1e-5
    }
    
    return model_params, training_params

def train_efficient_baselines(max_splits: int = 3) -> Dict[str, Any]:
    """
    Train baseline models efficiently.
    
    Args:
        max_splits: Maximum number of splits to train on
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting efficient baseline training on {max_splits} splits")
    
    results = {}
    
    # Train LightGBM model
    logger.info("=" * 60)
    logger.info("TRAINING LIGHTGBM MODEL")
    logger.info("=" * 60)
    
    lgb_model = LightGBMModel()
    lgb_results = lgb_model.train_all_splits(max_splits=max_splits, save_results=True)
    
    if lgb_results:
        lgb_summary = lgb_model.calculate_summary_metrics(lgb_results)
        results['lightgbm'] = {
            'results': lgb_results,
            'summary': lgb_summary
        }
        
        logger.info(f"LightGBM Summary:")
        logger.info(f"  Mean Test RMSE: {lgb_summary['mean_test_rmse']:.4f} ± {lgb_summary['std_test_rmse']:.4f}")
        logger.info(f"  Mean Test Dir Acc: {lgb_summary['mean_test_dir_accuracy']:.3f} ± {lgb_summary['std_test_dir_accuracy']:.3f}")
    
    # Train Temporal Transformer model with efficient settings
    logger.info("=" * 60)
    logger.info("TRAINING EFFICIENT TEMPORAL TRANSFORMER")
    logger.info("=" * 60)
    
    model_params, training_params = get_efficient_transformer_params()
    
    transformer_model = TemporalTransformerModel(sequence_length=12)  # 3 months
    transformer_results = transformer_model.train_all_splits(
        max_splits=max_splits,
        model_params=model_params,
        training_params=training_params,
        save_results=True
    )
    
    if transformer_results:
        transformer_summary = transformer_model.calculate_summary_metrics(transformer_results)
        results['transformer'] = {
            'results': transformer_results,
            'summary': transformer_summary
        }
        
        logger.info(f"Transformer Summary:")
        logger.info(f"  Mean Test RMSE: {transformer_summary['mean_test_rmse']:.4f} ± {transformer_summary['std_test_rmse']:.4f}")
        logger.info(f"  Mean Test Dir Acc: {transformer_summary['mean_test_dir_accuracy']:.3f} ± {transformer_summary['std_test_dir_accuracy']:.3f}")
    
    # Model comparison
    if 'lightgbm' in results and 'transformer' in results:
        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        lgb_test_rmse = results['lightgbm']['summary']['mean_test_rmse']
        trans_test_rmse = results['transformer']['summary']['mean_test_rmse']
        
        lgb_test_acc = results['lightgbm']['summary']['mean_test_dir_accuracy']
        trans_test_acc = results['transformer']['summary']['mean_test_dir_accuracy']
        
        logger.info("Test RMSE (lower is better):")
        if lgb_test_rmse < trans_test_rmse:
            logger.info(f"  1. LightGBM:    {lgb_test_rmse:.4f} ⭐")
            logger.info(f"  2. Transformer: {trans_test_rmse:.4f}")
        else:
            logger.info(f"  1. Transformer: {trans_test_rmse:.4f} ⭐")
            logger.info(f"  2. LightGBM:    {lgb_test_rmse:.4f}")
        
        logger.info("Test Directional Accuracy (higher is better):")
        if lgb_test_acc > trans_test_acc:
            logger.info(f"  1. LightGBM:    {lgb_test_acc:.3f} ⭐")
            logger.info(f"  2. Transformer: {trans_test_acc:.3f}")
        else:
            logger.info(f"  1. Transformer: {trans_test_acc:.3f} ⭐")
            logger.info(f"  2. LightGBM:    {lgb_test_acc:.3f}")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"models/baseline_comparison_{timestamp}.json"
    
    # Prepare serializable results
    serializable_results = {}
    for model_name, model_data in results.items():
        serializable_results[model_name] = {
            'summary': model_data['summary'],
            'num_splits': len(model_data['results'])
        }
    
    with open(comparison_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved comparison to: {comparison_file}")
    
    return results

def show_feature_importance(lgb_model: LightGBMModel, top_n: int = 15):
    """Show feature importance from LightGBM model."""
    try:
        feature_importance = lgb_model.get_feature_importance(top_n=top_n)
        
        logger.info(f"TOP {top_n} FEATURES:")
        logger.info("-" * 60)
        logger.info(f"{'Feature':<35} {'Importance':<12} {'Std':<8}")
        logger.info("-" * 60)
        
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['feature']:<35} {row['mean_importance']:>8.1f} {row['std_importance']:>8.1f}")
            
    except Exception as e:
        logger.warning(f"Could not show feature importance: {e}")

def main():
    """Main training function."""
    logger.info("Starting efficient baseline model training")
    
    try:
        # Train models on first 3 splits
        results = train_efficient_baselines(max_splits=3)
        
        # Show feature importance if LightGBM trained
        if 'lightgbm' in results:
            logger.info("=" * 60)
            lgb_model = LightGBMModel()
            # Re-train single split to get feature importance
            lgb_model.train_single_split(0)
            show_feature_importance(lgb_model)
        
        logger.info("=" * 60)
        logger.info("BASELINE TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Summary
        for model_name, model_data in results.items():
            summary = model_data['summary']
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  Test RMSE: {summary['mean_test_rmse']:.4f}")
            logger.info(f"  Test Dir Accuracy: {summary['mean_test_dir_accuracy']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()