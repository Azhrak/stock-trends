"""
Train temporal transformer models on multiple splits with optimized parameters.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from models.temporal_transformer import TemporalTransformerModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/juho/code/azhrak/stock-trends/transformer_training.log')
    ]
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track training progress with time estimates."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_times = []
    
    def update(self, task_time: float = None):
        """Update progress."""
        self.completed_tasks += 1
        if task_time:
            self.task_times.append(task_time)
        
        elapsed = time.time() - self.start_time
        progress = self.completed_tasks / self.total_tasks
        
        if self.task_times:
            avg_task_time = np.mean(self.task_times)
            remaining_tasks = self.total_tasks - self.completed_tasks
            eta = remaining_tasks * avg_task_time
        else:
            eta = None
        
        logger.info(f"Progress: {self.completed_tasks}/{self.total_tasks} ({progress:.1%}) - "
                   f"Elapsed: {elapsed:.1f}s" + 
                   (f" - ETA: {eta:.1f}s" if eta else ""))

def train_transformer_multiple_splits(
    num_splits: int = 3,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Train temporal transformer models on multiple splits.
    
    Args:
        num_splits: Number of splits to train on
        save_results: Whether to save results to disk
        
    Returns:
        List of training results for each split
    """
    logger.info(f"Training Temporal Transformer models on {num_splits} splits")
    
    # Optimized parameters for faster training
    model_params = {
        'd_model': 64,      # Reduced from 128
        'nhead': 4,         # Reduced from 8
        'num_layers': 2,    # Reduced from 3
        'dropout': 0.1
    }
    
    training_params = {
        'batch_size': 64,           # Increased batch size
        'learning_rate': 1e-3,      # Higher learning rate
        'num_epochs': 50,           # Reduced epochs
        'patience': 5,              # Reduced patience
        'weight_decay': 1e-4
    }
    
    # Initialize model with optimized parameters
    transformer_model = TemporalTransformerModel(
        sequence_length=13,  # Reduced from 26 (quarterly instead of half-year)
        device='cpu'  # Ensure CPU usage
    )
    transformer_model.model_params = model_params
    transformer_model.training_params = training_params
    
    # Initialize progress tracker
    progress = ProgressTracker(num_splits)
    
    # Train each split
    all_results = []
    for split_id in range(num_splits):
        try:
            logger.info(f"Training Temporal Transformer on split {split_id}")
            start_time = time.time()
            
            result = transformer_model.train_single_split(
                split_id, 
                model_params=model_params,
                training_params=training_params
            )
            
            training_time = time.time() - start_time
            result['training_time'] = training_time
            
            all_results.append(result)
            progress.update(training_time)
            
            logger.info(f"Split {split_id}: Test RMSE: {result['test_rmse']:.4f}, "
                       f"Dir Acc: {result['test_dir_accuracy']:.3f}, "
                       f"Time: {training_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to train split {split_id}: {e}")
            progress.update()
            continue
    
    # Calculate summary statistics
    if all_results:
        summary = calculate_summary(all_results)
        logger.info("TRANSFORMER SUMMARY RESULTS:")
        logger.info(f"  Mean Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
        logger.info(f"  Mean Test Dir Acc: {summary['mean_test_dir_accuracy']:.3f} ± {summary['std_test_dir_accuracy']:.3f}")
        logger.info(f"  Mean Training Time: {summary['mean_training_time']:.1f}s")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"/Users/juho/code/azhrak/stock-trends/models/transformer_results_{timestamp}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump({
                    'detailed_results': all_results,
                    'summary': summary,
                    'training_config': {
                        'num_splits': num_splits,
                        'splits_trained': len(all_results),
                        'model_params': model_params,
                        'training_params': training_params,
                        'timestamp': timestamp
                    }
                }, f, indent=2)
            
            logger.info(f"Transformer results saved to: {results_file}")
    
    return all_results

def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate summary statistics."""
    metrics = ['test_rmse', 'test_dir_accuracy', 'val_rmse', 'training_time']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
    
    return summary

def compare_models():
    """Compare LightGBM and Transformer results."""
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    
    models_dir = "/Users/juho/code/azhrak/stock-trends/models"
    
    # Find latest results files
    lgb_files = [f for f in os.listdir(models_dir) if f.startswith('lightgbm_results_') and f.endswith('.json')]
    transformer_files = [f for f in os.listdir(models_dir) if f.startswith('transformer_results_') and f.endswith('.json')]
    
    if not lgb_files:
        logger.warning("No LightGBM results found")
        return
    
    if not transformer_files:
        logger.warning("No Transformer results found")
        return
    
    # Load latest results
    lgb_file = os.path.join(models_dir, sorted(lgb_files)[-1])
    transformer_file = os.path.join(models_dir, sorted(transformer_files)[-1])
    
    try:
        with open(lgb_file, 'r') as f:
            lgb_results = json.load(f)
        
        with open(transformer_file, 'r') as f:
            transformer_results = json.load(f)
        
        # Extract summaries
        lgb_summary = lgb_results.get('summary', {})
        transformer_summary = transformer_results.get('summary', {})
        
        # Compare key metrics
        metrics = ['test_rmse', 'test_dir_accuracy', 'training_time']
        
        logger.info(f"{'Metric':<20} {'LightGBM':<15} {'Transformer':<15} {'Winner':<10}")
        logger.info("-" * 65)
        
        for metric in metrics:
            lgb_mean = lgb_summary.get(f'mean_{metric}', 0)
            transformer_mean = transformer_summary.get(f'mean_{metric}', 0)
            
            if metric in ['test_rmse', 'training_time']:
                # Lower is better
                winner = "LightGBM" if lgb_mean < transformer_mean else "Transformer"
            else:
                # Higher is better
                winner = "LightGBM" if lgb_mean > transformer_mean else "Transformer"
            
            logger.info(f"{metric:<20} {lgb_mean:<15.4f} {transformer_mean:<15.4f} {winner:<10}")
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")

def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("TEMPORAL TRANSFORMER TRAINING")
    logger.info("=" * 60)
    
    # Configuration
    NUM_SPLITS = 3  # Start with 3 splits for speed
    
    try:
        # Train transformer models
        results = train_transformer_multiple_splits(num_splits=NUM_SPLITS, save_results=True)
        
        if results:
            logger.info("=" * 60)
            logger.info("TRANSFORMER TRAINING COMPLETED!")
            logger.info(f"Trained {len(results)} models successfully")
            logger.info("=" * 60)
            
            # Compare with LightGBM
            compare_models()
        else:
            logger.error("No transformer models trained successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()