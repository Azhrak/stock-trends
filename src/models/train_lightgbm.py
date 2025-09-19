"""
Train baseline models on multiple splits with clear progress monitoring.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/juho/code/azhrak/stock-trends/training_progress.log')
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

def train_lightgbm_split(split_id: int, splits_dir: str) -> Dict[str, Any]:
    """Train LightGBM on a single split."""
    logger.info(f"Training LightGBM on split {split_id}")
    start_time = time.time()
    
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    
    # Load data
    split_dir = os.path.join(splits_dir, f"split_{split_id}")
    
    train_X = pd.read_parquet(os.path.join(split_dir, "train_X.parquet"))
    train_y = pd.read_parquet(os.path.join(split_dir, "train_y.parquet"))['target']
    val_X = pd.read_parquet(os.path.join(split_dir, "val_X.parquet"))
    val_y = pd.read_parquet(os.path.join(split_dir, "val_y.parquet"))['target']
    test_X = pd.read_parquet(os.path.join(split_dir, "test_X.parquet"))
    test_y = pd.read_parquet(os.path.join(split_dir, "test_y.parquet"))['target']
    
    # Simple parameters for speed
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'force_row_wise': True
    }
    
    # Create datasets
    train_data = lgb.Dataset(train_X, label=train_y)
    val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0)  # Silent
        ]
    )
    
    # Predictions
    train_pred = model.predict(train_X, num_iteration=model.best_iteration)
    val_pred = model.predict(val_X, num_iteration=model.best_iteration)
    test_pred = model.predict(test_X, num_iteration=model.best_iteration)
    
    # Metrics
    results = {
        'split_id': split_id,
        'model_type': 'lightgbm',
        'best_iteration': model.best_iteration,
        'train_rmse': float(np.sqrt(mean_squared_error(train_y, train_pred))),
        'val_rmse': float(np.sqrt(mean_squared_error(val_y, val_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(test_y, test_pred))),
        'train_mae': float(mean_absolute_error(train_y, train_pred)),
        'val_mae': float(mean_absolute_error(val_y, val_pred)),
        'test_mae': float(mean_absolute_error(test_y, test_pred)),
        'train_dir_acc': float(accuracy_score(train_y > 0, train_pred > 0)),
        'val_dir_acc': float(accuracy_score(val_y > 0, val_pred > 0)),
        'test_dir_acc': float(accuracy_score(test_y > 0, test_pred > 0)),
        'training_time': time.time() - start_time
    }
    
    logger.info(f"Split {split_id} - Test RMSE: {results['test_rmse']:.4f}, "
               f"Dir Acc: {results['test_dir_acc']:.3f}, "
               f"Time: {results['training_time']:.1f}s")
    
    return results

def train_multiple_splits(
    num_splits: int = 5,
    splits_dir: str = "/Users/juho/code/azhrak/stock-trends/data/processed/splits",
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """Train models on multiple splits with progress tracking."""
    
    logger.info(f"Training LightGBM models on {num_splits} splits")
    
    # Find available splits
    available_splits = []
    for item in os.listdir(splits_dir):
        if item.startswith('split_') and item[6:].isdigit():
            split_id = int(item[6:])
            available_splits.append(split_id)
    
    available_splits.sort()
    available_splits = available_splits[:num_splits]
    
    logger.info(f"Training on splits: {available_splits}")
    
    # Initialize progress tracker
    progress = ProgressTracker(len(available_splits))
    
    # Train each split
    all_results = []
    for split_id in available_splits:
        try:
            result = train_lightgbm_split(split_id, splits_dir)
            all_results.append(result)
            progress.update(result['training_time'])
            
        except Exception as e:
            logger.error(f"Failed to train split {split_id}: {e}")
            progress.update()
            continue
    
    # Calculate summary statistics
    if all_results:
        summary = calculate_summary(all_results)
        logger.info("SUMMARY RESULTS:")
        logger.info(f"  Mean Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
        logger.info(f"  Mean Test Dir Acc: {summary['mean_test_dir_acc']:.3f} ± {summary['std_test_dir_acc']:.3f}")
        logger.info(f"  Mean Training Time: {summary['mean_training_time']:.1f}s")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"/Users/juho/code/azhrak/stock-trends/models/lightgbm_results_{timestamp}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump({
                    'detailed_results': all_results,
                    'summary': summary,
                    'training_config': {
                        'num_splits': num_splits,
                        'splits_trained': len(all_results),
                        'timestamp': timestamp
                    }
                }, f, indent=2)
            
            logger.info(f"Results saved to: {results_file}")
    
    return all_results

def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate summary statistics."""
    metrics = ['test_rmse', 'test_dir_acc', 'val_rmse', 'training_time']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
    
    return summary

def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING - PROGRESSIVE VERSION")
    logger.info("=" * 60)
    
    # Configuration
    NUM_SPLITS = 5  # Start with 5 splits
    
    try:
        # Train models
        results = train_multiple_splits(num_splits=NUM_SPLITS, save_results=True)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Trained {len(results)} models successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()