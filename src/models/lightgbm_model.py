"""
LightGBM baseline model for stock return prediction.
Implements gradient boosted trees with cross-validation and hyperparameter tuning.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMModel:
    """LightGBM model for stock return prediction."""
    
    def __init__(self, model_dir: str = "models", splits_dir: str = "data/processed/splits"):
        """
        Initialize the LightGBM model.
        
        Args:
            model_dir: Directory to save model artifacts
            splits_dir: Directory containing split data
        """
        self.model_dir = model_dir
        self.splits_dir = splits_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Default hyperparameters
        self.default_params = {
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
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        self.models = {}  # Store models for each split
        self.feature_names = None
        self.training_history = []
    
    def load_split_data(self, split_id: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load data for a specific split.
        
        Args:
            split_id: Split identifier
            
        Returns:
            Tuple of (train_X, train_y, val_X, val_y, test_X, test_y)
        """
        split_dir = os.path.join(self.splits_dir, f"split_{split_id}")
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load training data
        train_X = pd.read_parquet(os.path.join(split_dir, "train_X.parquet"))
        train_y = pd.read_parquet(os.path.join(split_dir, "train_y.parquet"))['target']
        
        # Load validation data
        val_X = pd.read_parquet(os.path.join(split_dir, "val_X.parquet"))
        val_y = pd.read_parquet(os.path.join(split_dir, "val_y.parquet"))['target']
        
        # Load test data
        test_X = pd.read_parquet(os.path.join(split_dir, "test_X.parquet"))
        test_y = pd.read_parquet(os.path.join(split_dir, "test_y.parquet"))['target']
        
        # Store feature names if not already set
        if self.feature_names is None:
            self.feature_names = train_X.columns.tolist()
        
        logger.info(f"Loaded split {split_id}: Train {len(train_X)}, Val {len(val_X)}, Test {len(test_X)}")
        
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def train_single_split(
        self, 
        split_id: int, 
        params: Optional[Dict] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 100
    ) -> Dict[str, Any]:
        """
        Train model on a single split.
        
        Args:
            split_id: Split identifier
            params: LightGBM parameters (uses defaults if None)
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training LightGBM model for split {split_id}")
        
        # Load data
        train_X, train_y, val_X, val_y, test_X, test_y = self.load_split_data(split_id)
        
        # Use default params if none provided
        if params is None:
            params = self.default_params.copy()
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(train_X, label=train_y)
        val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Store model
        self.models[split_id] = model
        
        # Make predictions
        train_pred = model.predict(train_X, num_iteration=model.best_iteration)
        val_pred = model.predict(val_X, num_iteration=model.best_iteration)
        test_pred = model.predict(test_X, num_iteration=model.best_iteration)
        
        # Calculate metrics
        feature_importance = None
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, model.feature_importance()))
        
        results = {
            'split_id': split_id,
            'best_iteration': model.best_iteration,
            'train_rmse': np.sqrt(mean_squared_error(train_y, train_pred)),
            'train_mae': mean_absolute_error(train_y, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(val_y, val_pred)),
            'val_mae': mean_absolute_error(val_y, val_pred),
            'test_rmse': np.sqrt(mean_squared_error(test_y, test_pred)),
            'test_mae': mean_absolute_error(test_y, test_pred),
            'feature_importance': feature_importance,
            'predictions': {
                'train': train_pred.tolist(),
                'val': val_pred.tolist(), 
                'test': test_pred.tolist()
            },
            'actuals': {
                'train': train_y.tolist(),
                'val': val_y.tolist(),
                'test': test_y.tolist()
            }
        }
        
        # Calculate directional accuracy (sign prediction)
        train_dir_acc = accuracy_score(train_y > 0, train_pred > 0)
        val_dir_acc = accuracy_score(val_y > 0, val_pred > 0)
        test_dir_acc = accuracy_score(test_y > 0, test_pred > 0)
        
        results.update({
            'train_dir_accuracy': train_dir_acc,
            'val_dir_accuracy': val_dir_acc,
            'test_dir_accuracy': test_dir_acc
        })
        
        logger.info(f"Split {split_id} - Val RMSE: {results['val_rmse']:.4f}, "
                   f"Test RMSE: {results['test_rmse']:.4f}, "
                   f"Test Dir Acc: {results['test_dir_accuracy']:.3f}")
        
        return results
    
    def train_all_splits(
        self, 
        max_splits: Optional[int] = None,
        params: Optional[Dict] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Train models on all available splits.
        
        Args:
            max_splits: Maximum number of splits to train (None for all)
            params: LightGBM parameters
            save_results: Whether to save results to disk
            
        Returns:
            List of training results for each split
        """
        logger.info("Training LightGBM models on all splits")
        
        # Find available splits
        available_splits = []
        for split_dir in os.listdir(self.splits_dir):
            if split_dir.startswith('split_') and split_dir[6:].isdigit():
                split_id = int(split_dir[6:])
                available_splits.append(split_id)
        
        available_splits.sort()
        
        if max_splits is not None:
            available_splits = available_splits[:max_splits]
        
        logger.info(f"Training on {len(available_splits)} splits: {available_splits}")
        
        # Train models
        all_results = []
        for split_id in available_splits:
            try:
                results = self.train_single_split(split_id, params)
                all_results.append(results)
                self.training_history.append(results)
            except Exception as e:
                logger.error(f"Failed to train split {split_id}: {str(e)}")
                continue
        
        # Calculate summary statistics
        if all_results:
            summary = self.calculate_summary_metrics(all_results)
            logger.info(f"Overall Results - Mean Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
            logger.info(f"Mean Test Dir Accuracy: {summary['mean_test_dir_accuracy']:.3f} ± {summary['std_test_dir_accuracy']:.3f}")
        
        # Save results
        if save_results and all_results:
            self.save_training_results(all_results, summary)
        
        return all_results
    
    def calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate summary statistics across all splits.
        
        Args:
            results: List of training results
            
        Returns:
            Dictionary with summary metrics
        """
        metrics = ['train_rmse', 'val_rmse', 'test_rmse', 'train_mae', 'val_mae', 'test_mae',
                  'train_dir_accuracy', 'val_dir_accuracy', 'test_dir_accuracy']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
        
        return summary
    
    def save_training_results(self, results: List[Dict[str, Any]], summary: Dict[str, float]):
        """
        Save training results to disk.
        
        Args:
            results: List of training results
            summary: Summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.model_dir, f"lgb_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.model_dir, f"lgb_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save models
        for split_id, model in self.models.items():
            model_file = os.path.join(self.model_dir, f"lgb_split_{split_id}_{timestamp}.txt")
            model.save_model(model_file)
        
        logger.info(f"Saved training results to {self.model_dir}")
        logger.info(f"Results: {results_file}")
        logger.info(f"Summary: {summary_file}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across all trained models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance statistics
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        # Collect feature importance from all models
        all_importance = []
        for split_id, model in self.models.items():
            if self.feature_names is not None:
                importance = dict(zip(self.feature_names, model.feature_importance()))
                importance['split_id'] = split_id
                all_importance.append(importance)
        
        importance_df = pd.DataFrame(all_importance)
        
        # Calculate statistics
        feature_stats = []
        if self.feature_names is not None:
            for feature in self.feature_names:
                if feature in importance_df.columns:
                    values = importance_df[feature]
                    feature_stats.append({
                        'feature': feature,
                        'mean_importance': float(values.mean()),
                        'std_importance': float(values.std()),
                        'min_importance': float(values.min()),
                        'max_importance': float(values.max())
                    })
        
        stats_df = pd.DataFrame(feature_stats)
        stats_df = stats_df.sort_values('mean_importance', ascending=False)
        
        return stats_df.head(top_n)
    
    def predict_split(self, split_id: int, dataset: str = 'test') -> np.ndarray:
        """
        Make predictions for a specific split.
        
        Args:
            split_id: Split identifier
            dataset: Dataset to predict on ('train', 'val', 'test')
            
        Returns:
            Array of predictions
        """
        if split_id not in self.models:
            raise ValueError(f"Model for split {split_id} not found")
        
        # Load data
        train_X, train_y, val_X, val_y, test_X, test_y = self.load_split_data(split_id)
        
        if dataset == 'train':
            X = train_X
        elif dataset == 'val':
            X = val_X
        elif dataset == 'test':
            X = test_X
        else:
            raise ValueError("dataset must be 'train', 'val', or 'test'")
        
        model = self.models[split_id]
        predictions = model.predict(X, num_iteration=model.best_iteration)
        
        return predictions

def main():
    """Example usage of the LightGBM model."""
    
    # Initialize model
    lgb_model = LightGBMModel()
    
    # Train on all splits (limiting to first 3 for demo)
    results = lgb_model.train_all_splits(max_splits=3)
    
    if results:
        # Show feature importance
        feature_importance = lgb_model.get_feature_importance(top_n=10)
        print("\nTOP 10 FEATURES:")
        print("=" * 50)
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']:<30} {row['mean_importance']:>8.1f} ± {row['std_importance']:>6.1f}")
        
        # Show summary results
        summary = lgb_model.calculate_summary_metrics(results)
        print(f"\nOVERALL PERFORMANCE:")
        print("=" * 50)
        print(f"Test RMSE:        {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
        print(f"Test Dir Accuracy: {summary['mean_test_dir_accuracy']:.3f} ± {summary['std_test_dir_accuracy']:.3f}")

if __name__ == "__main__":
    main()