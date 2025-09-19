"""
Simple baseline training with extensive logging to debug issues.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

# Setup very verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/juho/code/azhrak/stock-trends/training.log')
    ]
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test basic data loading functionality."""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING")
    logger.info("=" * 60)
    
    splits_dir = "/Users/juho/code/azhrak/stock-trends/data/processed/splits"
    
    logger.info(f"Checking splits directory: {splits_dir}")
    if not os.path.exists(splits_dir):
        logger.error(f"Splits directory does not exist: {splits_dir}")
        return False
    
    # List available splits
    available_splits = []
    for item in os.listdir(splits_dir):
        if item.startswith('split_') and item[6:].isdigit():
            split_id = int(item[6:])
            available_splits.append(split_id)
    
    available_splits.sort()
    logger.info(f"Found {len(available_splits)} splits: {available_splits}")
    
    if not available_splits:
        logger.error("No splits found!")
        return False
    
    # Test loading first split
    split_id = available_splits[0]
    split_dir = os.path.join(splits_dir, f"split_{split_id}")
    
    logger.info(f"Testing split {split_id} in {split_dir}")
    
    try:
        logger.info("Loading train_X...")
        train_X = pd.read_parquet(os.path.join(split_dir, "train_X.parquet"))
        logger.info(f"train_X shape: {train_X.shape}")
        
        logger.info("Loading train_y...")
        train_y = pd.read_parquet(os.path.join(split_dir, "train_y.parquet"))['target']
        logger.info(f"train_y shape: {train_y.shape}")
        
        logger.info("Loading val_X...")
        val_X = pd.read_parquet(os.path.join(split_dir, "val_X.parquet"))
        logger.info(f"val_X shape: {val_X.shape}")
        
        logger.info("Loading val_y...")
        val_y = pd.read_parquet(os.path.join(split_dir, "val_y.parquet"))['target']
        logger.info(f"val_y shape: {val_y.shape}")
        
        logger.info("Data loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Data loading test FAILED: {e}")
        return False

def train_simple_lightgbm():
    """Train a very simple LightGBM model with extensive logging."""
    logger.info("=" * 60)
    logger.info("TRAINING SIMPLE LIGHTGBM")
    logger.info("=" * 60)
    
    try:
        import lightgbm as lgb
        logger.info("LightGBM imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import LightGBM: {e}")
        return False
    
    # Load data
    splits_dir = "/Users/juho/code/azhrak/stock-trends/data/processed/splits"
    split_dir = os.path.join(splits_dir, "split_0")
    
    logger.info("Loading data for training...")
    start_time = time.time()
    
    train_X = pd.read_parquet(os.path.join(split_dir, "train_X.parquet"))
    train_y = pd.read_parquet(os.path.join(split_dir, "train_y.parquet"))['target']
    val_X = pd.read_parquet(os.path.join(split_dir, "val_X.parquet"))
    val_y = pd.read_parquet(os.path.join(split_dir, "val_y.parquet"))['target']
    
    load_time = time.time() - start_time
    logger.info(f"Data loaded in {load_time:.2f} seconds")
    logger.info(f"Train: {train_X.shape}, Val: {val_X.shape}")
    
    # Check for any NaN values
    train_nans = train_X.isnull().sum().sum()
    val_nans = val_X.isnull().sum().sum()
    logger.info(f"NaN values - Train: {train_nans}, Val: {val_nans}")
    
    if train_nans > 0 or val_nans > 0:
        logger.warning("Found NaN values, filling with 0")
        train_X = train_X.fillna(0)
        val_X = val_X.fillna(0)
    
    # Simple LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 10,  # Very simple
        'learning_rate': 0.1,
        'verbose': 1,  # Enable verbose output
        'force_row_wise': True,  # Better for small datasets
    }
    
    logger.info(f"Creating LightGBM datasets...")
    start_time = time.time()
    
    train_data = lgb.Dataset(train_X, label=train_y)
    val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)
    
    dataset_time = time.time() - start_time
    logger.info(f"Datasets created in {dataset_time:.2f} seconds")
    
    logger.info("Starting training...")
    start_time = time.time()
    
    # Train with minimal iterations
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=20,  # Very few rounds
        callbacks=[
            lgb.early_stopping(stopping_rounds=5),
            lgb.log_evaluation(period=1)  # Log every iteration
        ]
    )
    
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    logger.info("Making predictions...")
    val_pred = model.predict(val_X, num_iteration=model.best_iteration)
    
    # Calculate simple metrics
    from sklearn.metrics import mean_squared_error
    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    logger.info("Simple LightGBM training COMPLETED successfully")
    
    return True

def main():
    """Main function with step-by-step testing."""
    logger.info("Starting simple baseline training with debugging")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    
    # Test 1: Data loading
    logger.info("\n" + "="*60)
    logger.info("TEST 1: DATA LOADING")
    logger.info("="*60)
    
    if not test_data_loading():
        logger.error("Data loading test failed, aborting")
        return
    
    # Test 2: Simple LightGBM
    logger.info("\n" + "="*60)
    logger.info("TEST 2: SIMPLE LIGHTGBM")
    logger.info("="*60)
    
    try:
        if train_simple_lightgbm():
            logger.info("All tests PASSED!")
        else:
            logger.error("LightGBM test failed")
    except Exception as e:
        logger.error(f"Unexpected error in LightGBM test: {e}", exc_info=True)
    
    logger.info("Script completed")

if __name__ == "__main__":
    main()