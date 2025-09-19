"""
Data splitting and labeling module for walk-forward cross-validation.
Implements time-series aware splits to avoid look-ahead bias.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardSplitter:
    """Implements walk-forward cross-validation for time series data."""
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initialize the walk-forward splitter.
        
        Args:
            processed_data_dir: Directory containing processed data files
        """
        self.processed_data_dir = processed_data_dir
    
    def load_feature_data(self, filename: str = "features_engineered.parquet") -> pd.DataFrame:
        """
        Load engineered feature data.
        
        Args:
            filename: Name of the feature data file
            
        Returns:
            DataFrame with engineered features
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature data file not found: {filepath}")
        
        logger.info(f"Loading feature data from {filepath}")
        df = pd.read_parquet(filepath)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        logger.info(f"Loaded {len(df)} feature records for {df['ticker'].nunique()} tickers")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def validate_target_coverage(self, df: pd.DataFrame, target_col: str = "target_return_12w") -> Dict:
        """
        Validate target label coverage and quality.
        
        Args:
            df: DataFrame with features and targets
            target_col: Name of the target column
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating target coverage for {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in data")
        
        # Calculate coverage statistics
        total_rows = len(df)
        non_null_targets = df[target_col].notna().sum()
        coverage_pct = (non_null_targets / total_rows) * 100
        
        # Date range analysis
        min_date = df['date'].min()
        max_date = df['date'].max()
        total_weeks = (max_date - min_date).days / 7
        
        # Target statistics
        target_stats = df[target_col].describe()
        
        results = {
            'total_rows': total_rows,
            'non_null_targets': non_null_targets,
            'coverage_percentage': coverage_pct,
            'date_range': (min_date, max_date),
            'total_weeks': total_weeks,
            'target_statistics': target_stats.to_dict(),
            'tickers_with_targets': df[df[target_col].notna()]['ticker'].nunique()
        }
        
        logger.info(f"Target coverage: {coverage_pct:.1f}% ({non_null_targets}/{total_rows})")
        logger.info(f"Date range: {min_date} to {max_date} ({total_weeks:.0f} weeks)")
        logger.info(f"Tickers with targets: {results['tickers_with_targets']}")
        
        return results
    
    def create_time_splits(
        self,
        df: pd.DataFrame,
        train_years: int = 6,
        validation_years: int = 1,
        test_years: int = 1,
        step_months: int = 6
    ) -> List[Dict]:
        """
        Create walk-forward time splits.
        
        Args:
            df: DataFrame with time series data
            train_years: Number of years for training
            validation_years: Number of years for validation
            test_years: Number of years for testing
            step_months: Number of months to step forward between splits
            
        Returns:
            List of split dictionaries with date ranges
        """
        logger.info(f"Creating walk-forward splits: {train_years}Y train, {validation_years}Y val, {test_years}Y test")
        
        # Get date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Calculate split parameters
        train_days = train_years * 365
        val_days = validation_years * 365
        test_days = test_years * 365
        step_days = step_months * 30
        
        splits = []
        split_id = 0
        
        # Start from minimum viable start date
        current_start = min_date
        
        while True:
            # Calculate split dates
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            val_start = train_end
            val_end = val_start + timedelta(days=val_days)
            test_start = val_end
            test_end = test_start + timedelta(days=test_days)
            
            # Check if we have enough data
            if test_end > max_date:
                break
            
            # Ensure we have data in each period
            train_data = df[(df['date'] >= train_start) & (df['date'] < train_end)]
            val_data = df[(df['date'] >= val_start) & (df['date'] < val_end)]
            test_data = df[(df['date'] >= test_start) & (df['date'] < test_end)]
            
            if len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0:
                split_info = {
                    'split_id': split_id,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'test_size': len(test_data)
                }
                splits.append(split_info)
                split_id += 1
            
            # Step forward
            current_start += timedelta(days=step_days)
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        # Log split details
        for i, split in enumerate(splits[:3]):  # Show first 3 splits
            logger.info(f"Split {i}: Train {split['train_start'].date()} to {split['train_end'].date()}, "
                       f"Val {split['val_start'].date()} to {split['val_end'].date()}, "
                       f"Test {split['test_start'].date()} to {split['test_end'].date()}")
        
        if len(splits) > 3:
            logger.info(f"... and {len(splits) - 3} more splits")
        
        return splits
    
    def get_split_data(
        self, 
        df: pd.DataFrame, 
        split_info: Dict, 
        target_col: str = "target_return_12w"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract train, validation, and test data for a specific split.
        
        Args:
            df: Full dataset
            split_info: Split information dictionary
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Extract data for each period
        train_data = df[
            (df['date'] >= split_info['train_start']) & 
            (df['date'] < split_info['train_end'])
        ].copy()
        
        val_data = df[
            (df['date'] >= split_info['val_start']) & 
            (df['date'] < split_info['val_end'])
        ].copy()
        
        test_data = df[
            (df['date'] >= split_info['test_start']) & 
            (df['date'] < split_info['test_end'])
        ].copy()
        
        # Remove rows with missing targets
        train_data = train_data[train_data[target_col].notna()]
        val_data = val_data[val_data[target_col].notna()]
        test_data = test_data[test_data[target_col].notna()]
        
        return train_data, val_data, test_data
    
    def prepare_features_and_targets(
        self, 
        df: pd.DataFrame, 
        target_col: str = "target_return_12w",
        feature_prefix_exclude: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare feature matrix and target vector.
        
        Args:
            df: DataFrame with features and targets
            target_col: Name of the target column
            feature_prefix_exclude: List of column prefixes to exclude from features
            
        Returns:
            Tuple of (features_df, targets_series, feature_names)
        """
        if feature_prefix_exclude is None:
            feature_prefix_exclude = ['target_', 'ticker', 'date', 'source', 'ingestion_timestamp', 'processing_timestamp']
        
        # Identify feature columns
        feature_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            if any(col.startswith(prefix) for prefix in feature_prefix_exclude):
                continue
            if df[col].dtype in ['object', 'datetime64[ns]']:
                continue
            feature_cols.append(col)
        
        # Extract features and targets
        features = df[feature_cols].copy()
        targets = df[target_col].copy()
        
        # Handle missing values in features (forward fill within each ticker)
        features_clean = []
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_features = features[ticker_mask].copy()
            ticker_features = ticker_features.fillna(method='ffill').fillna(method='bfill')
            features_clean.append(ticker_features)
        
        features = pd.concat(features_clean, ignore_index=True)
        
        logger.info(f"Prepared {len(feature_cols)} features for {len(features)} samples")
        logger.info(f"Target: {target_col}, non-null targets: {targets.notna().sum()}")
        
        return features, targets, feature_cols
    
    def save_split_data(
        self, 
        splits: List[Dict], 
        df: pd.DataFrame,
        target_col: str = "target_return_12w",
        output_prefix: str = "splits"
    ) -> str:
        """
        Save split information and prepared data.
        
        Args:
            splits: List of split dictionaries
            df: Full dataset
            target_col: Target column name
            output_prefix: Prefix for output files
            
        Returns:
            Directory path where split data was saved
        """
        # Create splits directory
        splits_dir = os.path.join(self.processed_data_dir, output_prefix)
        os.makedirs(splits_dir, exist_ok=True)
        
        # Save split metadata
        splits_df = pd.DataFrame(splits)
        splits_df.to_parquet(os.path.join(splits_dir, "split_metadata.parquet"), index=False)
        
        logger.info(f"Saving split data to {splits_dir}")
        
        # Save data for each split
        for i, split_info in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Get split data
            train_data, val_data, test_data = self.get_split_data(df, split_info, target_col)
            
            # Prepare features and targets for each set
            train_X, train_y, feature_names = self.prepare_features_and_targets(train_data, target_col)
            val_X, val_y, _ = self.prepare_features_and_targets(val_data, target_col)
            test_X, test_y, _ = self.prepare_features_and_targets(test_data, target_col)
            
            # Create split directory
            split_dir = os.path.join(splits_dir, f"split_{i}")
            os.makedirs(split_dir, exist_ok=True)
            
            # Save train/val/test data
            train_X.to_parquet(os.path.join(split_dir, "train_X.parquet"), index=False)
            train_y.to_frame('target').to_parquet(os.path.join(split_dir, "train_y.parquet"), index=False)
            
            val_X.to_parquet(os.path.join(split_dir, "val_X.parquet"), index=False)
            val_y.to_frame('target').to_parquet(os.path.join(split_dir, "val_y.parquet"), index=False)
            
            test_X.to_parquet(os.path.join(split_dir, "test_X.parquet"), index=False)
            test_y.to_frame('target').to_parquet(os.path.join(split_dir, "test_y.parquet"), index=False)
            
            # Save metadata for this split
            split_meta = {
                **split_info,
                'feature_names': feature_names,
                'train_samples': len(train_X),
                'val_samples': len(val_X),
                'test_samples': len(test_X)
            }
            
            pd.DataFrame([split_meta]).to_parquet(
                os.path.join(split_dir, "split_info.parquet"), 
                index=False
            )
        
        logger.info(f"Saved {len(splits)} splits to {splits_dir}")
        return splits_dir
    
    def run_split_pipeline(
        self,
        input_file: str = "features_engineered.parquet",
        target_col: str = "target_return_12w",
        train_years: int = 6,
        validation_years: int = 1,
        test_years: int = 1,
        step_months: int = 6
    ) -> Tuple[str, Dict]:
        """
        Run the complete splitting pipeline.
        
        Args:
            input_file: Input feature file
            target_col: Target column name
            train_years: Training period length
            validation_years: Validation period length
            test_years: Test period length
            step_months: Step size between splits
            
        Returns:
            Tuple of (splits_directory, validation_results)
        """
        logger.info("Starting walk-forward splitting pipeline...")
        
        # Load data
        df = self.load_feature_data(input_file)
        
        # Validate targets
        validation_results = self.validate_target_coverage(df, target_col)
        
        # Create time splits
        splits = self.create_time_splits(
            df, train_years, validation_years, test_years, step_months
        )
        
        # Save split data
        splits_dir = self.save_split_data(splits, df, target_col)
        
        logger.info("Walk-forward splitting pipeline completed")
        return splits_dir, validation_results

def main():
    """Example usage of the walk-forward splitter."""
    
    # Initialize splitter
    splitter = WalkForwardSplitter()
    
    # Run splitting pipeline
    splits_dir, validation_results = splitter.run_split_pipeline(
        target_col="target_return_12w",
        train_years=6,
        validation_years=1, 
        test_years=1,
        step_months=6
    )
    
    print(f"\nSPLITTING RESULTS:")
    print("=" * 40)
    print(f"Splits directory: {splits_dir}")
    print(f"Target coverage: {validation_results['coverage_percentage']:.1f}%")
    print(f"Date range: {validation_results['date_range'][0].date()} to {validation_results['date_range'][1].date()}")
    print(f"Tickers with targets: {validation_results['tickers_with_targets']}")

if __name__ == "__main__":
    main()