"""
Data preprocessing module for converting daily data to weekly format.
Handles alignment, aggregation, and timestamp management.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses raw data into weekly format suitable for modeling."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """
        Initialize the data preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
    
    def load_daily_prices(self, filename: str = "prices_daily.parquet") -> pd.DataFrame:
        """
        Load daily price data from parquet file.
        
        Args:
            filename: Name of the price data file
            
        Returns:
            DataFrame with daily price data
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Price data file not found: {filepath}")
        
        logger.info(f"Loading daily price data from {filepath}")
        df = pd.read_parquet(filepath)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        logger.info(f"Loaded {len(df)} daily price records for {df['ticker'].nunique()} tickers")
        return df
    
    def aggregate_to_weekly(self, daily_df: pd.DataFrame, week_ending: str = 'friday') -> pd.DataFrame:
        """
        Convert daily OHLCV data to weekly format.
        
        Args:
            daily_df: DataFrame with daily price data
            week_ending: Day of week for weekly aggregation ('friday', 'sunday')
            
        Returns:
            DataFrame with weekly aggregated data
        """
        logger.info(f"Aggregating daily data to weekly (week ending: {week_ending})")
        
        # Create a copy to avoid modifying original
        df = daily_df.copy()
        
        # Set the week ending day
        if week_ending.lower() == 'friday':
            # Week ending Friday (business week)
            df['week_end'] = df['date'] + pd.to_timedelta((4 - df['date'].dt.dayofweek) % 7, unit='d')
        elif week_ending.lower() == 'sunday':
            # Week ending Sunday (calendar week)
            df['week_end'] = df['date'] + pd.to_timedelta((6 - df['date'].dt.dayofweek) % 7, unit='d')
        else:
            raise ValueError("week_ending must be 'friday' or 'sunday'")
        
        # Aggregation rules for OHLCV data
        agg_rules = {
            'open': 'first',      # First open of the week
            'high': 'max',        # Highest high of the week
            'low': 'min',         # Lowest low of the week
            'close': 'last',      # Last close of the week
            'volume': 'sum',      # Total volume for the week
            'adj_close': 'last'   # Last adjusted close (if available)
        }
        
        # Group by ticker and week_end, then aggregate
        weekly_data = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Group by week and aggregate
            weekly_ticker = ticker_data.groupby('week_end').agg({
                col: agg_rules.get(col, 'last') for col in ticker_data.columns 
                if col not in ['ticker', 'date', 'week_end', 'ingestion_timestamp', 'source']
            })
            
            # Reset index and add ticker info
            weekly_ticker = weekly_ticker.reset_index()
            weekly_ticker['ticker'] = ticker
            weekly_ticker['date'] = weekly_ticker['week_end']  # Use week_end as date
            
            # Add metadata
            weekly_ticker['trading_days'] = ticker_data.groupby('week_end').size().values
            weekly_ticker['source'] = 'weekly_aggregated'
            weekly_ticker['processing_timestamp'] = datetime.now()
            
            weekly_data.append(weekly_ticker)
        
        # Combine all tickers
        if weekly_data:
            weekly_df = pd.concat(weekly_data, ignore_index=True)
            
            # Clean up columns
            weekly_df = weekly_df.drop('week_end', axis=1)
            
            # Calculate additional weekly metrics
            weekly_df = self._add_weekly_metrics(weekly_df)
            
            logger.info(f"Created {len(weekly_df)} weekly records for {weekly_df['ticker'].nunique()} tickers")
            return weekly_df
        else:
            logger.warning("No weekly data created")
            return pd.DataFrame()
    
    def _add_weekly_metrics(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional weekly metrics to the data.
        
        Args:
            weekly_df: DataFrame with weekly data
            
        Returns:
            DataFrame with additional metrics
        """
        df = weekly_df.copy()
        
        # Weekly return
        if 'close' in df.columns:
            df['weekly_return'] = df.groupby('ticker')['close'].pct_change()
        
        # Weekly volatility (high-low range)
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['weekly_volatility'] = (df['high'] - df['low']) / df['close']
        
        # Average daily volume
        if 'volume' in df.columns and 'trading_days' in df.columns:
            df['avg_daily_volume'] = df['volume'] / df['trading_days']
        
        # OHLC ratios
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['open_close_ratio'] = df['open'] / df['close']
            df['high_low_ratio'] = df['high'] / df['low']
        
        return df
    
    def load_macro_data(self, filename: str = "macro.parquet") -> Optional[pd.DataFrame]:
        """
        Load macro-economic data from parquet file.
        
        Args:
            filename: Name of the macro data file
            
        Returns:
            DataFrame with macro data or None if file doesn't exist
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Macro data file not found: {filepath}")
            return None
        
        logger.info(f"Loading macro data from {filepath}")
        df = pd.read_parquet(filepath)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        logger.info(f"Loaded {len(df)} macro data records")
        return df
    
    def align_macro_to_weekly(self, weekly_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align macro-economic data to weekly price data using forward-fill.
        
        Args:
            weekly_df: DataFrame with weekly price data
            macro_df: DataFrame with macro data
            
        Returns:
            DataFrame with weekly data including aligned macro features
        """
        if macro_df is None or macro_df.empty:
            logger.warning("No macro data to align")
            return weekly_df
        
        logger.info("Aligning macro data to weekly price data")
        
        # Get unique dates from weekly data
        weekly_dates = sorted(weekly_df['date'].unique())
        
        # Create a complete date range for forward-filling
        date_range = pd.date_range(
            start=min(weekly_dates[0], macro_df['date'].min()),
            end=max(weekly_dates[-1], macro_df['date'].max()),
            freq='D'
        )
        
        # Create complete macro dataset with forward-fill
        macro_complete = macro_df.set_index('date').reindex(date_range, method='ffill').reset_index()
        macro_complete = macro_complete.rename(columns={'index': 'date'})
        
        # Extract macro features for weekly dates
        weekly_macro = macro_complete[macro_complete['date'].isin(weekly_dates)].copy()
        
        # Merge with weekly price data
        macro_cols = [col for col in weekly_macro.columns 
                     if col not in ['date', 'ingestion_timestamp', 'source', 'processing_timestamp']]
        
        merged_df = weekly_df.merge(
            weekly_macro[['date'] + macro_cols],
            on='date',
            how='left'
        )
        
        logger.info(f"Added {len(macro_cols)} macro features to weekly data")
        return merged_df
    
    def create_lagged_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 4, 12]) -> pd.DataFrame:
        """
        Create lagged features for time series data.
        
        Args:
            df: DataFrame with time series data
            lag_periods: List of lag periods to create (in weeks)
            
        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Creating lagged features for periods: {lag_periods}")
        logger.info(f"Input columns: {list(df.columns)}")
        
        # Features to lag (exclude metadata columns)
        feature_cols = [col for col in df.columns if col not in [
            'ticker', 'date', 'source', 'ingestion_timestamp', 'processing_timestamp', 'trading_days'
        ]]
        
        logger.info(f"Features to lag: {feature_cols}")
        
        # Create lagged features for each ticker
        lagged_dfs = []
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy().sort_values('date')
            
            for lag in lag_periods:
                for col in feature_cols:
                    if col in ticker_df.columns:
                        ticker_df[f'{col}_lag_{lag}'] = ticker_df[col].shift(lag)
            
            lagged_dfs.append(ticker_df)
        
        result_df = pd.concat(lagged_dfs, ignore_index=True)
        
        logger.info(f"Created lagged features, new shape: {result_df.shape}")
        logger.info(f"Output columns: {list(result_df.columns)}")
        return result_df
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data in the weekly dataset.
        
        Args:
            df: DataFrame with potentially missing data
            method: Method for handling missing data ('forward_fill', 'drop', 'interpolate')
            
        Returns:
            DataFrame with missing data handled
        """
        logger.info(f"Handling missing data using method: {method}")
        
        original_shape = df.shape
        
        if method == 'forward_fill':
            # Forward fill within each ticker
            # First identify numeric columns to fill
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Create a copy to work with
            df_filled = df.copy()
            
            # Forward fill numeric columns within each ticker group
            for ticker in df['ticker'].unique():
                mask = df_filled['ticker'] == ticker
                df_filled.loc[mask, numeric_cols] = df_filled.loc[mask, numeric_cols].ffill()
                
        elif method == 'drop':
            # Drop rows with any missing values
            df_filled = df.dropna()
        elif method == 'interpolate':
            # Interpolate within each ticker
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_filled = df.copy()
            for ticker in df['ticker'].unique():
                mask = df_filled['ticker'] == ticker
                df_filled.loc[mask, numeric_cols] = df_filled.loc[mask, numeric_cols].interpolate()
        else:
            raise ValueError("method must be 'forward_fill', 'drop', or 'interpolate'")
        
        # Ensure we still have the essential columns
        if 'ticker' not in df_filled.columns:
            logger.error("ticker column missing after missing data handling")
            raise ValueError("ticker column was lost during processing")
        
        logger.info(f"Missing data handling: {original_shape} -> {df_filled.shape}")
        return df_filled
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        # Save to parquet
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        logger.info(f"Saved processed data to {filepath}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Tickers: {df['ticker'].nunique()}")
        
        return filepath
    
    def run_preprocessing_pipeline(
        self,
        week_ending: str = 'friday',
        lag_periods: List[int] = [1, 4, 12],
        missing_data_method: str = 'forward_fill'
    ) -> Dict[str, str]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            week_ending: Day of week for weekly aggregation
            lag_periods: Lag periods for feature creation
            missing_data_method: Method for handling missing data
            
        Returns:
            Dictionary with output file paths
        """
        logger.info("Starting preprocessing pipeline...")
        
        results = {}
        
        # 1. Load and aggregate price data
        daily_prices = self.load_daily_prices()
        weekly_prices = self.aggregate_to_weekly(daily_prices, week_ending)
        
        # Save intermediate result
        prices_path = self.save_processed_data(weekly_prices, "prices_weekly.parquet")
        results['prices_weekly'] = prices_path
        
        # 2. Load and align macro data (if available)
        macro_data = self.load_macro_data()
        if macro_data is not None:
            weekly_with_macro = self.align_macro_to_weekly(weekly_prices, macro_data)
        else:
            weekly_with_macro = weekly_prices
        
        # 3. Create lagged features
        weekly_with_lags = self.create_lagged_features(weekly_with_macro, lag_periods)
        
        # 4. Handle missing data
        weekly_clean = self.handle_missing_data(weekly_with_lags, missing_data_method)
        
        # Save final result
        final_path = self.save_processed_data(weekly_clean, "features_weekly.parquet")
        results['features_weekly'] = final_path
        
        logger.info("Preprocessing pipeline completed successfully")
        return results

def main():
    """Example usage of the data preprocessor."""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    results = preprocessor.run_preprocessing_pipeline(
        week_ending='friday',
        lag_periods=[1, 4, 12],
        missing_data_method='forward_fill'
    )
    
    print("\nPREPROCESSING RESULTS:")
    print("=" * 40)
    for key, path in results.items():
        print(f"{key}: {path}")

if __name__ == "__main__":
    main()