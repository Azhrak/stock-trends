"""
Feature engineering module for stock prediction.
Computes technical, fundamental, macro, and text-based features.
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

class FeatureEngineer:
    """Computes and manages features for stock prediction."""
    
    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initialize the feature engineer.
        
        Args:
            processed_data_dir: Directory containing processed data files
        """
        self.processed_data_dir = processed_data_dir
    
    def load_weekly_data(self, filename: str = "features_weekly.parquet") -> pd.DataFrame:
        """
        Load weekly processed data.
        
        Args:
            filename: Name of the weekly data file
            
        Returns:
            DataFrame with weekly data
        """
        filepath = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weekly data file not found: {filepath}")
        
        logger.info(f"Loading weekly data from {filepath}")
        df = pd.read_parquet(filepath)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        logger.info(f"Loaded {len(df)} weekly records for {df['ticker'].nunique()} tickers")
        return df
    
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical analysis features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical features
        """
        logger.info("Computing technical features...")
        
        result_dfs = []
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
            
            if len(ticker_data) < 52:  # Need at least 1 year of data
                result_dfs.append(ticker_data)
                continue
            
            # Moving averages
            for window in [4, 12, 26, 52]:  # 1, 3, 6, 12 months
                if 'close' in ticker_data.columns:
                    ticker_data[f'sma_{window}'] = ticker_data['close'].rolling(window=window).mean()
                    ticker_data[f'ema_{window}'] = ticker_data['close'].ewm(span=window).mean()
                    
                    # Price ratios
                    ticker_data[f'price_sma_{window}_ratio'] = ticker_data['close'] / ticker_data[f'sma_{window}']
                    ticker_data[f'price_ema_{window}_ratio'] = ticker_data['close'] / ticker_data[f'ema_{window}']
            
            # Volatility measures
            if 'close' in ticker_data.columns:
                # Rolling volatility (standard deviation of returns)
                returns = ticker_data['close'].pct_change()
                for window in [4, 12, 26]:
                    ticker_data[f'volatility_{window}w'] = returns.rolling(window=window).std() * np.sqrt(52)  # Annualized
                
                # Average True Range (ATR) approximation
                if all(col in ticker_data.columns for col in ['high', 'low', 'close']):
                    high_low = ticker_data['high'] - ticker_data['low']
                    high_close_prev = np.abs(ticker_data['high'] - ticker_data['close'].shift(1))
                    low_close_prev = np.abs(ticker_data['low'] - ticker_data['close'].shift(1))
                    
                    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
                    for window in [4, 12, 26]:
                        ticker_data[f'atr_{window}w'] = true_range.rolling(window=window).mean()
                        ticker_data[f'atr_pct_{window}w'] = ticker_data[f'atr_{window}w'] / ticker_data['close']
            
            # Momentum indicators
            if 'close' in ticker_data.columns:
                # Rate of Change (ROC)
                for period in [1, 4, 12, 26]:
                    ticker_data[f'roc_{period}w'] = ticker_data['close'].pct_change(periods=period)
                
                # Relative Strength Index (RSI) approximation
                delta = ticker_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                ticker_data['rsi_14w'] = 100 - (100 / (1 + rs))
                
                # MACD approximation
                ema_12 = ticker_data['close'].ewm(span=12).mean()
                ema_26 = ticker_data['close'].ewm(span=26).mean()
                ticker_data['macd'] = ema_12 - ema_26
                ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9).mean()
                ticker_data['macd_histogram'] = ticker_data['macd'] - ticker_data['macd_signal']
            
            # Volume features
            if 'volume' in ticker_data.columns:
                # Volume moving averages
                for window in [4, 12, 26]:
                    ticker_data[f'volume_sma_{window}'] = ticker_data['volume'].rolling(window=window).mean()
                    ticker_data[f'volume_ratio_{window}'] = ticker_data['volume'] / ticker_data[f'volume_sma_{window}']
                
                # On-Balance Volume (OBV) approximation
                if 'close' in ticker_data.columns:
                    price_change = ticker_data['close'].diff()
                    obv_change = ticker_data['volume'] * np.sign(price_change)
                    ticker_data['obv'] = obv_change.cumsum()
                    ticker_data['obv_sma_12'] = ticker_data['obv'].rolling(window=12).mean()
            
            # Support and resistance levels (simplified)
            if all(col in ticker_data.columns for col in ['high', 'low']):
                for window in [12, 26, 52]:
                    ticker_data[f'resistance_{window}w'] = ticker_data['high'].rolling(window=window).max()
                    ticker_data[f'support_{window}w'] = ticker_data['low'].rolling(window=window).min()
                    
                    # Avoid division by zero
                    support_resistance_diff = ticker_data[f'resistance_{window}w'] - ticker_data[f'support_{window}w']
                    ticker_data[f'price_position_{window}w'] = np.where(
                        support_resistance_diff > 0,
                        (ticker_data['close'] - ticker_data[f'support_{window}w']) / support_resistance_diff,
                        0.5  # Default to middle position if no range
                    )
            
            result_dfs.append(ticker_data)
        
        # Combine all ticker data
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        logger.info(f"Added technical features, new shape: {result_df.shape}")
        return result_df
    
    def compute_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional features (relative to market/peers).
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with cross-sectional features
        """
        logger.info("Computing cross-sectional features...")
        
        result_df = df.copy()
        
        # Group by date to compute market-relative features
        for date in df['date'].unique():
            date_mask = result_df['date'] == date
            date_data = result_df[date_mask].copy()
            
            if len(date_data) < 5:  # Need enough stocks for meaningful comparison
                continue
            
            # Market cap proxy (using volume and price)
            if all(col in date_data.columns for col in ['close', 'volume']):
                date_data['market_cap_proxy'] = date_data['close'] * date_data['volume']
                
                # Market cap quintiles
                date_data['market_cap_quintile'] = pd.qcut(
                    date_data['market_cap_proxy'], 
                    q=5, 
                    labels=False, 
                    duplicates='drop'
                )
            
            # Relative performance features
            if 'weekly_return' in date_data.columns:
                market_return = date_data['weekly_return'].median()  # Market proxy
                date_data['excess_return'] = date_data['weekly_return'] - market_return
                
                # Return percentiles
                date_data['return_percentile'] = date_data['weekly_return'].rank(pct=True)
            
            # Relative volatility
            if 'volatility_12w' in date_data.columns:
                market_vol = date_data['volatility_12w'].median()
                date_data['relative_volatility'] = date_data['volatility_12w'] / market_vol
            
            # Relative volume
            if 'volume' in date_data.columns:
                market_volume = date_data['volume'].median()
                date_data['relative_volume'] = date_data['volume'] / market_volume
            
            # Update the main dataframe
            result_df.loc[date_mask] = date_data
        
        logger.info(f"Added cross-sectional features, new shape: {result_df.shape}")
        return result_df
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum and trend features.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with momentum features
        """
        logger.info("Computing momentum features...")
        
        result_df = df.copy()
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            mask = result_df['ticker'] == ticker
            ticker_data = result_df[mask].copy().sort_values('date')
            
            if 'close' in ticker_data.columns:
                # Momentum over various periods
                for period in [1, 2, 4, 8, 12, 26, 52]:
                    if len(ticker_data) > period:
                        ticker_data[f'momentum_{period}w'] = (
                            ticker_data['close'] / ticker_data['close'].shift(period) - 1
                        )
                
                # Trend strength (linear regression slope)
                for window in [4, 12, 26]:
                    if len(ticker_data) >= window:
                        # Calculate rolling linear regression slope
                        def calc_slope(series):
                            if len(series) < 2:
                                return np.nan
                            x = np.arange(len(series))
                            slope = np.polyfit(x, series, 1)[0]
                            return slope
                        
                        ticker_data[f'trend_slope_{window}w'] = (
                            ticker_data['close'].rolling(window=window).apply(calc_slope, raw=False)
                        )
                
                # Momentum acceleration (second derivative)
                if 'momentum_4w' in ticker_data.columns:
                    ticker_data['momentum_acceleration'] = ticker_data['momentum_4w'].diff()
                
                # Reversal indicators
                # Check if current return is opposite to recent trend
                if all(col in ticker_data.columns for col in ['weekly_return', 'momentum_4w']):
                    ticker_data['reversal_indicator'] = (
                        (ticker_data['weekly_return'] > 0) & (ticker_data['momentum_4w'] < 0) |
                        (ticker_data['weekly_return'] < 0) & (ticker_data['momentum_4w'] > 0)
                    ).astype(int)
            
            # Update the main dataframe
            result_df.loc[mask] = ticker_data
        
        logger.info(f"Added momentum features, new shape: {result_df.shape}")
        return result_df
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute advanced volatility features.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with volatility features
        """
        logger.info("Computing volatility features...")
        
        result_df = df.copy()
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            mask = result_df['ticker'] == ticker
            ticker_data = result_df[mask].copy().sort_values('date')
            
            if 'close' in ticker_data.columns:
                returns = ticker_data['close'].pct_change()
                
                # Realized volatility over different windows
                for window in [4, 12, 26, 52]:
                    if len(ticker_data) >= window:
                        vol = returns.rolling(window=window).std() * np.sqrt(52)  # Annualized
                        ticker_data[f'realized_vol_{window}w'] = vol
                        
                        # Volatility percentiles (relative to historical)
                        if window >= 52:
                            ticker_data[f'vol_percentile_{window}w'] = vol.rolling(window=52).rank(pct=True)
                
                # Volatility clustering (GARCH-like features)
                if len(returns) > 1:
                    # High volatility regime indicator
                    vol_12w = returns.rolling(window=12).std()
                    vol_52w = returns.rolling(window=52).std()
                    ticker_data['high_vol_regime'] = (vol_12w > vol_52w * 1.5).astype(int)
                    
                    # Volatility momentum
                    vol_4w = returns.rolling(window=4).std()
                    ticker_data['vol_momentum'] = vol_4w / vol_12w
                
                # Extreme returns indicators
                return_abs = np.abs(returns)
                for threshold in [0.05, 0.10, 0.15]:  # 5%, 10%, 15% weekly moves
                    ticker_data[f'extreme_return_{int(threshold*100)}pct'] = (
                        return_abs > threshold
                    ).astype(int)
                
                # Downside volatility (semi-deviation)
                negative_returns = returns[returns < 0]
                for window in [12, 26]:
                    if len(ticker_data) >= window:
                        downside_vol = []
                        for i in range(len(returns)):
                            start_idx = max(0, i - window + 1)
                            period_returns = returns.iloc[start_idx:i+1]
                            negative_period = period_returns[period_returns < 0]
                            if len(negative_period) > 0:
                                downside_vol.append(negative_period.std() * np.sqrt(52))
                            else:
                                downside_vol.append(0)
                        
                        ticker_data[f'downside_vol_{window}w'] = downside_vol
            
            # Update the main dataframe
            result_df.loc[mask] = ticker_data
        
        logger.info(f"Added volatility features, new shape: {result_df.shape}")
        return result_df
    
    def create_target_labels(self, df: pd.DataFrame, horizon_weeks: int = 12) -> pd.DataFrame:
        """
        Create target labels for prediction.
        
        Args:
            df: DataFrame with stock data
            horizon_weeks: Number of weeks to look forward
            
        Returns:
            DataFrame with target labels
        """
        logger.info(f"Creating target labels with {horizon_weeks}-week horizon...")
        
        result_dfs = []
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
            
            if 'close' in ticker_data.columns:
                # Forward return (the main target)
                future_price = ticker_data['close'].shift(-horizon_weeks)
                current_price = ticker_data['close']
                ticker_data[f'target_return_{horizon_weeks}w'] = (future_price / current_price) - 1
                
                # Binary direction target
                ticker_data[f'target_direction_{horizon_weeks}w'] = (
                    ticker_data[f'target_return_{horizon_weeks}w'] > 0
                ).astype(int)
                
                # Categorical targets (quintiles) - only for non-missing values
                target_return = ticker_data[f'target_return_{horizon_weeks}w']
                valid_returns = target_return.dropna()
                if len(valid_returns) > 10:  # Need enough data for quintiles
                    # Calculate quintile breakpoints from historical data
                    quintiles = valid_returns.quantile([0.2, 0.4, 0.6, 0.8]).values
                    
                    def assign_quintile(ret):
                        if pd.isna(ret):
                            return np.nan
                        if ret <= quintiles[0]:
                            return 0  # Bottom quintile
                        elif ret <= quintiles[1]:
                            return 1
                        elif ret <= quintiles[2]:
                            return 2  # Middle quintile
                        elif ret <= quintiles[3]:
                            return 3
                        else:
                            return 4  # Top quintile
                    
                    ticker_data[f'target_quintile_{horizon_weeks}w'] = target_return.apply(assign_quintile)
            
            result_dfs.append(ticker_data)
        
        # Combine all ticker data
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        logger.info(f"Created target labels, new shape: {result_df.shape}")
        return result_df
    
    def run_feature_engineering(
        self,
        input_file: str = "features_weekly.parquet",
        output_file: str = "features_engineered.parquet",
        target_horizon: int = 12
    ) -> str:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            input_file: Input weekly data file
            output_file: Output file for engineered features
            target_horizon: Number of weeks for target labels
            
        Returns:
            Path to output file
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load data
        df = self.load_weekly_data(input_file)
        
        # Compute core features
        df = self.compute_technical_features(df)
        
        # Create target labels
        df = self.create_target_labels(df, target_horizon)
        
        # Save engineered features
        output_path = os.path.join(self.processed_data_dir, output_file)
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        logger.info(f"Feature engineering completed")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Saved to: {output_path}")
        
        # Print feature summary
        feature_cols = [col for col in df.columns if col not in [
            'ticker', 'date', 'source', 'ingestion_timestamp', 'processing_timestamp'
        ]]
        
        logger.info(f"Total features: {len(feature_cols)}")
        
        # Count feature types
        technical_features = [col for col in feature_cols if any(
            keyword in col for keyword in ['sma', 'ema', 'rsi', 'macd', 'atr', 'obv', 'support', 'resistance']
        )]
        
        momentum_features = [col for col in feature_cols if any(
            keyword in col for keyword in ['momentum', 'roc', 'trend', 'reversal']
        )]
        
        volatility_features = [col for col in feature_cols if any(
            keyword in col for keyword in ['vol', 'atr']
        )]
        
        target_features = [col for col in feature_cols if col.startswith('target_')]
        
        logger.info(f"Technical features: {len(technical_features)}")
        logger.info(f"Momentum features: {len(momentum_features)}")
        logger.info(f"Volatility features: {len(volatility_features)}")
        logger.info(f"Target features: {len(target_features)}")
        
        return output_path

def main():
    """Example usage of the feature engineer."""
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Run feature engineering pipeline
    output_path = feature_engineer.run_feature_engineering(
        input_file="features_weekly.parquet",
        output_file="features_engineered.parquet",
        target_horizon=12
    )
    
    print(f"\nFeature engineering completed!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()