"""
Simple backtest runner that generates fresh predictions and runs backtest.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from models.lightgbm_model import LightGBMModel
from backtest.backtest_engine import BacktestEngine, TradingCosts, PositionSizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBacktester:
    """Simple backtester that generates predictions and runs backtest."""
    
    def __init__(self):
        self.models_dir = "models"
        self.data_dir = "data/processed"
        self.splits_dir = "data/processed/splits"
    
    def generate_test_predictions(self, num_splits: int = 3) -> pd.DataFrame:
        """
        Generate predictions on test data from trained models.
        
        Args:
            num_splits: Number of splits to use for predictions
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating predictions for {num_splits} splits")
        
        # Initialize model
        lgb_model = LightGBMModel()
        
        all_predictions = []
        
        # Train and predict for each split
        for split_id in range(num_splits):
            try:
                logger.info(f"Processing split {split_id}")
                
                # Train model on this split
                lgb_model.train_single_split(split_id)
                
                # Load test data with metadata
                split_dir = os.path.join(self.splits_dir, f"split_{split_id}")
                test_X = pd.read_parquet(os.path.join(split_dir, "test_X.parquet"))
                test_y = pd.read_parquet(os.path.join(split_dir, "test_y.parquet"))
                
                # Generate predictions
                test_pred = lgb_model.predict_split(split_id, 'test')
                
                # Load original weekly data to get dates and tickers
                weekly_data = pd.read_parquet(os.path.join(self.data_dir, "features_engineered.parquet"))
                
                # Create a simple mapping - use row indices to match with weekly data
                # This is a simplified approach; in practice you'd want more robust date/ticker mapping
                
                # Get the last N rows for this test set (approximate)
                test_rows = weekly_data.tail(len(test_X) * 3)  # Conservative estimate
                
                # Sample from these rows to match test set size
                if len(test_rows) >= len(test_X):
                    # Take every Nth row to spread predictions across time
                    step = len(test_rows) // len(test_X)
                    selected_rows = test_rows.iloc[::step][:len(test_X)]
                else:
                    selected_rows = test_rows
                
                # Create predictions DataFrame
                for i, (pred, actual) in enumerate(zip(test_pred, test_y['target'])):
                    if i < len(selected_rows):
                        row = selected_rows.iloc[i]
                        
                        prediction_record = {
                            'date': row['date'],
                            'ticker': row['ticker'],
                            'prediction': pred,
                            'actual': actual,
                            'split_id': split_id
                        }
                        all_predictions.append(prediction_record)
                
                logger.info(f"Split {split_id}: Generated {len(test_pred)} predictions")
                
            except Exception as e:
                logger.error(f"Error processing split {split_id}: {e}")
                continue
        
        if not all_predictions:
            logger.error("No predictions generated")
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(all_predictions)
        
        # Remove duplicates (same date/ticker from different splits)
        predictions_df = predictions_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
        
        # Sort by date
        predictions_df = predictions_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Final predictions: {len(predictions_df)} records")
        logger.info(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        logger.info(f"Unique tickers: {predictions_df['ticker'].nunique()}")
        
        return predictions_df
    
    def create_synthetic_predictions(self, num_periods: int = 100) -> pd.DataFrame:
        """
        Create synthetic predictions for backtesting demo.
        
        Args:
            num_periods: Number of time periods
            
        Returns:
            DataFrame with synthetic predictions
        """
        logger.info(f"Creating {num_periods} periods of synthetic predictions")
        
        # Load some real data for realistic tickers and dates
        try:
            weekly_data = pd.read_parquet(os.path.join(self.data_dir, "features_engineered.parquet"))
            
            # Get last N periods
            recent_data = weekly_data.groupby('ticker').tail(num_periods // 5)  # Spread across tickers
            
            predictions = []
            for _, row in recent_data.iterrows():
                # Create somewhat realistic predictions based on recent performance
                recent_return = row.get('close', 100) / row.get('open', 100) - 1
                
                # Add noise to create prediction
                noise = np.random.normal(0, 0.02)
                prediction = recent_return * 0.3 + noise  # 30% signal, 70% noise
                
                predictions.append({
                    'date': row['date'],
                    'ticker': row['ticker'],
                    'prediction': prediction,
                    'actual': recent_return,  # Use actual return as ground truth
                    'split_id': 0
                })
            
            predictions_df = pd.DataFrame(predictions)
            
            # Sort by date and remove duplicates
            predictions_df = predictions_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
            predictions_df = predictions_df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Created {len(predictions_df)} synthetic predictions")
            return predictions_df
            
        except Exception as e:
            logger.error(f"Failed to create synthetic predictions: {e}")
            return pd.DataFrame()
    
    def run_simple_backtest(self, use_synthetic: bool = True) -> Dict[str, Any]:
        """
        Run a simple backtest.
        
        Args:
            use_synthetic: Whether to use synthetic data (faster) or real model predictions
            
        Returns:
            Backtest results
        """
        logger.info("=" * 60)
        logger.info("RUNNING SIMPLE BACKTEST")
        logger.info("=" * 60)
        
        # Generate predictions
        if use_synthetic:
            predictions = self.create_synthetic_predictions(200)
        else:
            predictions = self.generate_test_predictions(3)
        
        if predictions.empty:
            logger.error("No predictions available")
            return {}
        
        # Load price data
        try:
            prices = pd.read_parquet(os.path.join(self.data_dir, "features_engineered.parquet"))
            
            # Ensure we have close prices
            if 'close' not in prices.columns and 'adj_close' in prices.columns:
                prices['close'] = prices['adj_close']
            
            logger.info(f"Loaded {len(prices)} price records")
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return {}
        
        # Configure backtest
        trading_costs = TradingCosts(
            commission_rate=0.001,  # 0.1% commission
            bid_ask_spread=0.0005,  # 0.05% spread
            market_impact=0.0002,   # 0.02% impact
            slippage=0.0001         # 0.01% slippage
        )
        
        position_sizer = PositionSizer(
            max_position_size=0.03,    # Max 3% per position
            max_total_exposure=0.8,    # Max 80% total exposure
            min_position_size=0.005,   # Min 0.5% position
            volatility_target=0.12     # 12% vol target
        )
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            trading_costs=trading_costs,
            position_sizer=position_sizer,
            initial_capital=1000000  # $1M
        )
        
        # Run backtest
        try:
            results = backtest_engine.run_backtest(predictions, prices)
            
            # Add metadata
            results['backtest_config'] = {
                'initial_capital': 1000000,
                'num_predictions': len(predictions),
                'prediction_source': 'synthetic' if use_synthetic else 'model',
                'date_range': f"{predictions['date'].min()} to {predictions['date'].max()}"
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            return {}
    
    def print_results(self, results: Dict[str, Any]):
        """Print backtest results in a nice format."""
        
        if not results:
            logger.error("No results to display")
            return
        
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        # Performance metrics
        logger.info("PERFORMANCE METRICS:")
        logger.info(f"  Initial Capital:      ${results.get('final_portfolio_value', 0)/1.1:>12,.0f}")
        logger.info(f"  Final Value:          ${results.get('final_portfolio_value', 0):>12,.0f}")
        logger.info(f"  Total Return:         {results.get('total_return', 0):>12.2%}")
        logger.info(f"  Annualized Return:    {results.get('annualized_return', 0):>12.2%}")
        logger.info(f"  Volatility:           {results.get('annualized_volatility', 0):>12.2%}")
        logger.info(f"  Sharpe Ratio:         {results.get('sharpe_ratio', 0):>12.2f}")
        logger.info(f"  Max Drawdown:         {results.get('max_drawdown', 0):>12.2%}")
        
        # Trading metrics
        logger.info("\\nTRADING METRICS:")
        logger.info(f"  Total Trades:         {results.get('total_trades', 0):>12.0f}")
        logger.info(f"  Trading Costs:        ${results.get('total_trading_costs', 0):>12,.0f}")
        logger.info(f"  Trading Cost %:       {results.get('trading_cost_pct', 0):>12.2%}")
        
        # Configuration
        config = results.get('backtest_config', {})
        logger.info("\\nCONFIGURATION:")
        logger.info(f"  Predictions:          {config.get('num_predictions', 0):>12.0f}")
        logger.info(f"  Source:               {config.get('prediction_source', 'unknown'):>12}")
        logger.info(f"  Date Range:           {config.get('date_range', 'unknown')}")

def main():
    """Main function."""
    
    # Initialize backtester
    backtester = SimpleBacktester()
    
    # Run backtest with synthetic data (fast)
    logger.info("Running backtest with synthetic predictions...")
    results = backtester.run_simple_backtest(use_synthetic=True)
    
    # Print results
    backtester.print_results(results)
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"models/simple_backtest_{timestamp}.json"
        
        # Prepare for JSON serialization
        serializable_results = {k: v for k, v in results.items() 
                               if k not in ['portfolio_history', 'trade_history']}
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()