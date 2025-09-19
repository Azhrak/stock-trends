"""
Integration script to run backtests using trained model predictions.
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

from backtest.backtest_engine import BacktestEngine, TradingCosts, PositionSizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBacktester:
    """Run backtests using trained model predictions."""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        """
        Initialize model backtester.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing processed data
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        
    def load_model_results(self, results_file: str) -> Dict[str, Any]:
        """Load training results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def create_predictions_from_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create prediction DataFrame from model training results.
        
        Args:
            results: Model training results
            
        Returns:
            DataFrame with predictions for backtesting
        """
        all_predictions = []
        
        # Extract predictions from each split
        for split_result in results['detailed_results']:
            split_id = split_result['split_id']
            
            # Load the actual split data to get dates and tickers
            split_dir = os.path.join("data/processed/splits", f"split_{split_id}")
            
            # Load test data
            test_X = pd.read_parquet(os.path.join(split_dir, "test_X.parquet"))
            test_y = pd.read_parquet(os.path.join(split_dir, "test_y.parquet"))
            
            # Get predictions from results
            test_predictions = split_result.get('predictions', {}).get('test', [])
            test_actuals = split_result.get('actuals', {}).get('test', [])
            
            if len(test_predictions) != len(test_actuals):
                logger.warning(f"Prediction/actual length mismatch in split {split_id}")
                continue
            
            # Create prediction records
            for i, (pred, actual) in enumerate(zip(test_predictions, test_actuals)):
                if i < len(test_X):
                    # Extract date and ticker from test data index or columns
                    row_data = test_X.iloc[i]
                    
                    # Try to get date and ticker from various possible sources
                    date = None
                    ticker = None
                    
                    if 'date' in test_X.columns:
                        date = row_data['date']
                    elif hasattr(test_X.index, 'get_level_values'):
                        # MultiIndex case
                        if 'date' in test_X.index.names:
                            date = test_X.index.get_level_values('date')[i]
                        if 'ticker' in test_X.index.names:
                            ticker = test_X.index.get_level_values('ticker')[i]
                    
                    if 'ticker' in test_X.columns:
                        ticker = row_data['ticker']
                    
                    # If we can't get date/ticker, skip this prediction
                    if date is None or ticker is None:
                        continue
                    
                    all_predictions.append({
                        'date': date,
                        'ticker': ticker,
                        'prediction': pred,
                        'actual': actual,
                        'split_id': split_id
                    })
        
        if not all_predictions:
            logger.error("No valid predictions found in results")
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(all_predictions)
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(predictions_df['date']):
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Sort by date and ticker
        predictions_df = predictions_df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        logger.info(f"Created {len(predictions_df)} predictions from model results")
        logger.info(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        logger.info(f"Tickers: {predictions_df['ticker'].nunique()} unique")
        
        return predictions_df
    
    def run_model_backtest(
        self,
        results_file: str,
        initial_capital: float = 1000000,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run backtest using model predictions.
        
        Args:
            results_file: Path to model results JSON file
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            max_position_size: Max position size as fraction
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest for model results: {results_file}")
        
        # Load model results
        model_results = self.load_model_results(results_file)
        
        # Create predictions DataFrame
        predictions = self.create_predictions_from_results(model_results)
        
        if predictions.empty:
            logger.error("No predictions available for backtesting")
            return {}
        
        # Load price data
        try:
            prices = pd.read_parquet(os.path.join(self.data_dir, "weekly_features.parquet"))
            
            # Ensure required columns exist
            if 'close' not in prices.columns and 'adj_close' in prices.columns:
                prices['close'] = prices['adj_close']
            
            logger.info(f"Loaded price data: {len(prices)} records")
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return {}
        
        # Initialize backtest engine
        trading_costs = TradingCosts(commission_rate=transaction_cost)
        position_sizer = PositionSizer(max_position_size=max_position_size)
        
        backtest_engine = BacktestEngine(
            trading_costs=trading_costs,
            position_sizer=position_sizer,
            initial_capital=initial_capital
        )
        
        # Run backtest
        try:
            backtest_results = backtest_engine.run_backtest(predictions, prices)
            
            # Add model metadata
            backtest_results['model_info'] = {
                'results_file': results_file,
                'model_type': model_results.get('training_config', {}).get('model_type', 'unknown'),
                'num_splits': len(model_results.get('detailed_results', [])),
                'predictions_count': len(predictions)
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}
    
    def run_all_model_backtests(self) -> Dict[str, Any]:
        """Run backtests for all available model results."""
        
        results_files = []
        
        # Find all result files
        for file in os.listdir(self.models_dir):
            if file.endswith('_results_') and file.endswith('.json'):
                results_files.append(os.path.join(self.models_dir, file))
        
        if not results_files:
            logger.warning(f"No model results files found in {self.models_dir}")
            return {}
        
        logger.info(f"Found {len(results_files)} model result files")
        
        all_backtest_results = {}
        
        for results_file in results_files:
            try:
                # Extract model name from filename
                filename = os.path.basename(results_file)
                model_name = filename.split('_results_')[0]
                
                logger.info(f"Running backtest for {model_name}")
                
                backtest_results = self.run_model_backtest(results_file)
                
                if backtest_results:
                    all_backtest_results[model_name] = backtest_results
                    
                    # Log key results
                    total_return = backtest_results.get('total_return', 0)
                    sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
                    max_drawdown = backtest_results.get('max_drawdown', 0)
                    
                    logger.info(f"{model_name} results: "
                               f"Return {total_return:.2%}, "
                               f"Sharpe {sharpe_ratio:.2f}, "
                               f"Max DD {max_drawdown:.2%}")
                else:
                    logger.error(f"Backtest failed for {model_name}")
                    
            except Exception as e:
                logger.error(f"Error processing {results_file}: {e}")
                continue
        
        return all_backtest_results
    
    def save_backtest_results(self, results: Dict[str, Any], output_file: str):
        """Save backtest results to file."""
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                k: v for k, v in model_results.items() 
                if k not in ['portfolio_history', 'trade_history']  # These are large
            }
            
            # Add summary of histories
            if 'portfolio_history' in model_results:
                portfolio_history = model_results['portfolio_history']
                if portfolio_history:
                    serializable_results[model_name]['portfolio_summary'] = {
                        'start_value': portfolio_history[0]['portfolio_value'],
                        'end_value': portfolio_history[-1]['portfolio_value'],
                        'num_periods': len(portfolio_history)
                    }
            
            if 'trade_history' in model_results:
                trade_history = model_results['trade_history']
                serializable_results[model_name]['trade_summary'] = {
                    'total_trades': len(trade_history),
                    'total_volume': sum(abs(trade['value']) for trade in trade_history)
                }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to: {output_file}")

def main():
    """Main function to run model backtests."""
    
    logger.info("=" * 60)
    logger.info("MODEL BACKTESTING")
    logger.info("=" * 60)
    
    # Initialize backtester
    backtester = ModelBacktester()
    
    # Run backtests for all models
    all_results = backtester.run_all_model_backtests()
    
    if not all_results:
        logger.error("No backtest results generated")
        return
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"models/backtest_results_{timestamp}.json"
    backtester.save_backtest_results(all_results, output_file)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    
    for model_name, results in all_results.items():
        logger.info(f"\\n{model_name.upper()}:")
        logger.info(f"  Total Return:     {results.get('total_return', 0):>8.2%}")
        logger.info(f"  Annualized Return:{results.get('annualized_return', 0):>8.2%}")
        logger.info(f"  Sharpe Ratio:     {results.get('sharpe_ratio', 0):>8.2f}")
        logger.info(f"  Max Drawdown:     {results.get('max_drawdown', 0):>8.2%}")
        logger.info(f"  Total Trades:     {results.get('total_trades', 0):>8.0f}")

if __name__ == "__main__":
    main()