"""
Backtesting framework for stock prediction models.
Implements realistic trading simulation with transaction costs and position sizing.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingCosts:
    """Calculate realistic trading costs."""
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        bid_ask_spread: float = 0.0005,  # 0.05% bid-ask spread
        market_impact: float = 0.0002,   # 0.02% market impact
        slippage: float = 0.0001         # 0.01% slippage
    ):
        """
        Initialize trading costs.
        
        Args:
            commission_rate: Commission as fraction of trade value
            bid_ask_spread: Bid-ask spread as fraction
            market_impact: Market impact as fraction  
            slippage: Slippage as fraction
        """
        self.commission_rate = commission_rate
        self.bid_ask_spread = bid_ask_spread
        self.market_impact = market_impact
        self.slippage = slippage
        
        # Total transaction cost
        self.total_cost = commission_rate + bid_ask_spread + market_impact + slippage
        
        logger.info(f"Trading costs initialized - Total: {self.total_cost:.4f} ({self.total_cost*100:.2f}%)")
    
    def calculate_cost(self, trade_value: float) -> float:
        """Calculate total trading cost for a trade."""
        return trade_value * self.total_cost

class PositionSizer:
    """Handle position sizing and risk management."""
    
    def __init__(
        self,
        max_position_size: float = 0.05,    # Max 5% per position
        max_total_exposure: float = 1.0,    # Max 100% total exposure
        min_position_size: float = 0.001,   # Min 0.1% position
        volatility_target: float = 0.15     # 15% annual vol target
    ):
        """
        Initialize position sizer.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_total_exposure: Maximum total exposure as fraction
            min_position_size: Minimum position size as fraction
            volatility_target: Target portfolio volatility
        """
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.min_position_size = min_position_size
        self.volatility_target = volatility_target
        
        logger.info(f"Position sizing: Max {max_position_size:.1%}, "
                   f"Total exposure {max_total_exposure:.1%}, "
                   f"Vol target {volatility_target:.1%}")
    
    def calculate_position_size(
        self,
        prediction: float,
        confidence: float,
        stock_volatility: float,
        current_exposure: float
    ) -> float:
        """
        Calculate position size based on prediction and risk parameters.
        
        Args:
            prediction: Model prediction (expected return)
            confidence: Model confidence (0-1)
            stock_volatility: Stock's historical volatility
            current_exposure: Current total portfolio exposure
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base position size from prediction strength
        base_size = abs(prediction) * confidence
        
        # Scale by volatility (inverse vol weighting)
        if stock_volatility > 0:
            vol_adjusted_size = base_size * (self.volatility_target / stock_volatility)
        else:
            vol_adjusted_size = base_size
        
        # Apply position limits
        position_size = np.clip(vol_adjusted_size, self.min_position_size, self.max_position_size)
        
        # Check total exposure constraint
        if current_exposure + position_size > self.max_total_exposure:
            position_size = max(0, self.max_total_exposure - current_exposure)
        
        return position_size

class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(
        self,
        trading_costs: Optional[TradingCosts] = None,
        position_sizer: Optional[PositionSizer] = None,
        initial_capital: float = 1000000.0,  # $1M
        rebalance_frequency: str = 'weekly'
    ):
        """
        Initialize backtest engine.
        
        Args:
            trading_costs: Trading costs configuration
            position_sizer: Position sizing configuration
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance ('weekly', 'monthly')
        """
        self.trading_costs = trading_costs or TradingCosts()
        self.position_sizer = position_sizer or PositionSizer()
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        
        # Portfolio state
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> shares
        self.position_values = {}  # ticker -> dollar value
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.portfolio_history = []
        
        logger.info(f"Backtest engine initialized with ${initial_capital:,.0f} capital")
    
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """
        Load model predictions from file.
        
        Args:
            predictions_file: Path to predictions file
            
        Returns:
            DataFrame with predictions
        """
        if predictions_file.endswith('.json'):
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame (assuming specific format)
            # This will need to be adapted based on actual prediction format
            predictions = pd.DataFrame(data)
        else:
            predictions = pd.read_parquet(predictions_file)
        
        logger.info(f"Loaded {len(predictions)} predictions from {predictions_file}")
        return predictions
    
    def load_price_data(self, data_dir: str = "data/processed") -> pd.DataFrame:
        """
        Load historical price data for backtesting.
        
        Args:
            data_dir: Directory containing price data
            
        Returns:
            DataFrame with price data
        """
        price_file = os.path.join(data_dir, "weekly_features.parquet")
        
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price data not found: {price_file}")
        
        # Load and prepare price data
        prices = pd.read_parquet(price_file)
        
        # Ensure we have required columns
        required_cols = ['ticker', 'date', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in prices.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in price data: {missing_cols}")
        
        # Sort by date and ticker
        prices = prices.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        logger.info(f"Loaded price data: {len(prices)} records, "
                   f"{prices['ticker'].nunique()} tickers, "
                   f"Date range: {prices['date'].min()} to {prices['date'].max()}")
        
        return prices
    
    def calculate_stock_volatility(self, prices: pd.DataFrame, window: int = 26) -> pd.DataFrame:
        """
        Calculate rolling volatility for each stock.
        
        Args:
            prices: Price data
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility data
        """
        volatilities = []
        
        for ticker in prices['ticker'].unique():
            ticker_data = prices[prices['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            # Calculate returns
            ticker_data['returns'] = ticker_data['close'].pct_change()
            
            # Calculate rolling volatility (annualized)
            ticker_data['volatility'] = (
                ticker_data['returns'].rolling(window=window).std() * np.sqrt(52)
            )
            
            volatilities.append(ticker_data[['ticker', 'date', 'volatility']])
        
        volatility_df = pd.concat(volatilities, ignore_index=True)
        return volatility_df
    
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the main backtesting simulation.
        
        Args:
            predictions: Model predictions
            prices: Historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest simulation...")
        
        # Filter data by date range
        if start_date:
            predictions = predictions[predictions['date'] >= start_date]
            prices = prices[prices['date'] >= start_date]
        
        if end_date:
            predictions = predictions[predictions['date'] <= end_date]
            prices = prices[prices['date'] <= end_date]
        
        # Calculate volatilities
        volatilities = self.calculate_stock_volatility(prices)
        
        # Get unique trading dates
        trading_dates = sorted(predictions['date'].unique())
        logger.info(f"Backtesting {len(trading_dates)} periods from {trading_dates[0]} to {trading_dates[-1]}")
        
        # Run simulation
        for i, date in enumerate(trading_dates):
            self._simulate_trading_period(date, predictions, prices, volatilities)
            
            # Log progress
            if (i + 1) % 50 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates)
                logger.info(f"Progress: {i+1}/{len(trading_dates)} ({progress:.1%}) - "
                           f"Portfolio value: ${self.portfolio_value:,.0f}")
        
        # Calculate final performance metrics
        results = self._calculate_performance_metrics()
        
        logger.info("Backtest completed successfully")
        return results
    
    def _simulate_trading_period(
        self,
        date: str,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        volatilities: pd.DataFrame
    ):
        """Simulate trading for a single period."""
        
        # Get predictions for this date
        period_predictions = predictions[predictions['date'] == date]
        
        # Get prices for this date
        period_prices = prices[prices['date'] == date]
        
        # Get volatilities for this date  
        period_vols = volatilities[volatilities['date'] == date]
        
        if len(period_predictions) == 0:
            return
        
        # Update portfolio values with current prices
        self._update_portfolio_values(period_prices)
        
        # Calculate current total exposure
        current_exposure = sum(abs(val) for val in self.position_values.values()) / self.portfolio_value
        
        # Process each prediction
        for _, pred_row in period_predictions.iterrows():
            ticker = pred_row['ticker']
            prediction = pred_row.get('prediction', 0)
            
            # Get stock price and volatility
            stock_price_data = period_prices[period_prices['ticker'] == ticker]
            stock_vol_data = period_vols[period_vols['ticker'] == ticker]
            
            if len(stock_price_data) == 0:
                continue
            
            stock_price = stock_price_data.iloc[0]['close']
            stock_vol = stock_vol_data.iloc[0]['volatility'] if len(stock_vol_data) > 0 else 0.15
            
            # Calculate position size
            confidence = min(abs(prediction) / 0.1, 1.0)  # Normalize prediction to confidence
            position_size = self.position_sizer.calculate_position_size(
                prediction, confidence, stock_vol, current_exposure
            )
            
            # Execute trade
            self._execute_trade(ticker, prediction, position_size, stock_price, date)
            
            # Update exposure
            current_exposure = sum(abs(val) for val in self.position_values.values()) / self.portfolio_value
        
        # Record portfolio state
        self._record_portfolio_state(date)
    
    def _update_portfolio_values(self, prices: pd.DataFrame):
        """Update portfolio values based on current prices."""
        total_position_value = 0
        
        for ticker, shares in self.positions.items():
            ticker_price_data = prices[prices['ticker'] == ticker]
            
            if len(ticker_price_data) > 0:
                current_price = ticker_price_data.iloc[0]['close']
                position_value = shares * current_price
                self.position_values[ticker] = position_value
                total_position_value += position_value
            else:
                # If no price data, assume position value unchanged
                total_position_value += self.position_values.get(ticker, 0)
        
        self.portfolio_value = self.cash + total_position_value
    
    def _execute_trade(
        self,
        ticker: str,
        prediction: float,
        position_size: float,
        stock_price: float,
        date: str
    ):
        """Execute a trade for a specific stock."""
        
        if position_size < self.position_sizer.min_position_size:
            return
        
        # Calculate target position value
        target_value = position_size * self.portfolio_value * np.sign(prediction)
        
        # Calculate current position value
        current_value = self.position_values.get(ticker, 0)
        
        # Calculate trade size
        trade_value = target_value - current_value
        
        if abs(trade_value) < self.initial_capital * 0.001:  # Min trade size
            return
        
        # Calculate shares to trade
        shares_to_trade = trade_value / stock_price
        
        # Calculate trading costs
        trading_cost = self.trading_costs.calculate_cost(abs(trade_value))
        
        # Check if we have enough cash
        cash_needed = trade_value + trading_cost
        
        if cash_needed > self.cash:
            # Scale down trade to available cash
            available_cash = self.cash - trading_cost
            if available_cash <= 0:
                return
            
            trade_value = available_cash
            shares_to_trade = trade_value / stock_price
        
        # Execute trade
        self.positions[ticker] = self.positions.get(ticker, 0) + shares_to_trade
        self.position_values[ticker] = self.positions[ticker] * stock_price
        self.cash -= (trade_value + trading_cost)
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'ticker': ticker,
            'shares': shares_to_trade,
            'price': stock_price,
            'value': trade_value,
            'cost': trading_cost,
            'prediction': prediction,
            'position_size': position_size
        })
    
    def _record_portfolio_state(self, date: str):
        """Record current portfolio state."""
        total_position_value = sum(self.position_values.values())
        
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': total_position_value,
            'num_positions': len([v for v in self.position_values.values() if abs(v) > 100])
        })
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate final performance metrics."""
        
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.sort_values('date')
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Performance metrics
        total_return = (self.portfolio_value / self.initial_capital) - 1
        
        # Annualized metrics
        days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
        years = days / 365.25
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1/years) - 1
            annualized_vol = portfolio_df['returns'].std() * np.sqrt(52)  # Weekly data
        else:
            annualized_return = 0
            annualized_vol = 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum drawdown
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cummax']) - 1
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trading stats
        total_trades = len(self.trade_history)
        total_trading_costs = sum(trade['cost'] for trade in self.trade_history)
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.portfolio_value,
            'total_trades': total_trades,
            'total_trading_costs': total_trading_costs,
            'trading_cost_pct': total_trading_costs / self.initial_capital,
            'period_days': days,
            'portfolio_history': portfolio_df.to_dict('records'),
            'trade_history': self.trade_history
        }
        
        return results

def main():
    """Example usage of the backtesting framework."""
    
    # Initialize backtest engine
    backtest = BacktestEngine(initial_capital=1000000)
    
    # This would typically load actual model predictions
    # For demo, we'll create dummy data
    logger.info("Demo: Creating dummy predictions...")
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='W')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    dummy_predictions = []
    for date in dates:
        for ticker in tickers:
            dummy_predictions.append({
                'date': date,
                'ticker': ticker,
                'prediction': np.random.normal(0, 0.05)  # Random predictions
            })
    
    predictions_df = pd.DataFrame(dummy_predictions)
    
    logger.info("Note: In real usage, load actual price data and predictions")
    logger.info("Backtesting framework is ready for integration with model predictions")

if __name__ == "__main__":
    main()