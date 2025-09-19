"""
Equity price data ingestion using yfinance.
Downloads daily OHLCV data for specified tickers and saves to parquet.
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime, date
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EquityPriceIngestor:
    """Downloads and saves equity price data from Yahoo Finance."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the price data ingestor.
        
        Args:
            data_dir: Directory to save raw data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_prices(
        self, 
        tickers: List[str],
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Download daily price data for given tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            interval: Data interval ('1d', '1wk', '1mo', etc.)
            auto_adjust: Whether to auto-adjust prices for splits/dividends
            
        Returns:
            DataFrame with price data
        """
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        
        logger.info(f"Downloading price data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # Download data for all tickers
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                group_by='ticker',
                threads=True,
                progress=True
            )
            
            # Reshape data for easier processing
            if len(tickers) == 1:
                # Single ticker case - yfinance returns different structure
                data.columns.name = 'price_type'
                data = data.stack().unstack(level=0)
                data.index.names = ['date', 'ticker']
                data = data.reset_index()
                data['ticker'] = tickers[0]
            else:
                # Multiple tickers case
                data = data.stack(level=0).reset_index()
                data.columns.name = None
                data = data.rename(columns={'level_1': 'ticker'})
            
            # Ensure consistent column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add metadata
            data['ingestion_timestamp'] = datetime.now()
            data['source'] = 'yahoo_finance'
            
            logger.info(f"Downloaded {len(data)} records for {len(tickers)} tickers")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading price data: {str(e)}")
            raise
    
    def save_to_parquet(self, data: pd.DataFrame, filename: str = "prices_daily.parquet"):
        """
        Save price data to parquet file.
        
        Args:
            data: DataFrame with price data
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Ensure date column is datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort by ticker and date
            data = data.sort_values(['ticker', 'date'])
            
            # Save to parquet
            data.to_parquet(filepath, index=False, engine='pyarrow')
            logger.info(f"Saved price data to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            raise
    
    def get_sp500_tickers(self, limit: Optional[int] = None) -> List[str]:
        """
        Get S&P 500 ticker list from Wikipedia.
        
        Args:
            limit: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        try:
            # Get S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (some have dots that need to be replaced)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            if limit:
                tickers = tickers[:limit]
            
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            # Fallback to hardcoded list
            fallback_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
                'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO',
                'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'VZ', 'BMY', 'NEE'
            ]
            return fallback_tickers[:limit] if limit else fallback_tickers

def main():
    """Example usage of the equity price ingestor."""
    
    # Initialize ingestor
    ingestor = EquityPriceIngestor()
    
    # Get some tickers (limiting to 50 for initial testing)
    tickers = ingestor.get_sp500_tickers(limit=50)
    
    # Download price data
    price_data = ingestor.download_prices(
        tickers=tickers,
        start_date="2010-01-01",
        end_date=None  # defaults to today
    )
    
    # Save to file
    ingestor.save_to_parquet(price_data)
    
    print(f"Successfully downloaded and saved price data for {len(tickers)} tickers")
    print(f"Data shape: {price_data.shape}")
    print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")

if __name__ == "__main__":
    main()