"""
Main data ingestion coordinator.
Orchestrates all data downloads and provides a unified interface.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ingestion.equity_prices import EquityPriceIngestor
from ingestion.macro_data import MacroDataIngestor
from ingestion.news_data import NewsDataIngestor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestionCoordinator:
    """Coordinates all data ingestion processes."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data ingestion coordinator.
        
        Args:
            data_dir: Directory to save raw data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize ingestors
        self.equity_ingestor = EquityPriceIngestor(data_dir)
        self.macro_ingestor = MacroDataIngestor(data_dir=data_dir)
        self.news_ingestor = NewsDataIngestor(data_dir)
    
    def ingest_equity_data(
        self,
        tickers: Optional[List[str]] = None,
        num_tickers: int = 50,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest equity price data.
        
        Args:
            tickers: Specific tickers to download (overrides num_tickers)
            num_tickers: Number of S&P 500 tickers to download
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info("Starting equity data ingestion...")
        
        try:
            # Get tickers if not provided
            if tickers is None:
                tickers = self.equity_ingestor.get_sp500_tickers(limit=num_tickers)
            
            # Download price data
            price_data = self.equity_ingestor.download_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            # Save to file
            filepath = self.equity_ingestor.save_to_parquet(price_data)
            
            result = {
                'status': 'success',
                'num_tickers': len(tickers),
                'num_records': len(price_data),
                'date_range': (price_data['date'].min(), price_data['date'].max()),
                'filepath': filepath,
                'tickers': tickers
            }
            
            logger.info(f"Equity ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Equity ingestion failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def ingest_macro_data(
        self,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest macro-economic data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            api_key: FRED API key
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info("Starting macro data ingestion...")
        
        try:
            # Set API key if provided
            if api_key:
                self.macro_ingestor.fred = None  # Reset
                self.macro_ingestor.__init__(api_key=api_key, data_dir=self.data_dir)
            
            # Check if FRED API is available
            if self.macro_ingestor.fred is None:
                return {
                    'status': 'skipped',
                    'reason': 'No FRED API key available. Set FRED_API_KEY environment variable.'
                }
            
            # Download macro data
            macro_data = self.macro_ingestor.download_default_series(
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate derived features
            macro_data = self.macro_ingestor.calculate_derived_features(macro_data)
            
            # Save to file
            filepath = self.macro_ingestor.save_to_parquet(macro_data)
            
            result = {
                'status': 'success',
                'num_series': len([col for col in macro_data.columns 
                                 if col not in ['date', 'ingestion_timestamp', 'source']]),
                'num_records': len(macro_data),
                'date_range': (macro_data['date'].min(), macro_data['date'].max()),
                'filepath': filepath
            }
            
            logger.info(f"Macro ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Macro ingestion failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def ingest_news_data(
        self,
        tickers: List[str],
        days_back: int = 30,
        finnhub_api_key: Optional[str] = None,
        include_trends: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest news and alternative data.
        
        Args:
            tickers: List of tickers for news
            days_back: Number of days to look back for news
            finnhub_api_key: Finnhub API key
            include_trends: Whether to include Google Trends data
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info("Starting news data ingestion...")
        
        results = {}
        
        # Google Trends data (free)
        if include_trends:
            try:
                keywords = self.news_ingestor.get_finance_keywords(tickers)
                
                trends_data = self.news_ingestor.download_google_trends(
                    keywords=keywords[:20],  # Limit for API efficiency
                    timeframe="2020-01-01 2024-12-31"
                )
                
                if not trends_data.empty:
                    trends_filepath = self.news_ingestor.save_trends_to_parquet(trends_data)
                    
                    results['trends'] = {
                        'status': 'success',
                        'num_keywords': len(trends_data['keyword'].unique()),
                        'num_records': len(trends_data),
                        'filepath': trends_filepath
                    }
                else:
                    results['trends'] = {'status': 'no_data'}
                    
            except Exception as e:
                logger.error(f"Trends ingestion failed: {str(e)}")
                results['trends'] = {'status': 'failed', 'error': str(e)}
        
        # Finnhub news data (requires API key)
        if finnhub_api_key:
            try:
                news_data = self.news_ingestor.download_finnhub_news(
                    api_key=finnhub_api_key,
                    symbols=tickers,
                    days_back=days_back
                )
                
                if not news_data.empty:
                    news_filepath = self.news_ingestor.save_news_to_parquet(news_data)
                    
                    results['news'] = {
                        'status': 'success',
                        'num_articles': len(news_data),
                        'num_symbols': len(news_data['symbol'].unique()),
                        'filepath': news_filepath
                    }
                else:
                    results['news'] = {'status': 'no_data'}
                    
            except Exception as e:
                logger.error(f"News ingestion failed: {str(e)}")
                results['news'] = {'status': 'failed', 'error': str(e)}
        else:
            results['news'] = {
                'status': 'skipped',
                'reason': 'No Finnhub API key provided'
            }
        
        logger.info(f"News ingestion completed: {results}")
        return results
    
    def run_full_ingestion(
        self,
        num_tickers: int = 50,
        equity_start_date: str = "2010-01-01",
        macro_start_date: str = "2000-01-01",
        news_days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Run complete data ingestion pipeline.
        
        Args:
            num_tickers: Number of tickers to download
            equity_start_date: Start date for equity data
            macro_start_date: Start date for macro data
            news_days_back: Days to look back for news
            
        Returns:
            Dictionary with all ingestion results
        """
        logger.info("Starting full data ingestion pipeline...")
        
        results = {
            'timestamp': datetime.now(),
            'equity': {},
            'macro': {},
            'news': {}
        }
        
        # 1. Equity data
        equity_result = self.ingest_equity_data(
            num_tickers=num_tickers,
            start_date=equity_start_date
        )
        results['equity'] = equity_result
        
        # Get tickers for news data
        tickers = equity_result.get('tickers', [])
        
        # 2. Macro data
        macro_result = self.ingest_macro_data(
            start_date=macro_start_date
        )
        results['macro'] = macro_result
        
        # 3. News data
        news_result = self.ingest_news_data(
            tickers=tickers[:20],  # Limit for API efficiency
            days_back=news_days_back,
            finnhub_api_key=os.getenv('FINNHUB_API_KEY'),
            include_trends=True
        )
        results['news'] = news_result
        
        logger.info("Full ingestion pipeline completed")
        return results

def main():
    """Command-line interface for data ingestion."""
    
    parser = argparse.ArgumentParser(description='Stock prediction data ingestion')
    parser.add_argument('--data-dir', default='data/raw', help='Data directory')
    parser.add_argument('--num-tickers', type=int, default=50, help='Number of tickers')
    parser.add_argument('--equity-start', default='2010-01-01', help='Equity data start date')
    parser.add_argument('--macro-start', default='2000-01-01', help='Macro data start date')
    parser.add_argument('--news-days', type=int, default=30, help='News lookback days')
    parser.add_argument('--equity-only', action='store_true', help='Download equity data only')
    parser.add_argument('--macro-only', action='store_true', help='Download macro data only')
    parser.add_argument('--news-only', action='store_true', help='Download news data only')
    
    args = parser.parse_args()
    
    # Initialize coordinator
    coordinator = DataIngestionCoordinator(data_dir=args.data_dir)
    
    if args.equity_only:
        result = coordinator.ingest_equity_data(
            num_tickers=args.num_tickers,
            start_date=args.equity_start
        )
        print(f"Equity ingestion result: {result}")
        
    elif args.macro_only:
        result = coordinator.ingest_macro_data(
            start_date=args.macro_start
        )
        print(f"Macro ingestion result: {result}")
        
    elif args.news_only:
        # Need to get tickers first
        tickers = coordinator.equity_ingestor.get_sp500_tickers(limit=20)
        result = coordinator.ingest_news_data(
            tickers=tickers,
            days_back=args.news_days,
            finnhub_api_key=os.getenv('FINNHUB_API_KEY')
        )
        print(f"News ingestion result: {result}")
        
    else:
        # Run full pipeline
        results = coordinator.run_full_ingestion(
            num_tickers=args.num_tickers,
            equity_start_date=args.equity_start,
            macro_start_date=args.macro_start,
            news_days_back=args.news_days
        )
        
        print("\n" + "="*60)
        print("DATA INGESTION SUMMARY")
        print("="*60)
        
        for data_type, result in results.items():
            if data_type == 'timestamp':
                continue
            print(f"\n{data_type.upper()} DATA:")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    main()