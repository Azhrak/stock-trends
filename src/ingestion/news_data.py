"""
News and alternative data ingestion.
Downloads news data, earnings transcripts, and Google Trends data.
"""

import os
import pandas as pd
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import time
import logging
from pytrends.request import TrendReq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataIngestor:
    """Downloads and saves news and alternative data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the news data ingestor.
        
        Args:
            data_dir: Directory to save raw data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "news"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "trends"), exist_ok=True)
    
    def download_finnhub_news(
        self,
        api_key: str,
        symbols: List[str],
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Download company news from Finnhub API.
        
        Args:
            api_key: Finnhub API key
            symbols: List of stock symbols
            days_back: Number of days to look back for news
            
        Returns:
            DataFrame with news articles
        """
        base_url = "https://finnhub.io/api/v1/company-news"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        all_news = []
        
        logger.info(f"Downloading news for {len(symbols)} symbols from {from_date} to {to_date}")
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching news for {symbol}")
                
                params = {
                    'symbol': symbol,
                    'from': from_date,
                    'to': to_date,
                    'token': api_key
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                news_data = response.json()
                
                if news_data:
                    for article in news_data:
                        article['symbol'] = symbol
                        article['ingestion_timestamp'] = datetime.now()
                        article['source'] = 'finnhub'
                        all_news.append(article)
                
                # Rate limiting - Finnhub free tier has limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {str(e)}")
                continue
        
        if not all_news:
            logger.warning("No news data retrieved")
            return pd.DataFrame()
        
        news_df = pd.DataFrame(all_news)
        
        # Convert datetime columns
        if 'datetime' in news_df.columns:
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
        
        logger.info(f"Retrieved {len(news_df)} news articles")
        return news_df
    
    def download_google_trends(
        self,
        keywords: List[str],
        timeframe: str = "2020-01-01 2024-12-31",
        geo: str = "US",
        category: int = 0
    ) -> pd.DataFrame:
        """
        Download Google Trends data for given keywords.
        
        Args:
            keywords: List of search terms
            timeframe: Date range (e.g., "2020-01-01 2024-12-31")
            geo: Geographic location code
            category: Category code (0 for all categories)
            
        Returns:
            DataFrame with search trends
        """
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            
            all_trends = []
            
            # Google Trends API allows max 5 keywords per request
            keyword_chunks = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
            
            for chunk in keyword_chunks:
                logger.info(f"Downloading trends for: {chunk}")
                
                try:
                    pytrends.build_payload(
                        chunk, 
                        cat=category, 
                        timeframe=timeframe, 
                        geo=geo, 
                        gprop=''
                    )
                    
                    # Get interest over time
                    interest_df = pytrends.interest_over_time()
                    
                    if not interest_df.empty:
                        # Remove the 'isPartial' column if it exists
                        if 'isPartial' in interest_df.columns:
                            interest_df = interest_df.drop('isPartial', axis=1)
                        
                        # Reshape to long format
                        interest_df = interest_df.reset_index()
                        interest_df = pd.melt(
                            interest_df,
                            id_vars=['date'],
                            var_name='keyword',
                            value_name='search_interest'
                        )
                        
                        all_trends.append(interest_df)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading trends for {chunk}: {str(e)}")
                    time.sleep(5)  # Longer wait on error
                    continue
            
            if not all_trends:
                logger.warning("No trends data retrieved")
                return pd.DataFrame()
            
            # Combine all trends data
            trends_df = pd.concat(all_trends, ignore_index=True)
            
            # Add metadata
            trends_df['ingestion_timestamp'] = datetime.now()
            trends_df['source'] = 'google_trends'
            trends_df['geo'] = geo
            
            logger.info(f"Retrieved trends data for {len(trends_df)} keyword-date combinations")
            return trends_df
            
        except Exception as e:
            logger.error(f"Error downloading Google Trends data: {str(e)}")
            raise
    
    def get_finance_keywords(self, tickers: List[str]) -> List[str]:
        """
        Generate finance-related keywords for trends analysis.
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            List of keywords for trends search
        """
        # General finance keywords
        finance_keywords = [
            "stock market",
            "recession",
            "inflation",
            "federal reserve",
            "interest rates",
            "unemployment",
            "GDP",
            "earnings",
            "dividend",
            "market crash"
        ]
        
        # Add ticker-specific keywords (limited selection for API limits)
        ticker_keywords = []
        for ticker in tickers[:20]:  # Limit to avoid API quotas
            ticker_keywords.extend([
                f"{ticker} stock",
                f"{ticker} earnings",
                f"{ticker} news"
            ])
        
        all_keywords = finance_keywords + ticker_keywords
        return all_keywords[:50]  # Limit total keywords for API efficiency
    
    def save_news_to_parquet(self, data: pd.DataFrame, filename: str = "news.parquet"):
        """
        Save news data to parquet file.
        
        Args:
            data: DataFrame with news data
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, "news", filename)
        
        try:
            if not data.empty:
                # Ensure datetime columns
                if 'datetime' in data.columns:
                    data['datetime'] = pd.to_datetime(data['datetime'])
                
                # Sort by symbol and datetime
                if 'symbol' in data.columns and 'datetime' in data.columns:
                    data = data.sort_values(['symbol', 'datetime'])
                
                data.to_parquet(filepath, index=False, engine='pyarrow')
                logger.info(f"Saved news data to {filepath}")
            else:
                logger.warning(f"No data to save for {filename}")
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving news data: {str(e)}")
            raise
    
    def save_trends_to_parquet(self, data: pd.DataFrame, filename: str = "trends.parquet"):
        """
        Save trends data to parquet file.
        
        Args:
            data: DataFrame with trends data
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, "trends", filename)
        
        try:
            if not data.empty:
                # Ensure date column is datetime
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                
                # Sort by keyword and date
                data = data.sort_values(['keyword', 'date'])
                
                data.to_parquet(filepath, index=False, engine='pyarrow')
                logger.info(f"Saved trends data to {filepath}")
            else:
                logger.warning(f"No data to save for {filename}")
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving trends data: {str(e)}")
            raise

def main():
    """Example usage of the news data ingestor."""
    
    # Initialize ingestor
    ingestor = NewsDataIngestor()
    
    # Example tickers
    example_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Download Google Trends data (this works without API key)
    try:
        logger.info("Downloading Google Trends data...")
        keywords = ingestor.get_finance_keywords(example_tickers)
        
        trends_data = ingestor.download_google_trends(
            keywords=keywords[:10],  # Limit for demo
            timeframe="2023-01-01 2024-12-31"
        )
        
        if not trends_data.empty:
            ingestor.save_trends_to_parquet(trends_data)
            print(f"Successfully downloaded trends data: {trends_data.shape}")
        else:
            print("No trends data retrieved")
            
    except Exception as e:
        logger.error(f"Error with trends data: {str(e)}")
    
    # For Finnhub news, you need an API key
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    if finnhub_api_key:
        try:
            logger.info("Downloading Finnhub news data...")
            news_data = ingestor.download_finnhub_news(
                api_key=finnhub_api_key,
                symbols=example_tickers,
                days_back=7  # Last week only for demo
            )
            
            if not news_data.empty:
                ingestor.save_news_to_parquet(news_data)
                print(f"Successfully downloaded news data: {news_data.shape}")
            else:
                print("No news data retrieved")
                
        except Exception as e:
            logger.error(f"Error with news data: {str(e)}")
    else:
        print("FINNHUB_API_KEY not set - skipping news download")
        print("Get a free key at: https://finnhub.io/")

if __name__ == "__main__":
    main()