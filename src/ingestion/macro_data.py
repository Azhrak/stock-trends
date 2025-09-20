"""
Macro-economic data ingestion using FRED API.
Downloads economic indicators and saves to parquet.
"""

import os
import sys
import pandas as pd
from fredapi import Fred
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from utils.env_utils import load_environment
    load_environment()
except ImportError:
    pass  # Continue without env loading if not available

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroDataIngestor:
    """Downloads and saves macro-economic data from FRED."""
    
    # Common economic indicators with their FRED series IDs
    DEFAULT_SERIES = {
        'cpi': 'CPIAUCSL',  # Consumer Price Index
        'unemployment': 'UNRATE',  # Unemployment Rate
        '10y_treasury': 'GS10',  # 10-Year Treasury Rate
        '3m_treasury': 'GS3M',  # 3-Month Treasury Rate
        'fed_funds': 'FEDFUNDS',  # Federal Funds Rate
        'gdp': 'GDP',  # Gross Domestic Product
        'industrial_production': 'INDPRO',  # Industrial Production Index
        'housing_starts': 'HOUST',  # Housing Starts
        'retail_sales': 'RSAFS',  # Retail Sales
        'consumer_sentiment': 'UMCSENT',  # University of Michigan Consumer Sentiment
        'vix': 'VIXCLS',  # CBOE Volatility Index
        'dollar_index': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index
    }
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/raw"):
        """
        Initialize the macro data ingestor.
        
        Args:
            api_key: FRED API key (can also be set via FRED_API_KEY env var)
            data_dir: Directory to save raw data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize FRED API
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
        
        if api_key is None:
            logger.warning("No FRED API key provided. Set FRED_API_KEY environment variable or pass api_key parameter.")
            logger.warning("You can get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            self.fred = None
        else:
            self.fred = Fred(api_key=api_key)
            logger.info("FRED API initialized successfully")
    
    def download_series(
        self, 
        series_ids: Dict[str, str],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download multiple economic series from FRED.
        
        Args:
            series_ids: Dictionary mapping names to FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a') - None uses original
            
        Returns:
            DataFrame with all series data
        """
        if self.fred is None:
            raise ValueError("FRED API not initialized. Please provide API key.")
        
        logger.info(f"Downloading {len(series_ids)} macro series from {start_date} to {end_date or 'today'}")
        
        all_series = {}
        
        for name, series_id in series_ids.items():
            try:
                logger.info(f"Downloading {name} ({series_id})")
                
                # Only pass frequency if it's specified
                if frequency is not None:
                    series = self.fred.get_series(
                        series_id,
                        start_date=start_date,
                        end_date=end_date,
                        frequency=frequency
                    )
                else:
                    series = self.fred.get_series(
                        series_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if not series.empty:
                    all_series[name] = series
                    logger.info(f"Downloaded {len(series)} observations for {name}")
                else:
                    logger.warning(f"No data returned for {name} ({series_id})")
                    
            except Exception as e:
                logger.error(f"Error downloading {name} ({series_id}): {str(e)}")
                continue
        
        if not all_series:
            raise ValueError("No series data was successfully downloaded")
        
        # Combine all series into a single DataFrame
        combined_df = pd.DataFrame(all_series)
        combined_df.index.name = 'date'
        combined_df = combined_df.reset_index()
        
        # Add metadata
        combined_df['ingestion_timestamp'] = datetime.now()
        combined_df['source'] = 'fred'
        
        logger.info(f"Combined {len(all_series)} series into DataFrame with {len(combined_df)} rows")
        return combined_df
    
    def calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived macro features.
        
        Args:
            data: DataFrame with macro data
            
        Returns:
            DataFrame with additional derived features
        """
        df = data.copy()
        
        # Yield spread (10Y - 3M)
        if '10y_treasury' in df.columns and '3m_treasury' in df.columns:
            df['yield_spread'] = df['10y_treasury'] - df['3m_treasury']
        
        # Real interest rate (Fed Funds - CPI YoY change)
        if 'fed_funds' in df.columns and 'cpi' in df.columns:
            df['cpi_yoy'] = df['cpi'].pct_change(periods=12) * 100  # Annual inflation
            df['real_fed_funds'] = df['fed_funds'] - df['cpi_yoy']
        
        # GDP growth rate (quarterly YoY)
        if 'gdp' in df.columns:
            df['gdp_yoy'] = df['gdp'].pct_change(periods=4) * 100  # Annual growth
        
        # Industrial production momentum
        if 'industrial_production' in df.columns:
            df['indpro_mom'] = df['industrial_production'].pct_change() * 100  # Monthly change
            df['indpro_yoy'] = df['industrial_production'].pct_change(periods=12) * 100  # Annual change
        
        logger.info("Calculated derived macro features")
        return df
    
    def save_to_parquet(self, data: pd.DataFrame, filename: str = "macro.parquet"):
        """
        Save macro data to parquet file.
        
        Args:
            data: DataFrame with macro data
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Ensure date column is datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date
            data = data.sort_values('date')
            
            # Save to parquet
            data.to_parquet(filepath, index=False, engine='pyarrow')
            logger.info(f"Saved macro data to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            raise
    
    def download_default_series(
        self,
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download the default set of macro-economic indicators.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with default macro series
        """
        return self.download_series(
            series_ids=self.DEFAULT_SERIES,
            start_date=start_date,
            end_date=end_date
        )

def main():
    """Example usage of the macro data ingestor."""
    
    # Check for API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("FRED API key not found!")
        print("Please set the FRED_API_KEY environment variable.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\nExample: export FRED_API_KEY='your_api_key_here'")
        return
    
    # Initialize ingestor
    ingestor = MacroDataIngestor(api_key=api_key)
    
    # Download default macro series
    macro_data = ingestor.download_default_series(
        start_date="2000-01-01"
    )
    
    # Calculate derived features
    macro_data = ingestor.calculate_derived_features(macro_data)
    
    # Save to file
    ingestor.save_to_parquet(macro_data)
    
    print(f"Successfully downloaded and saved macro data")
    print(f"Data shape: {macro_data.shape}")
    print(f"Date range: {macro_data['date'].min()} to {macro_data['date'].max()}")
    print(f"Available series: {[col for col in macro_data.columns if col not in ['date', 'ingestion_timestamp', 'source']]}")

if __name__ == "__main__":
    main()