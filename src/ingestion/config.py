# Stock Market Data Ingestion Configuration

# Default tickers for testing (subset of S&P 500)
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO',
    'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'VZ', 'BMY', 'NEE',
    'NFLX', 'ADBE', 'CRM', 'LLY', 'XOM', 'MRK', 'ACN', 'NOW', 'ORCL', 'INTC'
]

# Date ranges
EQUITY_START_DATE = "2010-01-01"
MACRO_START_DATE = "2000-01-01"

# API Configuration
# Set these as environment variables:
# export FRED_API_KEY="your_fred_api_key"
# export FINNHUB_API_KEY="your_finnhub_api_key"

# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Ingestion settings
MAX_TICKERS_FOR_NEWS = 20  # Limit for API efficiency
NEWS_LOOKBACK_DAYS = 30
TRENDS_TIMEFRAME = "2020-01-01 2024-12-31"

# Google Trends keywords
FINANCE_KEYWORDS = [
    "stock market", "recession", "inflation", "federal reserve", 
    "interest rates", "unemployment", "GDP", "earnings", "dividend"
]