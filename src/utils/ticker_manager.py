"""
Ticker management utilities for the stock trends prediction project.
Handles loading, saving, and validating stock ticker lists.
"""

import os
from pathlib import Path
from typing import List, Set, Optional
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

class TickerManager:
    """Manages stock ticker configuration and validation."""
    
    def __init__(self, config_file: str = "config/tickers.txt"):
        """
        Initialize ticker manager.
        
        Args:
            config_file: Path to the ticker configuration file
        """
        self.config_file = Path(config_file)
        self.project_root = Path(__file__).parent.parent.parent
        self.full_config_path = self.project_root / self.config_file
        
        # Ensure config directory exists
        self.full_config_path.parent.mkdir(exist_ok=True)
        
        # Create default config if it doesn't exist
        if not self.full_config_path.exists():
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default ticker configuration file."""
        # First, try to copy from example file
        example_file = self.full_config_path.parent / "tickers.example.txt"
        
        if example_file.exists():
            logger.info(f"Creating {self.config_file} from example file")
            try:
                # Copy content from example file
                content = example_file.read_text()
                self.full_config_path.write_text(content)
                logger.info(f"Created ticker config from {example_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to copy from example file: {e}")
        
        # Fallback to programmatic defaults
        try:
            # Try to import from config.py for comprehensive defaults
            from ingestion.config import DEFAULT_TICKERS
            default_tickers = DEFAULT_TICKERS
            logger.info(f"Using {len(DEFAULT_TICKERS)} default tickers from config.py")
        except ImportError:
            # Fallback to minimal list if config.py not available
            default_tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"
            ]
            logger.info("Using fallback default tickers")
        
        self.save_tickers(default_tickers)
        logger.info(f"Created default ticker config at {self.full_config_path}")
    
    def load_tickers(self) -> List[str]:
        """
        Load tickers from configuration file.
        
        Returns:
            List of ticker symbols
        """
        try:
            with open(self.full_config_path, 'r') as f:
                tickers = []
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        tickers.append(line.upper())
                
                if not tickers:
                    logger.warning("No tickers found in config file, using defaults")
                    return ["AAPL", "MSFT", "GOOGL", "AMZN"]
                
                return tickers
                
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.full_config_path}, creating default")
            self._create_default_config()
            return self.load_tickers()
        except Exception as e:
            logger.error(f"Error reading ticker config: {e}")
            return ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Safe fallback
    
    def save_tickers(self, tickers: List[str]):
        """
        Save tickers to configuration file.
        
        Args:
            tickers: List of ticker symbols to save
        """
        # Validate and clean tickers
        cleaned_tickers = []
        for ticker in tickers:
            ticker = ticker.strip().upper()
            # Allow alphanumeric characters and hyphens for stocks like BRK-B
            if ticker and all(c.isalnum() or c == '-' for c in ticker):
                cleaned_tickers.append(ticker)
        
        if not cleaned_tickers:
            raise ValueError("No valid tickers provided")
        
        try:
            with open(self.full_config_path, 'w') as f:
                f.write("# Stock tickers to analyze - one per line\n")
                f.write("# Edit this file to customize your stock selection\n")
                for ticker in cleaned_tickers:
                    f.write(f"{ticker}\n")
            
            logger.info(f"Saved {len(cleaned_tickers)} tickers to {self.full_config_path}")
            
        except Exception as e:
            logger.error(f"Error saving ticker config: {e}")
            raise
    
    def add_tickers(self, new_tickers: List[str]) -> List[str]:
        """
        Add new tickers to existing configuration.
        
        Args:
            new_tickers: List of ticker symbols to add
            
        Returns:
            Updated list of all tickers
        """
        current_tickers = set(self.load_tickers())
        
        for ticker in new_tickers:
            ticker = ticker.strip().upper()
            if ticker and ticker.isalnum():
                current_tickers.add(ticker)
        
        updated_tickers = sorted(list(current_tickers))
        self.save_tickers(updated_tickers)
        
        return updated_tickers
    
    def remove_tickers(self, tickers_to_remove: List[str]) -> List[str]:
        """
        Remove tickers from configuration.
        
        Args:
            tickers_to_remove: List of ticker symbols to remove
            
        Returns:
            Updated list of remaining tickers
        """
        current_tickers = set(self.load_tickers())
        
        for ticker in tickers_to_remove:
            ticker = ticker.strip().upper()
            current_tickers.discard(ticker)
        
        if not current_tickers:
            raise ValueError("Cannot remove all tickers. At least one ticker must remain.")
        
        updated_tickers = sorted(list(current_tickers))
        self.save_tickers(updated_tickers)
        
        return updated_tickers
    
    def validate_tickers(self, tickers: Optional[List[str]] = None) -> dict:
        """
        Validate that tickers exist and are tradeable.
        
        Args:
            tickers: List of tickers to validate (defaults to current config)
            
        Returns:
            Dict with validation results
        """
        if tickers is None:
            tickers = self.load_tickers()
        
        results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }
        
        logger.info(f"Validating {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                # Try to fetch basic info
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if info and 'symbol' in info:
                    results['valid'].append(ticker)
                    
                    # Check for any warnings
                    if info.get('regularMarketPrice') is None:
                        results['warnings'].append(f"{ticker}: No recent price data")
                else:
                    results['invalid'].append(ticker)
                    
            except Exception as e:
                logger.debug(f"Validation failed for {ticker}: {e}")
                results['invalid'].append(ticker)
        
        logger.info(f"Validation complete: {len(results['valid'])} valid, "
                   f"{len(results['invalid'])} invalid, {len(results['warnings'])} warnings")
        
        return results
    
    def get_config_path(self) -> str:
        """Get the full path to the ticker configuration file."""
        return str(self.full_config_path)