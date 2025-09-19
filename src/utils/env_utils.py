"""
Environment variable utilities for the stock trends prediction project.
"""

import os
from pathlib import Path
from typing import Optional

def load_environment():
    """
    Load environment variables from .env file if available.
    This should be called at the start of the application.
    """
    try:
        from dotenv import load_dotenv
        
        # Look for .env file in project root
        env_path = Path(__file__).parent.parent / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
            return True
        else:
            # Try current directory as fallback
            load_dotenv()
            return True
            
    except ImportError:
        # python-dotenv not installed, continue without it
        return False

def get_api_key(key_name: str, required: bool = False) -> Optional[str]:
    """
    Get API key from environment variables with helpful error messages.
    
    Args:
        key_name: Name of the environment variable
        required: Whether this key is required for operation
        
    Returns:
        API key value or None if not found
        
    Raises:
        ValueError: If required=True and key not found
    """
    value = os.getenv(key_name)
    
    if not value and required:
        raise ValueError(
            f"Required environment variable {key_name} not found. "
            f"Please set it in your environment or .env file. "
            f"See .env.example for reference."
        )
    
    return value

def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    return get_api_key('FRED_API_KEY')

def get_finnhub_api_key() -> Optional[str]:
    """Get Finnhub API key from environment.""" 
    return get_api_key('FINNHUB_API_KEY')

def check_api_keys_status() -> dict:
    """
    Check status of all API keys.
    
    Returns:
        Dictionary with API key status information
    """
    status = {
        'fred': {
            'available': get_fred_api_key() is not None,
            'key': 'FRED_API_KEY',
            'url': 'https://fred.stlouisfed.org/docs/api/api_key.html',
            'description': 'Federal Reserve Economic Data (macroeconomic indicators)'
        },
        'finnhub': {
            'available': get_finnhub_api_key() is not None,
            'key': 'FINNHUB_API_KEY', 
            'url': 'https://finnhub.io/register',
            'description': 'Finnhub Stock API (financial news data)'
        }
    }
    
    return status

def print_api_status():
    """Print status of API keys for user information."""
    print("=" * 60)
    print("API KEYS STATUS")
    print("=" * 60)
    
    status = check_api_keys_status()
    
    for service, info in status.items():
        status_str = "✅ Available" if info['available'] else "❌ Missing"
        print(f"{service.upper():10} {status_str}")
        if not info['available']:
            print(f"          Set {info['key']} environment variable")
            print(f"          Get API key: {info['url']}")
        print()
    
    missing_keys = [info['key'] for info in status.values() if not info['available']]
    
    if missing_keys:
        print("📝 To add missing API keys:")
        print("   1. Copy .env.example to .env")
        print("   2. Edit .env with your API keys")
        print("   3. Restart the application")
        print()
        print("⚠️  Project will work with limited functionality without API keys")
    else:
        print("🎉 All API keys configured!")
    
    print("=" * 60)

if __name__ == "__main__":
    # Load environment and print status
    load_environment()
    print_api_status()