#!/usr/bin/env python3
"""
Stock Trends Prediction CLI
Command-line interface for running the stock prediction pipeline.
"""

import sys
import os
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file if available
try:
    from utils.env_utils import load_environment
    load_environment()
except ImportError:
    pass  # Continue without dotenv if not available

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockTrendsCLI:
    """Main CLI class for stock trends prediction pipeline."""
    
    def __init__(self):
        """Initialize CLI."""
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        
        # Check if uv is available
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError()
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.error("uv is not installed. Please install it first:")
            logger.error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
            logger.error("  or: brew install uv")
            sys.exit(1)
    
    def run_python_script(self, script_path: str, args: Optional[List[str]] = None) -> int:
        """
        Run a Python script with uv.
        
        Args:
            script_path: Path to Python script relative to src/
            args: Additional arguments to pass to script
            
        Returns:
            Exit code
        """
        full_script_path = self.src_dir / script_path
        
        if not full_script_path.exists():
            logger.error(f"Script not found: {full_script_path}")
            return 1
        
        cmd = ["uv", "run", "python", str(full_script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            return result.returncode
        except subprocess.CalledProcessError as e:
            logger.error(f"Script failed with exit code {e.returncode}")
            return e.returncode
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return 1
    
    def data_pipeline(self, args):
        """Run data ingestion and preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("RUNNING DATA PIPELINE")
        logger.info("=" * 60)
        
        steps = [
            ("Downloading equity prices", "ingestion/equity_prices.py"),
            ("Downloading macro data", "ingestion/macro_data.py"),
            ("Downloading news data", "ingestion/news_data.py"),
            ("Weekly aggregation", "preprocess/weekly_aggregator.py"),
            ("Engineering features", "features/feature_engineering.py")
        ]
        
        for step_name, script in steps:
            logger.info(f"Step: {step_name}")
            exit_code = self.run_python_script(script)
            if exit_code != 0:
                logger.error(f"Data pipeline failed at step: {step_name}")
                return exit_code
        
        logger.info("Data pipeline completed successfully!")
        return 0
    
    def train_models(self, args):
        """Train baseline models."""
        logger.info("=" * 60)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("=" * 60)
        
        # Create models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        steps = [
            ("Training LightGBM", "models/train_lightgbm.py"),
            ("Training Transformer", "models/train_transformer.py"),
            ("Comparing models", "models/compare_models.py")
        ]
        
        for step_name, script in steps:
            logger.info(f"Step: {step_name}")
            exit_code = self.run_python_script(script)
            if exit_code != 0:
                logger.error(f"Model training failed at step: {step_name}")
                return exit_code
        
        logger.info("Model training completed successfully!")
        return 0
    
    def run_backtest(self, args):
        """Run backtesting analysis."""
        logger.info("=" * 60)
        logger.info("RUNNING BACKTEST ANALYSIS")
        logger.info("=" * 60)
        
        exit_code = self.run_python_script("backtest/simple_backtest.py")
        
        if exit_code == 0:
            logger.info("Backtesting completed successfully!")
        else:
            logger.error("Backtesting failed!")
        
        return exit_code
    
    def generate_explainability(self, args):
        """Generate explainability reports."""
        logger.info("=" * 60)
        logger.info("GENERATING EXPLAINABILITY REPORTS")
        logger.info("=" * 60)
        
        # Create reports directory
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        steps = [
            ("Model explainability", "explainability/model_explainer.py"),
            ("SHAP analysis", "explainability/shap_explainer.py"),
            ("Model validation", "explainability/model_validator.py")
        ]
        
        for step_name, script in steps:
            logger.info(f"Step: {step_name}")
            exit_code = self.run_python_script(script)
            if exit_code != 0:
                logger.error(f"Explainability failed at step: {step_name}")
                return exit_code
        
        logger.info("Explainability analysis completed successfully!")
        return 0
    
    def run_all(self, args):
        """Run complete end-to-end pipeline."""
        logger.info("=" * 60)
        logger.info("RUNNING COMPLETE PIPELINE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run each pipeline step
        pipeline_steps = [
            ("Data pipeline", self.data_pipeline),
            ("Model training", self.train_models),
            ("Backtesting", self.run_backtest),
            ("Explainability", self.generate_explainability)
        ]
        
        for step_name, step_func in pipeline_steps:
            logger.info(f"\\n{'='*20} {step_name.upper()} {'='*20}")
            exit_code = step_func(args)
            if exit_code != 0:
                logger.error(f"Pipeline failed at: {step_name}")
                return exit_code
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info("\\nResults available in:")
        logger.info("- models/ - Trained models and results")
        logger.info("- reports/ - Analysis reports and visualizations")
        logger.info("- data/ - Processed data and splits")
        
        return 0
    
    def run_tests(self, args):
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        # Run pytest with uv
        cmd = ["uv", "run", "pytest", "tests/", "-v"]
        
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=html"])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return 1
    
    def validate_setup(self, args):
        """Validate project setup and dependencies."""
        logger.info("Validating project setup...")
        
        # Check uv is available and project is synced
        try:
            result = subprocess.run(['uv', 'run', 'python', '--version'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode != 0:
                logger.error("uv environment not properly set up")
                return 1
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.error("uv is not available or project not synced")
            return 1
        
        # Check required directories
        required_dirs = ["src", "data", "models", "reports"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                logger.info(f"Creating directory: {dir_path}")
                dir_path.mkdir(exist_ok=True)
        
        # Test Python imports
        test_imports = [
            "pandas", "numpy", "lightgbm", "torch", "sklearn", 
            "yfinance", "matplotlib", "seaborn", "shap"
        ]
        
        for module in test_imports:
            try:
                result = subprocess.run(
                    ["uv", "run", "python", "-c", f"import {module}"],
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    logger.info(f"✓ {module}")
                else:
                    logger.error(f"✗ {module} - {result.stderr.strip()}")
                    return 1
            except Exception as e:
                logger.error(f"✗ {module} - {e}")
                return 1
        
        # Check API keys status
        try:
            from utils.env_utils import print_api_status
            print_api_status()
        except ImportError:
            logger.warning("Could not check API keys status")
        
        logger.info("Project setup validation passed!")
        return 0
    
    def list_tickers(self, args):
        """List currently configured stock tickers."""
        try:
            from utils.ticker_manager import TickerManager
            
            ticker_manager = TickerManager()
            tickers = ticker_manager.load_tickers()
            
            logger.info("=" * 60)
            logger.info("CURRENT STOCK TICKERS")
            logger.info("=" * 60)
            
            if args.validate:
                logger.info("Validating tickers...")
                validation = ticker_manager.validate_tickers(tickers)
                
                if validation['valid']:
                    logger.info(f"✓ Valid tickers ({len(validation['valid'])}):")
                    for ticker in validation['valid']:
                        logger.info(f"  {ticker}")
                
                if validation['invalid']:
                    logger.error(f"✗ Invalid tickers ({len(validation['invalid'])}):")
                    for ticker in validation['invalid']:
                        logger.error(f"  {ticker}")
                
                if validation['warnings']:
                    logger.warning(f"⚠ Warnings ({len(validation['warnings'])}):")
                    for warning in validation['warnings']:
                        logger.warning(f"  {warning}")
            else:
                logger.info(f"Currently tracking {len(tickers)} stocks:")
                for i, ticker in enumerate(tickers, 1):
                    logger.info(f"  {i:2d}. {ticker}")
            
            logger.info("")
            logger.info(f"Configuration file: {ticker_manager.get_config_path()}")
            logger.info("Use 'uv run cli.py tickers update' to modify the list")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error listing tickers: {e}")
            return 1
    
    def show_default_tickers(self, args):
        """Show the default ticker list available in config.py."""
        try:
            logger.info("=" * 60)
            logger.info("DEFAULT STOCK TICKERS (from config.py)")
            logger.info("=" * 60)
            
            try:
                from ingestion.config import DEFAULT_TICKERS
                logger.info(f"Available default tickers ({len(DEFAULT_TICKERS)}):")
                
                # Display in rows of 5 for better readability
                for i in range(0, len(DEFAULT_TICKERS), 5):
                    row = DEFAULT_TICKERS[i:i+5]
                    logger.info("  " + "  ".join(f"{ticker:<6}" for ticker in row))
                
                logger.info("")
                logger.info("These tickers represent a subset of S&P 500 companies.")
                logger.info("Use 'uv run cli.py tickers update --set [TICKERS]' to replace your current list.")
                logger.info("Use 'uv run cli.py tickers update --add [TICKERS]' to add specific ones.")
                
            except ImportError:
                logger.error("Could not import DEFAULT_TICKERS from config.py")
                return 1
                
            return 0
            
        except Exception as e:
            logger.error(f"Error showing default tickers: {e}")
            return 1
    
    def update_tickers(self, args):
        """Update the stock ticker configuration."""
        try:
            from utils.ticker_manager import TickerManager
            
            ticker_manager = TickerManager()
            
            if args.set:
                # Replace entire list
                new_tickers = [t.strip().upper() for t in args.set]
                logger.info(f"Setting tickers to: {new_tickers}")
                
                # Validate first if requested
                if args.validate:
                    validation = ticker_manager.validate_tickers(new_tickers)
                    if validation['invalid']:
                        logger.error(f"Invalid tickers found: {validation['invalid']}")
                        if not args.force:
                            logger.error("Use --force to proceed anyway")
                            return 1
                
                ticker_manager.save_tickers(new_tickers)
                logger.info(f"✓ Updated ticker list with {len(new_tickers)} stocks")
            
            elif args.reset_to_defaults:
                # Reset to defaults from config.py
                try:
                    from ingestion.config import DEFAULT_TICKERS
                    logger.info(f"Resetting to {len(DEFAULT_TICKERS)} default tickers from config.py")
                    
                    # Validate first if requested
                    if args.validate:
                        validation = ticker_manager.validate_tickers(DEFAULT_TICKERS)
                        if validation['invalid']:
                            logger.error(f"Invalid default tickers found: {validation['invalid']}")
                            if not args.force:
                                logger.error("Use --force to proceed anyway")
                                return 1
                    
                    ticker_manager.save_tickers(DEFAULT_TICKERS)
                    logger.info(f"✓ Reset to default ticker list with {len(DEFAULT_TICKERS)} stocks")
                    
                except ImportError:
                    logger.error("Could not import DEFAULT_TICKERS from config.py")
                    return 1
                
            elif args.add:
                # Add to existing list
                tickers_to_add = [t.strip().upper() for t in args.add]
                logger.info(f"Adding tickers: {tickers_to_add}")
                
                # Validate first if requested
                if args.validate:
                    validation = ticker_manager.validate_tickers(tickers_to_add)
                    if validation['invalid']:
                        logger.error(f"Invalid tickers found: {validation['invalid']}")
                        if not args.force:
                            logger.error("Use --force to proceed anyway")
                            return 1
                
                updated_tickers = ticker_manager.add_tickers(tickers_to_add)
                logger.info(f"✓ Added {len(tickers_to_add)} tickers. Total: {len(updated_tickers)}")
                
            elif args.remove:
                # Remove from existing list
                tickers_to_remove = [t.strip().upper() for t in args.remove]
                logger.info(f"Removing tickers: {tickers_to_remove}")
                
                updated_tickers = ticker_manager.remove_tickers(tickers_to_remove)
                logger.info(f"✓ Removed {len(tickers_to_remove)} tickers. Remaining: {len(updated_tickers)}")
                
            else:
                logger.error("No action specified. Use --set, --add, --remove, or --reset-to-defaults")
                return 1
            
            # Show final list
            final_tickers = ticker_manager.load_tickers()
            logger.info("")
            logger.info("Updated ticker list:")
            for i, ticker in enumerate(final_tickers, 1):
                logger.info(f"  {i:2d}. {ticker}")
            
            logger.info("")
            logger.info("💡 Remember to run 'make data' to download new data if you added stocks")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error updating tickers: {e}")
            return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Trends Prediction Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data          # Run data pipeline only
  %(prog)s models        # Train models only  
  %(prog)s all           # Run complete pipeline
  %(prog)s test          # Run unit tests
  %(prog)s validate      # Validate setup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data pipeline
    data_parser = subparsers.add_parser('data', help='Run data ingestion and preprocessing')
    data_parser.set_defaults(func=StockTrendsCLI().data_pipeline)
    
    # Model training
    models_parser = subparsers.add_parser('models', help='Train baseline models')
    models_parser.set_defaults(func=StockTrendsCLI().train_models)
    
    # Backtesting
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting analysis')
    backtest_parser.set_defaults(func=StockTrendsCLI().run_backtest)
    
    # Explainability
    explain_parser = subparsers.add_parser('explain', help='Generate explainability reports')
    explain_parser.set_defaults(func=StockTrendsCLI().generate_explainability)
    
    # Complete pipeline
    all_parser = subparsers.add_parser('all', help='Run complete end-to-end pipeline')
    all_parser.set_defaults(func=StockTrendsCLI().run_all)
    
    # Testing
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    test_parser.set_defaults(func=StockTrendsCLI().run_tests)
    
    # Validation
    validate_parser = subparsers.add_parser('validate', help='Validate project setup')
    validate_parser.set_defaults(func=StockTrendsCLI().validate_setup)
    
    # Ticker management
    tickers_parser = subparsers.add_parser('tickers', help='Manage stock ticker configuration')
    tickers_subparsers = tickers_parser.add_subparsers(dest='tickers_command', help='Ticker management commands')
    
    # List tickers
    list_parser = tickers_subparsers.add_parser('list', help='List current stock tickers')
    list_parser.add_argument('--validate', action='store_true', 
                           help='Validate that tickers are tradeable')
    list_parser.set_defaults(func=StockTrendsCLI().list_tickers)
    
    # Show defaults
    defaults_parser = tickers_subparsers.add_parser('defaults', help='Show default ticker list from config.py')
    defaults_parser.set_defaults(func=StockTrendsCLI().show_default_tickers)
    
    # Update tickers
    update_parser = tickers_subparsers.add_parser('update', help='Update stock ticker list')
    update_group = update_parser.add_mutually_exclusive_group(required=True)
    update_group.add_argument('--set', nargs='+', metavar='TICKER',
                            help='Replace entire ticker list (e.g., --set AAPL MSFT GOOGL)')
    update_group.add_argument('--add', nargs='+', metavar='TICKER',
                            help='Add tickers to current list (e.g., --add NVDA AMD)')
    update_group.add_argument('--remove', nargs='+', metavar='TICKER',
                            help='Remove tickers from current list (e.g., --remove TSLA)')
    update_group.add_argument('--reset-to-defaults', action='store_true',
                            help='Reset to default tickers from config.py')
    update_parser.add_argument('--validate', action='store_true',
                             help='Validate tickers before updating')
    update_parser.add_argument('--force', action='store_true',
                             help='Force update even if validation fails')
    update_parser.set_defaults(func=StockTrendsCLI().update_tickers)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI and run command
    cli = StockTrendsCLI()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())