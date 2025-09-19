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
        self.venv_python = self.project_root / ".venv" / "bin" / "python"
        
        # Check if virtual environment exists
        if not self.venv_python.exists():
            logger.error("Virtual environment not found. Please run 'make setup' first.")
            sys.exit(1)
    
    def run_python_script(self, script_path: str, args: Optional[List[str]] = None) -> int:
        """
        Run a Python script with the virtual environment.
        
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
        
        cmd = [str(self.venv_python), str(full_script_path)]
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
            ("Downloading data", "ingestion/download_data.py"),
            ("Preprocessing data", "preprocess/preprocess_data.py"),
            ("Engineering features", "features/feature_engineer.py"),
            ("Creating splits", "preprocess/create_splits.py")
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
            ("Training LightGBM", "models/train_progressive.py"),
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
        
        # Run pytest with the virtual environment
        cmd = [str(self.venv_python), "-m", "pytest", "tests/", "-v"]
        
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
        
        # Check virtual environment
        if not self.venv_python.exists():
            logger.error("Virtual environment not found")
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
                    [str(self.venv_python), "-c", f"import {module}"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info(f"✓ {module}")
                else:
                    logger.error(f"✗ {module} - {result.stderr.strip()}")
                    return 1
            except Exception as e:
                logger.error(f"✗ {module} - {e}")
                return 1
        
        logger.info("Project setup validation passed!")
        return 0

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