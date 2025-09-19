# AI Agents Development Guide

This document provides comprehensive guidance for AI agents working on machine learning and data science projects, based on best practices implemented in the Stock Trends Prediction project.

## Project Architecture & Structure

### Modern Python Project Setup
```
project-root/
├── src/                    # Source code (never put code in root)
│   ├── ingestion/         # Data collection modules
│   ├── preprocess/        # Data preprocessing  
│   ├── features/          # Feature engineering
│   ├── models/            # ML model implementations
│   ├── backtest/          # Trading/evaluation framework
│   ├── explainability/    # Model interpretation
│   └── utils/             # Shared utilities
├── data/                  # Data directories (gitignored)
│   ├── raw/              # Original downloaded data
│   ├── processed/        # Cleaned/transformed data
│   └── splits/           # Train/validation/test splits
├── models/               # Trained models and results (gitignored)
├── reports/              # Analysis outputs (gitignored)
├── tests/                # Unit tests
├── pyproject.toml        # Modern Python packaging
├── requirements.txt      # Dependency listing
├── Makefile             # Build automation
├── cli.py               # Command-line interface
├── .env.example         # Environment variable template
└── .gitignore           # Comprehensive git ignore rules
```

**Key Principles:**
- Keep source code in `src/` directory
- Separate data/models/reports from code
- Use meaningful module names by functionality
- Include both Makefile and CLI for different workflows
- Modern packaging with `pyproject.toml`

## Dependency Management

### pyproject.toml Configuration
```toml
[project]
dependencies = [
    # Core ML/Data Science
    "pandas>=2.0.0",
    "numpy>=1.24.0", 
    "scikit-learn>=1.3.0",
    
    # ML Models
    "lightgbm>=4.0.0",
    "torch>=2.0.0",
    
    # Environment
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"]
gpu = ["torch[cuda]>=2.0.0"]
all = ["project-name[dev,gpu]"]
```

**Best Practices:**
- Use version constraints (>=X.Y.Z) for stability
- Group dependencies by purpose with comments
- Optional dependency groups for different use cases
- Include development tools configuration

## CLI & Automation Design

### Dual Interface Pattern
Provide both **Makefile** (simple) and **CLI** (advanced) interfaces:

```python
# cli.py structure
class ProjectCLI:
    def __init__(self):
        # Load environment variables
        # Setup paths and validation
        
    def data_pipeline(self, args):
        # Orchestrate multiple data processing steps
        
    def train_models(self, args):
        # Run model training with progress tracking
```

```makefile
# Makefile targets
setup:              # Environment setup
data:               # Data pipeline
models:             # Model training
test:               # Run tests
all:                # Complete pipeline
```

**Key Features:**
- Environment variable loading at startup
- Progress tracking and time estimates
- Error handling with helpful messages
- Validation commands for setup verification
- Both simple (make) and detailed (CLI) workflows

## Environment Variable Management

### Professional .env Workflow
```bash
# .env.example (committed to git)
FRED_API_KEY=your_fred_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

```python
# env_utils.py
def load_environment():
    """Load .env file with graceful fallback."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        return True
    except ImportError:
        return False

def get_api_key(key_name: str, required: bool = False):
    """Get API key with helpful error messages."""
    value = os.getenv(key_name)
    if not value and required:
        raise ValueError(f"Required {key_name} not found. See .env.example")
    return value
```

**Best Practices:**
- Provide `.env.example` template
- Never commit actual `.env` files
- Graceful degradation when keys missing
- Clear error messages with setup instructions
- Status checking built into validation

## Data Pipeline Architecture

### Modular Data Processing
```python
# ingestion/equity_prices.py
class EquityPriceIngestor:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def download_prices(self, tickers: List[str], start_date: str):
        # Download with error handling and progress tracking
        # Save to standardized format (parquet)
        # Log data quality metrics
```

**Design Patterns:**
- **Class-based components** for state management
- **Standardized data formats** (parquet for efficiency)
- **Comprehensive logging** with metrics
- **Error handling** with graceful failure
- **Progress tracking** for long operations

### Data Quality & Validation
```python
def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality checks."""
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': (df['date'].min(), df['date'].max()),
        'unique_tickers': df['ticker'].nunique(),
        'data_quality_score': calculate_quality_score(df)
    }
```

## Feature Engineering Best Practices

### Systematic Feature Generation
```python
class FeatureEngineer:
    def __init__(self, target_horizon: int = 12):
        self.target_horizon = target_horizon
        self.feature_groups = {
            'price': self._create_price_features,
            'technical': self._create_technical_features,
            'volume': self._create_volume_features,
            'volatility': self._create_volatility_features
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all feature groups with logging."""
        for group_name, func in self.feature_groups.items():
            df = func(df)
            logger.info(f"Added {group_name} features, shape: {df.shape}")
        return df
```

**Feature Engineering Principles:**
- **Organized by category** (price, technical, volume, etc.)
- **Parameterized horizons** for different timeframes
- **Domain knowledge integration** (financial indicators)
- **Comprehensive coverage** (80+ features implemented)
- **Reproducible computation** with consistent methodology

## Model Training Framework

### Multi-Model Architecture
```python
# Abstract base for all models
class BaseModel(ABC):
    @abstractmethod
    def train_single_split(self, split_id: int) -> Dict[str, Any]:
        pass
    
    def train_all_splits(self, max_splits: int = 5):
        """Walk-forward cross-validation."""
        results = []
        for split_id in range(max_splits):
            result = self.train_single_split(split_id)
            results.append(result)
        return self._aggregate_results(results)

# Specific implementations
class LightGBMModel(BaseModel):
    def train_single_split(self, split_id: int):
        # Time-aware train/val/test splitting
        # Hyperparameter optimization
        # Early stopping with validation
        # Feature importance extraction
        
class TemporalTransformerModel(BaseModel):
    def train_single_split(self, split_id: int):
        # Sequence preparation
        # Attention mechanism training
        # Learning rate scheduling
        # Model checkpointing
```

**Training Best Practices:**
- **Time-aware splitting** (no data leakage)
- **Walk-forward validation** (realistic backtesting)
- **Early stopping** (prevent overfitting)
- **Progress tracking** with time estimates
- **Comprehensive metrics** (RMSE, directional accuracy)
- **Model persistence** with timestamps

### Model Comparison Framework
```python
def compare_models(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare multiple models with ranking."""
    metrics = ['test_rmse', 'test_dir_accuracy', 'val_rmse']
    
    for metric in metrics:
        # Sort models by performance
        if metric.endswith('_rmse'):
            sorted_models = sorted(model_results.items(), 
                                 key=lambda x: x[1]['mean'])  # Lower better
        else:
            sorted_models = sorted(model_results.items(), 
                                 key=lambda x: x[1]['mean'], reverse=True)  # Higher better
        
        # Display rankings with statistical significance
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            logger.info(f"{rank}. {model_name:<15} {stats['mean']:>8.4f} ± {stats['std']:>6.4f}")
```

## Explainability & Interpretation

### SHAP Integration
```python
class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)  # For tree models
        
    def generate_explanations(self, X_test, n_samples: int = 500):
        """Generate SHAP explanations with visualizations."""
        shap_values = self.explainer.shap_values(X_test[:n_samples])
        
        # Create comprehensive visualizations
        self._create_summary_plot(shap_values, X_test)
        self._create_dependence_plots(shap_values, X_test)
        self._create_feature_importance_plot(shap_values)
        
        return self._analyze_feature_attribution(shap_values, X_test)
```

**Explainability Components:**
- **SHAP analysis** for individual predictions
- **Feature importance** rankings  
- **Dependence plots** for feature interactions
- **Model validation** with consistency checks
- **Investment insights** extraction from patterns

## Testing & Validation Strategy

### Comprehensive Test Suite
```python
# tests/test_pipeline.py
class TestFeatureEngineer:
    def test_feature_generation_reproducibility(self):
        """Ensure feature engineering is deterministic."""
        engineer1 = FeatureEngineer(random_seed=42)
        engineer2 = FeatureEngineer(random_seed=42)
        
        features1 = engineer1.engineer_features(self.sample_data)
        features2 = engineer2.engineer_features(self.sample_data)
        
        pd.testing.assert_frame_equal(features1, features2)

class TestDataValidation:
    def test_data_directory_structure(self):
        """Validate project structure exists."""
        required_dirs = ["data/raw", "data/processed", "models", "reports"]
        for dir_path in required_dirs:
            assert Path(dir_path).exists()
```

**Testing Principles:**
- **Reproducibility tests** (random seed consistency)
- **Data validation** (structure and quality)
- **Model consistency** (training determinism)
- **Pipeline integration** (end-to-end workflows)
- **Configuration validation** (environment setup)

## Git & Version Control

### Comprehensive .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Data & Models (generated files)
data/raw/
data/processed/
models/*.pkl
models/*_results_*.json  # Timestamped files
reports/*.png

# Environment & Secrets
.env
.env.local
api_keys.json

# ML/Data Science specific
*.h5
checkpoints/
wandb/
mlruns/

# Keep templates and documentation
!.env.example
!models/.gitkeep
!data/raw/.gitkeep
```

**Version Control Best Practices:**
- **Ignore generated files** (models, results, logs)
- **Template files included** (.env.example)
- **Directory structure preserved** (.gitkeep files)
- **Timestamped patterns** (*_results_*.json)
- **API keys protected** (never commit secrets)

## Documentation Standards

### README Structure
```markdown
# Project Title
Brief description with key results upfront

## Project Overview
- Core technologies and approaches
- Key performance metrics
- Main insights and findings

## Quick Start
### Prerequisites
### Setup and Run
### Environment Variables (Optional)

## Project Structure
Detailed directory layout with explanations

## Available Commands
Both Make and CLI commands documented

## Pipeline Components
Technical details of each component

## Output Reports
What gets generated and where to find it
```

**Documentation Principles:**
- **Results upfront** (key metrics in overview)
- **Quick start guide** for immediate usage
- **Environment setup** clearly explained
- **Multiple interfaces** (CLI + Make) documented
- **Output organization** (what gets generated where)

## Development Workflow

### Code Organization Patterns
```python
# 1. Imports organized by type
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from utils.env_utils import load_environment
from models.base import BaseModel

# 2. Class-based components with clear interfaces
class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        
    def process(self) -> Dict[str, Any]:
        """Main processing method with error handling."""
        try:
            return self._process_implementation()
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

# 3. Progress tracking for long operations
def train_models(self, max_splits: int = 5):
    tracker = ProgressTracker(max_splits)
    
    for split_id in range(max_splits):
        start_time = time.time()
        result = self.train_single_split(split_id)
        
        tracker.update(time.time() - start_time)
        logger.info(f"Progress: {tracker.get_status()}")
```

### Error Handling & Logging
```python
# Structured logging with metrics
logger.info(f"Training completed - RMSE: {rmse:.4f}, Accuracy: {acc:.3f}")

# Graceful degradation
try:
    macro_data = self.download_macro_data()
except APIError as e:
    logger.warning(f"Macro data unavailable: {e}")
    macro_data = None  # Continue without it

# User-friendly error messages
if not api_key:
    raise ValueError(
        f"Missing {key_name}. Please set in .env file. "
        f"Get free key at: {api_url}"
    )
```

## AI Agent Guidelines

### When Working on ML Projects:

1. **Architecture First**
   - Always start with project structure
   - Separate data/models/code clearly
   - Use modern Python packaging (pyproject.toml)

2. **Environment Management**
   - Create .env.example for API keys
   - Implement graceful degradation
   - Add validation commands

3. **Data Pipeline Design**
   - Modular, class-based components
   - Comprehensive logging and metrics
   - Progress tracking for long operations
   - Standardized data formats (parquet)

4. **Model Training Framework**
   - Time-aware cross-validation
   - Multiple model support with common interface
   - Comprehensive evaluation metrics
   - Model comparison with rankings

5. **Code Quality**
   - Type hints for complex functions
   - Error handling with helpful messages
   - Reproducibility through random seeds
   - Testing for critical components

6. **User Experience**
   - Both CLI and Make interfaces
   - Clear documentation with examples
   - Status checking and validation
   - Professional git workflow

7. **Documentation & Maintenance**
   - Results-first README structure
   - Comprehensive .gitignore
   - Environment variable documentation
   - Code organization principles

## Documentation Standards and Tone

### Professional Documentation Guidelines

**Tone and Style:**
- Use clear, professional language throughout all documentation
- Avoid emojis in project documentation, READMEs, code comments, and technical guides
- Focus on clarity and precision over visual appeal
- Write for technical professionals and maintainers

**Documentation Structure:**
- Lead with concrete results and metrics
- Provide clear, step-by-step instructions
- Include comprehensive examples
- Document all configuration options and environment variables
- Maintain consistent formatting and structure

**Code Comments:**
- Use descriptive docstrings for all functions and classes
- Explain complex algorithms and business logic
- Document parameter types and return values
- Include usage examples for key functions

**README Files:**
- Start with project overview and key results
- Provide quick start instructions
- Document all available commands and interfaces
- Include troubleshooting sections
- Maintain up-to-date dependency lists

This project demonstrates production-ready ML pipeline development with modern Python practices, comprehensive automation, and professional user experience design.