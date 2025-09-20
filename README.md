# Stock Trends Prediction Project<a name="stock-trends-prediction-project"></a>

A comprehensive machine learning pipeline for predicting stock price movements using technical analysis, time series modeling, and explainable AI techniques.

## Table of Contents<a name="table-of-contents"></a>

<!-- mdformat-toc start --slug=github --maxlevel=3 --minlevel=1 -->

- [Stock Trends Prediction Project](#stock-trends-prediction-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Key Results](#key-results)
    - [Model Performance](#model-performance)
    - [Key Insights](#key-insights)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Setup and Run](#setup-and-run)
    - [Environment Variables (Optional)](#environment-variables-optional)
    - [Alternative CLI Usage](#alternative-cli-usage)
  - [How to Use This Tool](#how-to-use-this-tool)
    - [What This Tool Does](#what-this-tool-does)
    - [Input: What You Provide](#input-what-you-provide)
    - [Expected Outcomes: What You Get](#expected-outcomes-what-you-get)
    - [Step-by-Step Usage Guide](#step-by-step-usage-guide)
    - [Real-World Application Examples](#real-world-application-examples)
    - [Understanding the Output](#understanding-the-output)
    - [Performance & Requirements](#performance--requirements)
  - [Project Structure](#project-structure)
  - [Available Commands](#available-commands)
    - [Make Commands](#make-commands)
    - [CLI Commands](#cli-commands)
  - [Pipeline Components](#pipeline-components)
    - [1. Data Ingestion](#1-data-ingestion)
    - [2. Feature Engineering](#2-feature-engineering)
    - [3. Model Training](#3-model-training)
    - [4. Backtesting Framework](#4-backtesting-framework)
    - [5. Explainability Analysis](#5-explainability-analysis)
  - [Output Reports](#output-reports)
    - [Visualizations](#visualizations)
    - [Data Reports](#data-reports)
  - [Testing](#testing)
  - [Model Interpretability](#model-interpretability)
    - [Top Features (by SHAP importance)](#top-features-by-shap-importance)
    - [Investment Implications](#investment-implications)
  - [Configuration](#configuration)
    - [Ticker Configuration](#ticker-configuration)
    - [Model Parameters](#model-parameters)
    - [Trading Parameters](#trading-parameters)
  - [Dependencies](#dependencies)
  - [Development](#development)
    - [Code Quality](#code-quality)
    - [Adding New Features](#adding-new-features)
    - [Adding New Models](#adding-new-models)
  - [Performance Benchmarks](#performance-benchmarks)
    - [Expected Results](#expected-results)
    - [Reproducibility](#reproducibility)
  - [Contributing](#contributing)
  - [License](#license)
  - [References](#references)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues & Solutions](#common-issues--solutions)
    - [Getting Help](#getting-help)
  - [Contributing](#contributing-1)

<!-- mdformat-toc end -->

## Project Overview<a name="project-overview"></a>

This project implements a complete end-to-end system for predicting 12-week forward stock returns using:

- **Technical Analysis Features**: 80+ engineered features including moving averages, momentum indicators, volatility measures, and price patterns
- **Baseline Models**: LightGBM gradient boosting and Temporal Transformer neural networks
- **Walk-Forward Validation**: Realistic backtesting with time-aware cross-validation
- **Explainable AI**: SHAP analysis and feature importance for model interpretability
- **Realistic Trading Simulation**: Transaction costs, slippage, and position sizing

## Key Results<a name="key-results"></a>

### Model Performance<a name="model-performance"></a>

- **Transformer Model**: 0.102 ± 0.005 RMSE, 67.3% ± 5.6% directional accuracy
- **LightGBM Model**: 0.109 ± 0.016 RMSE, 69.9% ± 7.3% directional accuracy
- **Transformer wins overall** with 6.4% better test RMSE

### Key Insights<a name="key-insights"></a>

- **Most important feature**: On-Balance Volume (OBV) by both traditional and SHAP analysis
- **Feature categories**: Momentum indicators are highly predictive, volatility measures crucial for risk assessment
- **Investment implications**: Trend-following strategies may be effective, volume confirmation valuable

## Quick Start<a name="quick-start"></a>

### Prerequisites<a name="prerequisites"></a>

- Python 3.10+
- **uv** (fast Python package manager) - [Install uv](https://docs.astral.sh/uv/)
- macOS, Linux, or Windows
- Internet connection for data download

### Setup and Run<a name="setup-and-run"></a>

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# Clone and setup
git clone <repository-url>
cd stock-trends

# Setup environment and run complete pipeline (using uv)
make setup
make all
```

### Environment Variables (Optional)<a name="environment-variables-optional"></a>

The project works without API keys but with limited functionality. For full features, configure these APIs:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (get them for free):
# - FRED_API_KEY: https://fred.stlouisfed.org/docs/api/api_key.html
# - FINNHUB_API_KEY: https://finnhub.io/register
```

**Required Environment Variables:**

- `FRED_API_KEY` - Federal Reserve Economic Data API (macroeconomic indicators)
- `FINNHUB_API_KEY` - Finnhub Stock API (financial news data)

**What happens without API keys:**

- **Still works**: Stock price data (yfinance), technical analysis, model training, backtesting
- **Limited**: No macroeconomic data (GDP, inflation) or financial news sentiment

**Rate Limits (Free Tiers):**

- FRED API: 120 requests/minute
- Finnhub API: 60 requests/minute

### Alternative CLI Usage<a name="alternative-cli-usage"></a>

```bash
# Direct CLI commands (recommended - ensures correct environment)
uv run cli.py validate    # Validate setup
uv run cli.py all          # Run complete pipeline
uv run cli.py models       # Train models only
uv run cli.py explain      # Generate explainability reports

# Make commands - two approaches:
make all                  # Traditional: individual scripts with Make orchestration  
make cli-all              # Modern: unified CLI with integrated error handling

# Other Make commands
make validate             # Same as: uv run cli.py validate
make cli-validate         # Same as: uv run cli.py validate (unified interface)
```

## How to Use This Tool<a name="how-to-use-this-tool"></a>

### What This Tool Does<a name="what-this-tool-does"></a>

This tool predicts **12-week forward stock returns** using machine learning models trained on technical analysis indicators. It helps you:

1. **Identify trending stocks** - Find stocks likely to move up or down over the next 3 months
1. **Understand why predictions work** - SHAP analysis explains which technical indicators drive predictions
1. **Validate trading strategies** - Realistic backtesting with transaction costs and slippage
1. **Compare ML approaches** - LightGBM vs Transformer models with performance metrics

### Input: What You Provide<a name="input-what-you-provide"></a>

The tool works with **any publicly traded stocks**. You can analyze:

```python
# Default: Large-cap tech stocks (in src/ingestion/equity_prices.py)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

# Customize: Edit the ticker list to analyze your stocks
CUSTOM_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMD", "INTC"]  # ETFs and individual stocks
```

**To analyze your own stocks:**

1. **CLI commands**: `uv run cli.py tickers update --set YOUR STOCKS HERE`
1. **Make commands**: `make tickers-list` (view current), `make tickers-reset` (use defaults)
1. **Manual editing**: Edit `config/tickers.txt` directly
1. **Run pipeline**: `make all` or `make cli-all`

### Expected Outcomes: What You Get<a name="expected-outcomes-what-you-get"></a>

#### 1. **Prediction Models** (`models/` directory)

- **LightGBM Model**: Fast, interpretable gradient boosting
- **Transformer Model**: Deep learning for complex patterns
- **Performance Metrics**: RMSE, directional accuracy, statistical significance

#### 2. **Stock Predictions** (`reports/` directory)

- **Individual stock forecasts**: 12-week return predictions for each ticker
- **Confidence intervals**: Prediction uncertainty ranges
- **Directional signals**: Buy/Hold/Sell recommendations based on predicted returns

#### 3. **Feature Importance Analysis**

- **Top predictive indicators**: Which technical analysis metrics matter most
- **SHAP explanations**: Why specific predictions were made
- **Feature categories**: Price momentum, volume patterns, volatility measures

#### 4. **Backtesting Results**

- **Historical performance**: How well predictions would have worked in the past
- **Risk-adjusted returns**: Sharpe ratios, maximum drawdown
- **Transaction cost impact**: Realistic trading simulation with fees

### Step-by-Step Usage Guide<a name="step-by-step-usage-guide"></a>

#### Step 1: Complete Pipeline (Recommended First Run)

```bash
# Option A: Traditional Make approach (fine-grained control)
make all
# Runs individual scripts: setup → data → features → models → backtest → explainability
# Good for: debugging specific steps, development workflow

# Option B: Unified CLI approach (integrated error handling)  
make cli-all
# Runs unified pipeline: uv run cli.py all
# Good for: production runs, consistent logging, progress tracking

# Both take: 15-30 minutes depending on number of stocks and internet speed
# Both produce identical results - choose based on your workflow preference
```

#### Step 2: Analyze Results

```bash
# Check what was generated
ls reports/           # Analysis reports and visualizations
ls models/           # Trained models and performance metrics

# Key files to examine:
# - reports/model_comparison.json    # Performance comparison
# - reports/feature_importance.png   # Which indicators matter most
# - reports/shap_analysis.png        # Prediction explanations
# - reports/backtest_results.json    # Trading simulation results
```

#### Step 3: Interpret Predictions

```python
# Example: Reading model predictions
import json

# Load performance comparison
with open('models/model_comparison.json') as f:
    comparison = json.load(f)
    
print(f"Best model: {comparison['best_model']['name']}")
print(f"Test RMSE: {comparison['best_model']['test_rmse']:.4f}")
print(f"Directional accuracy: {comparison['best_model']['directional_accuracy']:.1%}")

# Load individual stock predictions (if available)
with open('reports/stock_predictions.json') as f:
    predictions = json.load(f)
    
for stock, pred in predictions.items():
    direction = "UP" if pred['predicted_return'] > 0.02 else "DOWN" if pred['predicted_return'] < -0.02 else "HOLD"
    print(f"{stock}: {direction} ({pred['predicted_return']:+.1%} predicted return)")
```

#### Step 4: Customize for Your Use Case

**A. Analyze Different Stocks**

```bash
# Method 1: Use CLI ticker management
uv run cli.py tickers update --set YOUR STOCKS HERE

# Method 2: Use Make ticker commands
make tickers-list         # View current stocks
make tickers-defaults     # See all 40 S&P 500 options
make tickers-reset        # Reset to all defaults
# Or: uv run cli.py tickers update --add NVDA AMD INTC

# Then re-run pipeline with new stocks
make data features models  # Traditional approach
# Or: make cli-all         # Unified approach (full pipeline)
```

**B. Adjust Prediction Horizon**

```bash
# Edit src/features/feature_engineering.py
# Change TARGET_HORIZON = 8  # For 8-week predictions instead of 12
make features models       # Re-train with new horizon
```

**C. Focus on Specific Analysis**

```bash
make models     # Only train models (skip data download)
make explain    # Only generate explainability reports
make backtest   # Only run trading simulation
```

### Real-World Application Examples<a name="real-world-application-examples"></a>

#### Example 1: Weekly Stock Selection

```bash
# 1. Run pipeline on your watchlist
# 2. Check reports/stock_predictions.json for top-ranked stocks
# 3. Use SHAP analysis to understand why certain stocks are predicted to rise
# 4. Consider position sizing based on prediction confidence
```

#### Example 2: Strategy Validation

```bash
# 1. Backtest results show historical performance
# 2. Compare model predictions vs buy-and-hold
# 3. Analyze transaction costs impact on returns
# 4. Use feature importance to refine your manual analysis
```

#### Example 3: Research and Education

```bash
# 1. Understand which technical indicators actually predict returns
# 2. Compare tree-based (LightGBM) vs neural network (Transformer) approaches
# 3. Learn about realistic trading costs and market frictions
# 4. Study model explainability with SHAP analysis
```

### Understanding the Output<a name="understanding-the-output"></a>

**Good Predictions:**

- Directional accuracy > 60% (better than random)
- Low RMSE values (< 0.15 for 12-week returns)
- Consistent performance across multiple validation splits
- Clear feature importance patterns

**Limitations to Remember:**

- Past performance doesn't guarantee future results
- Models work best in trending markets, struggle in highly volatile periods
- Transaction costs significantly impact short-term trading strategies
- Predictions are probabilities, not certainties

**When to Trust Predictions:**

- High model confidence (low prediction variance)
- Strong technical indicator alignment
- Consistent with broader market trends
- Validated on recent out-of-sample data

### Performance & Requirements<a name="performance--requirements"></a>

**Expected Runtime:**

- **Complete pipeline (`make all`)**: 15-30 minutes
  - Data download: 2-5 minutes (depends on internet speed)
  - Feature engineering: 3-8 minutes
  - Model training: 8-15 minutes (LightGBM + Transformer)
  - Backtesting & analysis: 2-5 minutes

**System Requirements:**

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (for data, models, reports)
- **CPU**: Multi-core recommended for faster training
- **GPU**: Optional (Transformer training can benefit from CUDA)

**Data Requirements:**

- **Minimum**: 2 years of historical data per stock
- **Recommended**: 5+ years for robust feature engineering
- **Frequency**: Daily stock prices (automatically downloaded)
- **Sources**: Yahoo Finance (free), FRED & Finnhub (optional with API keys)

## Project Structure<a name="project-structure"></a>

```
stock-trends/
├── src/                    # Source code
│   ├── ingestion/         # Data downloading (yfinance)
│   ├── preprocess/        # Data preprocessing and splits
│   ├── features/          # Feature engineering (80+ features)
│   ├── models/            # LightGBM & Transformer models
│   ├── backtest/          # Trading simulation framework
│   ├── explainability/    # SHAP analysis & model interpretation
│   └── utils/             # Utility functions
├── data/                  # Data directories
│   ├── raw/              # Downloaded price data
│   ├── processed/        # Preprocessed features
│   └── splits/           # Train/validation/test splits
├── models/               # Trained models and results
├── reports/              # Analysis reports and visualizations
├── tests/                # Unit tests
├── Makefile             # Build automation
├── cli.py               # Command-line interface
├── pyproject.toml       # Python dependencies and project config
└── uv.lock             # Locked dependency versions
```

## Available Commands<a name="available-commands"></a>

### Make Commands<a name="make-commands"></a>

```bash
# Traditional pipeline (individual scripts)
make setup              # Setup virtual environment
make data               # Download and preprocess data
make features           # Generate features
make models             # Train baseline models
make backtest           # Run backtesting analysis
make explainability     # Generate explainability reports
make test               # Run unit tests
make all                # Run complete pipeline

# CLI-based alternatives (unified interface)
make cli-data           # Run data pipeline via CLI
make cli-models         # Train models via CLI
make cli-backtest       # Run backtesting via CLI
make cli-explain        # Generate explainability via CLI
make cli-test           # Run tests via CLI
make cli-validate       # Validate setup via CLI
make cli-all            # Run complete pipeline via CLI

# Ticker management
make tickers-list       # List current stock tickers
make tickers-defaults   # Show default ticker options (40 S&P 500 stocks)
make tickers-reset      # Reset to default tickers

# Ticker analysis (explainability)
make ticker-list                    # List all available tickers in reports
make ticker-analyze TICKER=AAPL     # Analyze specific ticker
make ticker-detailed TICKER=TSLA    # Detailed analysis with SHAP features

# Utilities
make check-env          # Check if uv environment is ready
make clean              # Clean temporary files and cache
make format             # Format code with black
make lint               # Run code quality checks
```

### CLI Commands<a name="cli-commands"></a>

```bash
uv run cli.py data      # Data pipeline
uv run cli.py models    # Model training
uv run cli.py backtest  # Backtesting
uv run cli.py explain   # Explainability
uv run cli.py test      # Unit tests
uv run cli.py validate  # Validate setup

# Ticker management
uv run cli.py tickers list                    # View current stocks
uv run cli.py tickers defaults               # View available defaults
uv run cli.py tickers update --add NVDA AMD  # Add stocks
uv run cli.py tickers update --reset-to-defaults  # Reset to 40 S&P stocks
```

## Pipeline Components<a name="pipeline-components"></a>

### 1. Data Ingestion<a name="1-data-ingestion"></a>

- Downloads stock data from Yahoo Finance (yfinance)
- 30 tickers, 5+ years of daily data
- Converts to weekly frequency for modeling

### 2. Feature Engineering<a name="2-feature-engineering"></a>

- **Price Features**: Returns, moving averages, price ratios
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Volume Features**: OBV, volume moving averages, volume ratios
- **Volatility Features**: Realized volatility across multiple timeframes
- **Support/Resistance**: Price levels and position within ranges

### 3. Model Training<a name="3-model-training"></a>

- **Walk-Forward Cross-Validation**: 5 splits with realistic temporal constraints
- **LightGBM**: Gradient boosted trees with hyperparameter optimization
- **Temporal Transformer**: Neural network with attention mechanism for sequences
- **Early Stopping**: Prevents overfitting with validation monitoring

### 4. Backtesting Framework<a name="4-backtesting-framework"></a>

- **Realistic Trading Costs**: Commission (0.1%), bid-ask spread (0.05%), market impact
- **Position Sizing**: Risk-based position sizing with volatility targeting
- **Performance Metrics**: Sharpe ratio, maximum drawdown, hit rate analysis

### 5. Explainability Analysis<a name="5-explainability-analysis"></a>

- **Feature Importance**: Traditional LightGBM importance rankings
- **SHAP Analysis**: Individual prediction explanations and global feature attribution
- **Ticker-Specific Analysis**: Stock-by-stock model behavior and feature importance
- **Model Validation**: Performance benchmarking and consistency checks

#### Ticker-Specific Analysis

The enhanced explainability system now provides detailed analysis for individual stocks:

```bash
# Using Makefile commands (recommended)
make ticker-list                    # List all available tickers
make ticker-analyze TICKER=AAPL     # Analyze Apple stock
make ticker-detailed TICKER=TSLA    # Detailed Tesla analysis

# Or using direct Python commands
uv run ticker_analysis.py --list           # List all available tickers
uv run ticker_analysis.py AAPL             # Analyze Apple stock
uv run ticker_analysis.py TSLA --detailed  # Detailed Tesla analysis

# Example output for AAPL:
# - Sample count: 42 predictions
# - Average prediction: 0.0310 (3.10% expected return)
# - Top features: OBV, volatility_12w, price_position_26w
# - Specific prediction examples with dates and SHAP contributions
```

**Key Insights Available:**

- **Per-Ticker Statistics**: Sample counts, average predictions, volatility
- **Feature Importance by Stock**: How the model weighs features differently for each ticker
- **Prediction Examples**: Specific dates and explanations for high/low predictions
- **Model Behavior**: Understanding why the model makes different predictions for different stocks

## Output Reports<a name="output-reports"></a>

The pipeline generates comprehensive reports in the `reports/` directory:

### Visualizations<a name="visualizations"></a>

- `shap_summary_plot.png` - SHAP feature impact visualization
- `shap_feature_importance.png` - Global feature importance
- `shap_dependence_*.png` - Feature dependence plots for top 5 features
- `lightgbm_feature_importance.png` - Traditional feature importance
- `prediction_analysis.png` - Prediction vs actual analysis

### Data Reports<a name="data-reports"></a>

- `lightgbm_explainability_report_*.json` - Comprehensive LightGBM analysis
- `prediction_explanation_report_*.json` - **SHAP-based prediction explanations with ticker information**
- `validation_report_*.json` - Model validation and performance summary
- `model_comparison_report.json` - Head-to-head model comparison

**Enhanced Reports Now Include:**

- Individual stock ticker names and dates for each prediction
- Ticker-specific feature importance (e.g., what drives AAPL vs TSLA predictions)
- Per-stock statistics and model behavior analysis
- Easy filtering by ticker using the `ticker_analysis.py` tool

## Testing<a name="testing"></a>

The project includes comprehensive unit tests covering:

- Feature engineering reproducibility
- Model initialization and training
- Trading cost calculations
- Position sizing logic
- Data validation and structure
- Reproducibility guarantees

Run tests with:

```bash
make test
# or
uv run cli.py test --coverage  # with coverage report
```

## Model Interpretability<a name="model-interpretability"></a>

### Top Features (by SHAP importance)<a name="top-features-by-shap-importance"></a>

1. **OBV** (On-Balance Volume) - 0.0058
1. **Volatility 12w** - 0.0045
1. **Price Position 26w** - 0.0041
1. **Support 52w** - 0.0026
1. **ATR % 12w** - 0.0025

### Investment Implications<a name="investment-implications"></a>

- **Momentum indicators** are highly predictive → trend-following strategies effective
- **Volatility measures** are important → risk assessment crucial
- **Volume confirmation** valuable → volume-based signals significant
- **Technical indicators** dominate → technical analysis approaches supported
- **Price levels** influential → support/resistance analysis matters

## Configuration<a name="configuration"></a>

### Ticker Configuration<a name="ticker-configuration"></a>

The project uses a file-based ticker configuration system:

- **`config/tickers.example.txt`** - Template file (committed to git)
- **`config/tickers.txt`** - Your personal configuration (gitignored)

When you first run the project, `config/tickers.txt` is automatically created from the example file. You can then customize it without affecting version control.

**Managing your ticker list:**

```bash
# View current tickers
make tickers-list

# Edit with CLI
uv run cli.py tickers update --set AAPL MSFT GOOGL

# Or edit the file directly
vim config/tickers.txt
```

### Model Parameters<a name="model-parameters"></a>

- **LightGBM**: Learning rate 0.05, early stopping after 100 rounds
- **Transformer**: 6 layers, 8 attention heads, 128 hidden dimensions
- **Training**: 5 walk-forward splits, 12-week prediction horizon

### Trading Parameters<a name="trading-parameters"></a>

- **Transaction Costs**: 0.128% total (commission + spread + impact + slippage)
- **Position Sizing**: Max 5% per position, 15% volatility target
- **Rebalancing**: Weekly with 12-week holding period

## Dependencies<a name="dependencies"></a>

**Package Manager**: [uv](https://docs.astral.sh/uv/) (fast, reliable Python package manager)

Core libraries managed by uv:

- `pandas`, `numpy` - Data manipulation
- `lightgbm` - Gradient boosting
- `torch`, `transformers` - Neural networks
- `scikit-learn` - ML utilities
- `yfinance` - Data download
- `matplotlib`, `seaborn` - Visualization
- `shap` - Model explainability
- `pytest` - Testing

All dependencies are defined in `pyproject.toml` and locked in `uv.lock` for reproducible builds.

**Why no requirements.txt?** Modern Python projects use `pyproject.toml` + `uv.lock` instead:

- `pyproject.toml` - Human-readable dependency specification
- `uv.lock` - Exact versions for reproducible builds (auto-generated)
- Much faster and more reliable than the old pip + requirements.txt approach

## Development<a name="development"></a>

### Code Quality<a name="code-quality"></a>

```bash
make lint     # Code quality checks
make format   # Auto-format code
```

### Adding New Features<a name="adding-new-features"></a>

1. Add feature engineering logic to `src/features/feature_engineering.py`
1. Update feature lists in model training scripts
1. Add unit tests in `tests/`
1. Run validation: `uv run cli.py test`

### Adding New Models<a name="adding-new-models"></a>

1. Create model class in `src/models/`
1. Implement training loop following existing patterns
1. Add to comparison script `src/models/compare_models.py`
1. Update explainability analysis if needed

## Performance Benchmarks<a name="performance-benchmarks"></a>

### Expected Results<a name="expected-results"></a>

- **RMSE**: < 0.15 (both models achieve ~0.10)
- **Directional Accuracy**: > 55% (Transformer achieves 67%)
- **Training Time**: ~2-5 minutes per model on modern CPU
- **Memory Usage**: < 2GB RAM for full pipeline

### Reproducibility<a name="reproducibility"></a>

- Fixed random seeds ensure consistent results
- Unit tests validate reproducibility across runs
- Virtual environment pins exact dependency versions

## Contributing<a name="contributing"></a>

1. Follow existing code structure and patterns
1. Add unit tests for new functionality
1. Run full test suite: `make test`
1. Update documentation as needed
1. Use black code formatting: `make format`

## License<a name="license"></a>

This project is for educational and research purposes. See individual data source terms for usage restrictions.

## References<a name="references"></a>

- **Technical Analysis**: Murphy, J. "Technical Analysis of the Financial Markets"
- **SHAP**: Lundberg, S. "A Unified Approach to Interpreting Model Predictions"
- **Walk-Forward Analysis**: Pardo, R. "The Evaluation and Optimization of Trading Strategies"
- **Data Source**: Yahoo Finance via yfinance library

______________________________________________________________________

**Built with dedication for the stock prediction community**

## Troubleshooting<a name="troubleshooting"></a>

### Common Issues & Solutions<a name="common-issues--solutions"></a>

**Problem: "uv is not installed" error**

```bash
# Solution: Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
# Then restart your terminal
```

**Problem: Data download fails or is very slow**

```bash
# Solution 1: Check internet connection and retry
make clean && make data

# Solution 2: Reduce stock list for testing
make tickers-list                    # See current stocks
uv run cli.py tickers update --set AAPL MSFT  # Use just 2 stocks for testing
# Or: make tickers-reset && uv run cli.py tickers update --set AAPL MSFT
```

**Problem: Invalid ticker errors or typos in ticker list**

```bash
# Solution 1: Validate your ticker list
uv run cli.py tickers validate      # Shows which tickers are invalid

# Solution 2: Fix automatically suggested
uv run cli.py tickers update --remove INVALID_TICKER  # Remove problematic ticker

# Solution 3: Start fresh with defaults
uv run cli.py tickers update --reset-to-defaults
```

**Problem: Pipeline fails with "YFTzMissingError" or "Quote not found"**

```bash
# This happens when tickers are delisted or misspelled
# Solution: Run validation and clean up invalid tickers
uv run cli.py tickers validate
# Then follow the suggested fix commands
```

**Problem: Model training takes too long**

```bash
# Solution 1: Use CPU-only for faster setup
# Edit src/models/train_transformer.py, set device='cpu'

# Solution 2: Train only one model type
make setup
uv run src/models/train_lightgbm.py  # Fast gradient boosting only
```

**Problem: "Permission denied" or file access errors**

```bash
# Solution: Check file permissions and available disk space
df -h  # Check disk space
```

**Problem: Missing dependencies or import errors**

```bash
# Solution: Re-sync environment
make clean
make setup
uv run cli.py validate  # Should show all green checkmarks
```

**Problem: API rate limits (with API keys configured)**

```bash
# Solution: The tool automatically handles rate limits, but if you see errors:
# - FRED API: Wait 1 minute between runs
# - Finnhub API: Wait 1 minute between runs
# - Or run without API keys (limited features but fully functional)
```

**Problem: Predictions seem unrealistic**

```bash
# Solution: This is normal! Remember:
# - Stock prediction is inherently uncertain
# - Models predict probabilities, not certainties
# - Check model confidence in reports/model_comparison.json
# - Directional accuracy >60% is considered good
```

### Getting Help<a name="getting-help"></a>

1. **Validate setup**: `uv run cli.py validate` should pass all checks
1. **Check logs**: Error messages usually indicate the specific issue
1. **Start simple**: Try with just 2-3 stocks first
1. **Review documentation**: Each module has detailed docstrings
1. **Check disk space**: Pipeline generates ~1-2GB of data and models

## Contributing<a name="contributing-1"></a>
