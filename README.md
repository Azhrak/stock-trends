# Stock Trends Prediction Project

A comprehensive machine learning pipeline for predicting stock price movements using technical analysis, time series modeling, and explainable AI techniques.

## Project Overview

This project implements a complete end-to-end system for predicting 12-week forward stock returns using:

- **Technical Analysis Features**: 80+ engineered features including moving averages, momentum indicators, volatility measures, and price patterns
- **Baseline Models**: LightGBM gradient boosting and Temporal Transformer neural networks
- **Walk-Forward Validation**: Realistic backtesting with time-aware cross-validation
- **Explainable AI**: SHAP analysis and feature importance for model interpretability
- **Realistic Trading Simulation**: Transaction costs, slippage, and position sizing

## Key Results

### Model Performance
- **Transformer Model**: 0.102 ± 0.005 RMSE, 67.3% ± 5.6% directional accuracy
- **LightGBM Model**: 0.109 ± 0.016 RMSE, 69.9% ± 7.3% directional accuracy
- **Transformer wins overall** with 6.4% better test RMSE

### Key Insights
- **Most important feature**: On-Balance Volume (OBV) by both traditional and SHAP analysis
- **Feature categories**: Momentum indicators are highly predictive, volatility measures crucial for risk assessment
- **Investment implications**: Trend-following strategies may be effective, volume confirmation valuable

## Quick Start

### Prerequisites
- Python 3.10+
- **uv** (fast Python package manager) - [Install uv](https://docs.astral.sh/uv/)
- macOS, Linux, or Windows
- Internet connection for data download

### Setup and Run
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

### Environment Variables (Optional)

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

### Alternative CLI Usage
```bash
# Using the CLI directly (uv automatically manages the environment)
python cli.py validate    # Validate setup
python cli.py all          # Run complete pipeline
python cli.py models       # Train models only
python cli.py explain      # Generate explainability reports

# Or run with uv explicitly
uv run python cli.py validate
uv run python cli.py all
```

## How to Use This Tool

### What This Tool Does
This tool predicts **12-week forward stock returns** using machine learning models trained on technical analysis indicators. It helps you:

1. **Identify trending stocks** - Find stocks likely to move up or down over the next 3 months
2. **Understand why predictions work** - SHAP analysis explains which technical indicators drive predictions
3. **Validate trading strategies** - Realistic backtesting with transaction costs and slippage
4. **Compare ML approaches** - LightGBM vs Transformer models with performance metrics

### Input: What You Provide
The tool works with **any publicly traded stocks**. You can analyze:

```python
# Default: Large-cap tech stocks (in src/ingestion/equity_prices.py)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

# Customize: Edit the ticker list to analyze your stocks
CUSTOM_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMD", "INTC"]  # ETFs and individual stocks
```

**To analyze your own stocks:**
1. Edit `src/ingestion/equity_prices.py` 
2. Replace `DEFAULT_TICKERS` with your stock list
3. Run the pipeline: `make all`

### Expected Outcomes: What You Get

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

### Step-by-Step Usage Guide

#### Step 1: Complete Pipeline (Recommended First Run)
```bash
make all
# This runs: data download → feature engineering → model training → backtesting → analysis
# Takes: 15-30 minutes depending on number of stocks and internet speed
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
# Edit src/ingestion/equity_prices.py
# Change DEFAULT_TICKERS = ["YOUR", "STOCKS", "HERE"]
make data features models  # Re-run pipeline with new stocks
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

### Real-World Application Examples

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

### Understanding the Output

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

### Performance & Requirements

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

## Project Structure

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
└── requirements.txt     # Python dependencies
```

## Available Commands

### Make Commands
```bash
make setup              # Setup virtual environment
make data               # Download and preprocess data
make features           # Generate features
make models             # Train baseline models
make backtest           # Run backtesting analysis
make explainability     # Generate explainability reports
make test               # Run unit tests
make all                # Run complete pipeline
```

### CLI Commands
```bash
python cli.py data      # Data pipeline
python cli.py models    # Model training
python cli.py backtest  # Backtesting
python cli.py explain   # Explainability
python cli.py test      # Unit tests
python cli.py validate  # Validate setup
```

## Pipeline Components

### 1. Data Ingestion
- Downloads stock data from Yahoo Finance (yfinance)
- 30 tickers, 5+ years of daily data
- Converts to weekly frequency for modeling

### 2. Feature Engineering
- **Price Features**: Returns, moving averages, price ratios
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Volume Features**: OBV, volume moving averages, volume ratios
- **Volatility Features**: Realized volatility across multiple timeframes
- **Support/Resistance**: Price levels and position within ranges

### 3. Model Training
- **Walk-Forward Cross-Validation**: 5 splits with realistic temporal constraints
- **LightGBM**: Gradient boosted trees with hyperparameter optimization
- **Temporal Transformer**: Neural network with attention mechanism for sequences
- **Early Stopping**: Prevents overfitting with validation monitoring

### 4. Backtesting Framework
- **Realistic Trading Costs**: Commission (0.1%), bid-ask spread (0.05%), market impact
- **Position Sizing**: Risk-based position sizing with volatility targeting
- **Performance Metrics**: Sharpe ratio, maximum drawdown, hit rate analysis

### 5. Explainability Analysis
- **Feature Importance**: Traditional LightGBM importance rankings
- **SHAP Analysis**: Individual prediction explanations and global feature attribution
- **Model Validation**: Performance benchmarking and consistency checks

## Output Reports

The pipeline generates comprehensive reports in the `reports/` directory:

### Visualizations
- `shap_summary_plot.png` - SHAP feature impact visualization
- `shap_feature_importance.png` - Global feature importance
- `shap_dependence_*.png` - Feature dependence plots for top 5 features
- `lightgbm_feature_importance.png` - Traditional feature importance
- `prediction_analysis.png` - Prediction vs actual analysis

### Data Reports
- `lightgbm_explainability_report_*.json` - Comprehensive LightGBM analysis
- `prediction_explanation_report_*.json` - SHAP-based prediction explanations
- `validation_report_*.json` - Model validation and performance summary
- `model_comparison_report.json` - Head-to-head model comparison

## Testing

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
python cli.py test --coverage  # with coverage report
```

## Model Interpretability

### Top Features (by SHAP importance)
1. **OBV** (On-Balance Volume) - 0.0058
2. **Volatility 12w** - 0.0045
3. **Price Position 26w** - 0.0041
4. **Support 52w** - 0.0026
5. **ATR % 12w** - 0.0025

### Investment Implications
- **Momentum indicators** are highly predictive → trend-following strategies effective
- **Volatility measures** are important → risk assessment crucial
- **Volume confirmation** valuable → volume-based signals significant
- **Technical indicators** dominate → technical analysis approaches supported
- **Price levels** influential → support/resistance analysis matters

## Configuration

### Model Parameters
- **LightGBM**: Learning rate 0.05, early stopping after 100 rounds
- **Transformer**: 6 layers, 8 attention heads, 128 hidden dimensions
- **Training**: 5 walk-forward splits, 12-week prediction horizon

### Trading Parameters  
- **Transaction Costs**: 0.128% total (commission + spread + impact + slippage)
- **Position Sizing**: Max 5% per position, 15% volatility target
- **Rebalancing**: Weekly with 12-week holding period

## Dependencies

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

## Development

### Code Quality
```bash
make lint     # Code quality checks
make format   # Auto-format code
```

### Adding New Features
1. Add feature engineering logic to `src/features/feature_engineering.py`
2. Update feature lists in model training scripts
3. Add unit tests in `tests/`
4. Run validation: `python cli.py test`

### Adding New Models
1. Create model class in `src/models/`
2. Implement training loop following existing patterns
3. Add to comparison script `src/models/compare_models.py`
4. Update explainability analysis if needed

## Performance Benchmarks

### Expected Results
- **RMSE**: < 0.15 (both models achieve ~0.10)
- **Directional Accuracy**: > 55% (Transformer achieves 67%)
- **Training Time**: ~2-5 minutes per model on modern CPU
- **Memory Usage**: < 2GB RAM for full pipeline

### Reproducibility
- Fixed random seeds ensure consistent results
- Unit tests validate reproducibility across runs
- Virtual environment pins exact dependency versions

## Contributing

1. Follow existing code structure and patterns
2. Add unit tests for new functionality
3. Run full test suite: `make test`
4. Update documentation as needed
5. Use black code formatting: `make format`

## License

This project is for educational and research purposes. See individual data source terms for usage restrictions.

## References

- **Technical Analysis**: Murphy, J. "Technical Analysis of the Financial Markets"
- **SHAP**: Lundberg, S. "A Unified Approach to Interpreting Model Predictions"
- **Walk-Forward Analysis**: Pardo, R. "The Evaluation and Optimization of Trading Strategies"
- **Data Source**: Yahoo Finance via yfinance library

---

**Built with dedication for the stock prediction community**

## Troubleshooting

### Common Issues & Solutions

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
# Edit src/ingestion/equity_prices.py, use fewer tickers like ["AAPL", "MSFT"]
```

**Problem: Model training takes too long**
```bash
# Solution 1: Use CPU-only for faster setup
# Edit src/models/train_transformer.py, set device='cpu'

# Solution 2: Train only one model type
make setup
uv run python src/models/train_lightgbm.py  # Fast gradient boosting only
```

**Problem: "Permission denied" or file access errors**
```bash
# Solution: Check file permissions and available disk space
chmod +x cli.py
df -h  # Check disk space
```

**Problem: Missing dependencies or import errors**
```bash
# Solution: Re-sync environment
make clean
make setup
python cli.py validate  # Should show all green checkmarks
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

### Getting Help

1. **Validate setup**: `python cli.py validate` should pass all checks
2. **Check logs**: Error messages usually indicate the specific issue
3. **Start simple**: Try with just 2-3 stocks first
4. **Review documentation**: Each module has detailed docstrings
5. **Check disk space**: Pipeline generates ~1-2GB of data and models

## Contributing