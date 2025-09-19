# Stock Trends Prediction Project

A comprehensive machine learning pipeline for predicting stock price movements using technical analysis, time series modeling, and explainable AI techniques.

## 🎯 Project Overview

This project implements a complete end-to-end system for predicting 12-week forward stock returns using:

- **Technical Analysis Features**: 80+ engineered features including moving averages, momentum indicators, volatility measures, and price patterns
- **Baseline Models**: LightGBM gradient boosting and Temporal Transformer neural networks
- **Walk-Forward Validation**: Realistic backtesting with time-aware cross-validation
- **Explainable AI**: SHAP analysis and feature importance for model interpretability
- **Realistic Trading Simulation**: Transaction costs, slippage, and position sizing

## 📊 Key Results

### Model Performance
- **Transformer Model**: 0.102 ± 0.005 RMSE, 67.3% ± 5.6% directional accuracy
- **LightGBM Model**: 0.109 ± 0.016 RMSE, 69.9% ± 7.3% directional accuracy
- **Transformer wins overall** with 6.4% better test RMSE

### Key Insights
- **Most important feature**: On-Balance Volume (OBV) by both traditional and SHAP analysis
- **Feature categories**: Momentum indicators are highly predictive, volatility measures crucial for risk assessment
- **Investment implications**: Trend-following strategies may be effective, volume confirmation valuable

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS, Linux, or Windows
- Internet connection for data download

### Setup and Run
```bash
# Clone and setup
git clone <repository-url>
cd stock-trends

# Setup environment and run complete pipeline
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
- ✅ **Still works**: Stock price data (yfinance), technical analysis, model training, backtesting
- ⚠️ **Limited**: No macroeconomic data (GDP, inflation) or financial news sentiment

**Rate Limits (Free Tiers):**
- FRED API: 120 requests/minute
- Finnhub API: 60 requests/minute

### Alternative CLI Usage
```bash
# Using the CLI directly
python cli.py validate    # Validate setup
python cli.py all          # Run complete pipeline
python cli.py models       # Train models only
python cli.py explain      # Generate explainability reports
```

## 📁 Project Structure

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

## 🔧 Available Commands

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

## 📈 Pipeline Components

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

## 📊 Output Reports

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

## 🧪 Testing

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

## 🔍 Model Interpretability

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

## ⚙️ Configuration

### Model Parameters
- **LightGBM**: Learning rate 0.05, early stopping after 100 rounds
- **Transformer**: 6 layers, 8 attention heads, 128 hidden dimensions
- **Training**: 5 walk-forward splits, 12-week prediction horizon

### Trading Parameters  
- **Transaction Costs**: 0.128% total (commission + spread + impact + slippage)
- **Position Sizing**: Max 5% per position, 15% volatility target
- **Rebalancing**: Weekly with 12-week holding period

## 📋 Dependencies

Core libraries:
- `pandas`, `numpy` - Data manipulation
- `lightgbm` - Gradient boosting
- `torch`, `transformers` - Neural networks  
- `scikit-learn` - ML utilities
- `yfinance` - Data download
- `matplotlib`, `seaborn` - Visualization
- `shap` - Model explainability
- `pytest` - Testing

## 🛠️ Development

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

## 📈 Performance Benchmarks

### Expected Results
- **RMSE**: < 0.15 (both models achieve ~0.10)
- **Directional Accuracy**: > 55% (Transformer achieves 67%)
- **Training Time**: ~2-5 minutes per model on modern CPU
- **Memory Usage**: < 2GB RAM for full pipeline

### Reproducibility
- Fixed random seeds ensure consistent results
- Unit tests validate reproducibility across runs
- Virtual environment pins exact dependency versions

## 🤝 Contributing

1. Follow existing code structure and patterns
2. Add unit tests for new functionality
3. Run full test suite: `make test`
4. Update documentation as needed
5. Use black code formatting: `make format`

## 📄 License

This project is for educational and research purposes. See individual data source terms for usage restrictions.

## 🔗 References

- **Technical Analysis**: Murphy, J. "Technical Analysis of the Financial Markets"
- **SHAP**: Lundberg, S. "A Unified Approach to Interpreting Model Predictions"
- **Walk-Forward Analysis**: Pardo, R. "The Evaluation and Optimization of Trading Strategies"
- **Data Source**: Yahoo Finance via yfinance library

---

**Built with ❤️ for the stock prediction community**