# Stock Trends Prediction Project Makefile
# Provides convenient commands for project management and reproducibility

# Default shell
SHELL := /bin/bash

# Project directories
PROJECT_ROOT := $(shell pwd)
SRC_DIR := src
DATA_DIR := data
MODELS_DIR := models
REPORTS_DIR := reports

# UV and Python interpreter  
PYTHON := uv run python
UV := uv
CLI := uv run cli.py

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

.PHONY: help setup clean data features models backtest explainability all test lint \
        init-dirs check-env ticker-list analyze detailed fresh \
        current-analysis pipeline-data pipeline-models pipeline-analysis

# Initialize project directories
init-dirs:
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODELS_DIR) $(REPORTS_DIR)

help: ## Show this help message
	@echo "$(BLUE)Stock Trends Prediction Project$(RESET)"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'

# Environment management
check-env: ## Check if uv environment is ready
	@if ! command -v uv &> /dev/null; then \
		echo "$(RED)Error: uv is not installed. Please install it first:$(RESET)"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  or: brew install uv"; \
		exit 1; \
	fi

setup: check-env ## Setup project environment and dependencies
	@echo "$(YELLOW)Setting up project environment with uv...$(RESET)"
	$(UV) sync
	@echo "$(GREEN)Environment setup complete!$(RESET)"

clean: ## Clean temporary files and cache
	@echo "$(YELLOW)Cleaning project...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN)Project cleaned!$(RESET)"

data: check-env init-dirs ## Download and preprocess data
	@echo "$(YELLOW)Downloading and preprocessing data...$(RESET)"
	$(PYTHON) $(SRC_DIR)/ingestion/equity_prices.py
	$(PYTHON) $(SRC_DIR)/ingestion/macro_data.py
	$(PYTHON) $(SRC_DIR)/ingestion/news_data.py
	$(PYTHON) $(SRC_DIR)/preprocess/weekly_aggregator.py
	@echo "$(GREEN)Data processing complete!$(RESET)"

features: data ## Generate features for modeling
	@echo "$(YELLOW)Generating features...$(RESET)"
	$(PYTHON) $(SRC_DIR)/features/feature_engineering.py
	@echo "$(GREEN)Feature engineering complete!$(RESET)"

models: features init-dirs ## Train baseline models (LightGBM and Transformer)
	@echo "$(YELLOW)Training baseline models...$(RESET)"
	@echo "Training LightGBM model..."
	$(PYTHON) $(SRC_DIR)/models/train_lightgbm.py
	@echo "Training Transformer model..."
	$(PYTHON) $(SRC_DIR)/models/train_transformer.py
	@echo "Comparing models..."
	$(PYTHON) $(SRC_DIR)/models/compare_models.py
	@echo "$(GREEN)Model training complete!$(RESET)"

backtest: models ## Run backtesting analysis
	@echo "$(YELLOW)Running backtesting analysis...$(RESET)"
	$(PYTHON) $(SRC_DIR)/backtest/simple_backtest.py
	@echo "$(GREEN)Backtesting complete!$(RESET)"

explainability: models init-dirs ## Generate model explainability reports
	@echo "$(YELLOW)Generating explainability reports...$(RESET)"
	$(PYTHON) $(SRC_DIR)/explainability/model_explainer.py
	$(PYTHON) $(SRC_DIR)/explainability/shap_explainer.py
	$(PYTHON) $(SRC_DIR)/explainability/model_validator.py
	@echo "$(GREEN)Explainability analysis complete!$(RESET)"

# Development and Testing
test: check-env ## Run unit tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)Tests complete!$(RESET)"

lint: check-env ## Run code quality checks
	@echo "$(YELLOW)Running code quality checks...$(RESET)"
	$(PYTHON) -m flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	$(PYTHON) -m black $(SRC_DIR) --check --line-length=100
	@echo "$(GREEN)Code quality checks complete!$(RESET)"

format: check-env ## Format code with black
	@echo "$(YELLOW)Formatting code...$(RESET)"
	$(PYTHON) -m black $(SRC_DIR) --line-length=100
	@echo "$(GREEN)Code formatting complete!$(RESET)"

install-dev: check-env ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(RESET)"
	$(UV) add --dev pytest flake8 black pytest-cov
	@echo "$(GREEN)Development dependencies installed!$(RESET)"

# Pipeline targets
pipeline-data: data ## Complete data pipeline

pipeline-models: models ## Complete model training pipeline  

pipeline-analysis: backtest explainability ## Complete analysis pipeline

all: pipeline-analysis ## Run complete end-to-end pipeline
	@echo "$(GREEN)===============================================$(RESET)"
	@echo "$(GREEN)  COMPLETE PIPELINE EXECUTION FINISHED!$(RESET)"
	@echo "$(GREEN)===============================================$(RESET)"
	@echo ""
	@echo "$(BLUE)Results Summary:$(RESET)"
	@echo "- Data processed and features engineered"
	@echo "- Baseline models trained and compared"
	@echo "- Backtesting analysis completed"
	@echo "- Explainability reports generated"
	@echo ""
	@echo "$(BLUE)Check these directories for outputs:$(RESET)"
	@echo "- $(MODELS_DIR)/ - Trained models and results"
	@echo "- $(REPORTS_DIR)/ - Analysis reports and visualizations"
	@echo "- $(DATA_DIR)/ - Processed data and splits"

# Analysis Commands
current-analysis: explainability ## Generate analysis with current data (latest split)
	@echo "$(YELLOW)Generating analysis with current data (latest split)...$(RESET)"
	$(PYTHON) -c "import sys; sys.path.append('src'); from explainability.shap_explainer import SHAPExplainer; explainer = SHAPExplainer(); explainer.create_prediction_explanation_report(split_id=15)"
	@echo "$(GREEN)Current data analysis complete! Use ticker-analyze to view results.$(RESET)"

# Ticker Analysis Commands
ticker-list: ## List all available tickers in explainability reports
	@echo "$(YELLOW)Listing available tickers for analysis...$(RESET)"
	$(PYTHON) ticker_analysis.py --list

# Clean syntax: make analyze AAPL, make detailed AAPL, make fresh AAPL
analyze: ## Analyze specific ticker (usage: make analyze AAPL)
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "$(RED)Error: Please specify ticker. Example: make analyze AAPL$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Analyzing ticker: $(word 2,$(MAKECMDGOALS))...$(RESET)"
	$(PYTHON) ticker_analysis.py $(word 2,$(MAKECMDGOALS))

detailed: ## Detailed analysis of specific ticker (usage: make detailed AAPL)
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "$(RED)Error: Please specify ticker. Example: make detailed TSLA$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running detailed analysis for ticker: $(word 2,$(MAKECMDGOALS))...$(RESET)"
	$(PYTHON) ticker_analysis.py $(word 2,$(MAKECMDGOALS)) --detailed

fresh: ## Generate fresh analysis + analyze ticker (usage: make fresh AAPL)
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "$(RED)Error: Please specify ticker. Example: make fresh AAPL$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Generating fresh analysis for $(word 2,$(MAKECMDGOALS))...$(RESET)"
	@make current-analysis > /dev/null 2>&1
	$(PYTHON) ticker_analysis.py $(word 2,$(MAKECMDGOALS))

# Suppress "No rule to make target" errors for ticker symbols
%:
	@:

# Documentation and Docker (Optional)
docs: ## Generate project documentation
	@echo "$(YELLOW)Generating documentation...$(RESET)"
	@echo "# Stock Trends Prediction Project" > README_generated.md
	@echo "" >> README_generated.md
	@echo "## Quick Start" >> README_generated.md
	@echo "\`\`\`bash" >> README_generated.md
	@echo "make setup && make all" >> README_generated.md
	@echo "\`\`\`" >> README_generated.md
	@echo "" >> README_generated.md
	@echo "## Available Commands" >> README_generated.md
	@make help | tail -n +4 >> README_generated.md
	@echo "$(GREEN)Documentation generated: README_generated.md$(RESET)"

toc: ## Generate table of contents for README.md
	@echo "$(YELLOW)Generating table of contents for README.md...$(RESET)"
	$(UV) run mdformat --wrap=no --end-of-line=keep README.md
	@echo "$(GREEN)Table of contents updated in README.md$(RESET)"

toc-preview: ## Preview table of contents without modifying README.md
	@echo "$(YELLOW)Previewing table of contents...$(RESET)"
	$(UV) run mdformat --check --wrap=no --end-of-line=keep README.md

docker-build: ## Build Docker image (optional)
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker build -t stock-trends .
	@echo "$(GREEN)Docker image built!$(RESET)"

docker-run: docker-build ## Run pipeline in Docker (optional)
	@echo "$(YELLOW)Running pipeline in Docker...$(RESET)"
	docker run -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models -v $(PWD)/reports:/app/reports stock-trends
	@echo "$(GREEN)Docker run complete!$(RESET)"