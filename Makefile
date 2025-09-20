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
        cli-data cli-models cli-backtest cli-explain cli-test cli-validate cli-all \
        tickers-list tickers-defaults tickers-reset pipeline-data pipeline-models pipeline-analysis

help: ## Show this help message
	@echo "$(BLUE)Stock Trends Prediction Project$(RESET)"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Setup project environment and dependencies
	@echo "$(YELLOW)Setting up project environment with uv...$(RESET)"
	@if ! command -v uv &> /dev/null; then \
		echo "$(RED)Error: uv is not installed. Please install it first:$(RESET)"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  or: brew install uv"; \
		exit 1; \
	fi
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

data: ## Download and preprocess data
	@echo "$(YELLOW)Downloading and preprocessing data...$(RESET)"
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed
	$(PYTHON) $(SRC_DIR)/ingestion/equity_prices.py
	$(PYTHON) $(SRC_DIR)/ingestion/macro_data.py
	$(PYTHON) $(SRC_DIR)/ingestion/news_data.py
	$(PYTHON) $(SRC_DIR)/preprocess/weekly_aggregator.py
	@echo "$(GREEN)Data processing complete!$(RESET)"

features: ## Generate features for modeling
	@echo "$(YELLOW)Generating features...$(RESET)"
	$(PYTHON) $(SRC_DIR)/features/feature_engineering.py
	@echo "$(GREEN)Feature engineering complete!$(RESET)"

splits: ## Create train/validation/test splits
	@echo "$(YELLOW)Creating data splits...$(RESET)"
	@echo "Note: Data splits are integrated into feature engineering process"
	@echo "$(GREEN)Data splits process integrated!$(RESET)"

models: ## Train baseline models (LightGBM and Transformer)
	@echo "$(YELLOW)Training baseline models...$(RESET)"
	mkdir -p $(MODELS_DIR)
	@echo "Training LightGBM model..."
	$(PYTHON) $(SRC_DIR)/models/train_lightgbm.py
	@echo "Training Transformer model..."
	$(PYTHON) $(SRC_DIR)/models/train_transformer.py
	@echo "Comparing models..."
	$(PYTHON) $(SRC_DIR)/models/compare_models.py
	@echo "$(GREEN)Model training complete!$(RESET)"

backtest: ## Run backtesting analysis
	@echo "$(YELLOW)Running backtesting analysis...$(RESET)"
	$(PYTHON) $(SRC_DIR)/backtest/simple_backtest.py
	@echo "$(GREEN)Backtesting complete!$(RESET)"

explainability: ## Generate model explainability reports
	@echo "$(YELLOW)Generating explainability reports...$(RESET)"
	mkdir -p $(REPORTS_DIR)
	$(PYTHON) $(SRC_DIR)/explainability/model_explainer.py
	$(PYTHON) $(SRC_DIR)/explainability/shap_explainer.py
	$(PYTHON) $(SRC_DIR)/explainability/model_validator.py
	@echo "$(GREEN)Explainability analysis complete!$(RESET)"

test: ## Run unit tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)Tests complete!$(RESET)"

lint: ## Run code quality checks
	@echo "$(YELLOW)Running code quality checks...$(RESET)"
	$(PYTHON) -m flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	$(PYTHON) -m black $(SRC_DIR) --check --line-length=100
	@echo "$(GREEN)Code quality checks complete!$(RESET)"

format: ## Format code with black
	@echo "$(YELLOW)Formatting code...$(RESET)"
	$(PYTHON) -m black $(SRC_DIR) --line-length=100
	@echo "$(GREEN)Code formatting complete!$(RESET)"

# Pipeline targets
pipeline-data: setup data features splits ## Complete data pipeline

pipeline-models: pipeline-data models ## Complete model training pipeline

pipeline-analysis: pipeline-models backtest explainability ## Complete analysis pipeline

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

# CLI-based alternatives (using unified CLI interface)
cli-data: ## Run data pipeline via CLI
	@echo "$(YELLOW)Running data pipeline via CLI...$(RESET)"
	$(CLI) data

cli-models: ## Train models via CLI  
	@echo "$(YELLOW)Training models via CLI...$(RESET)"
	$(CLI) models

cli-backtest: ## Run backtesting via CLI
	@echo "$(YELLOW)Running backtesting via CLI...$(RESET)"
	$(CLI) backtest

cli-explain: ## Generate explainability reports via CLI
	@echo "$(YELLOW)Generating explainability via CLI...$(RESET)"
	$(CLI) explain

cli-test: ## Run tests via CLI
	@echo "$(YELLOW)Running tests via CLI...$(RESET)"
	$(CLI) test

cli-validate: ## Validate setup via CLI
	@echo "$(YELLOW)Validating setup via CLI...$(RESET)"
	$(CLI) validate

cli-all: ## Run complete pipeline via CLI
	@echo "$(YELLOW)Running complete pipeline via CLI...$(RESET)"
	$(CLI) all

# Ticker management
tickers-list: ## List current stock tickers
	@echo "$(YELLOW)Listing current tickers...$(RESET)"
	$(CLI) tickers list

tickers-defaults: ## Show default ticker options
	@echo "$(YELLOW)Showing default ticker options...$(RESET)"
	$(CLI) tickers defaults

tickers-reset: ## Reset to default tickers (40 S&P 500 stocks)
	@echo "$(YELLOW)Resetting to default tickers...$(RESET)"
	$(CLI) tickers update --reset-to-defaults

# Utility targets
check-env: ## Check if uv environment is ready
	@echo "$(YELLOW)Checking uv environment...$(RESET)"
	@if ! command -v uv &> /dev/null; then \
		echo "$(RED)Error: uv is not installed. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)uv environment ready!$(RESET)"

install-dev: check-env ## Install development dependencies with uv
	@echo "$(YELLOW)Installing development dependencies with uv...$(RESET)"
	$(UV) add --dev pytest flake8 black pytest-cov
	@echo "$(GREEN)Development dependencies installed!$(RESET)"

# Documentation
docs: ## Generate project documentation
	@echo "$(YELLOW)Generating documentation...$(RESET)"
	@echo "# Stock Trends Prediction Project" > README_generated.md
	@echo "" >> README_generated.md
	@echo "## Quick Start" >> README_generated.md
	@echo "\`\`\`bash" >> README_generated.md
	@echo "# Setup environment" >> README_generated.md
	@echo "make setup" >> README_generated.md
	@echo "" >> README_generated.md
	@echo "# Run complete pipeline" >> README_generated.md
	@echo "make all" >> README_generated.md
	@echo "\`\`\`" >> README_generated.md
	@echo "" >> README_generated.md
	@echo "## Available Commands" >> README_generated.md
	@make help | tail -n +4 >> README_generated.md
	@echo "$(GREEN)Documentation generated: README_generated.md$(RESET)"

# Docker support (optional)
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker build -t stock-trends .
	@echo "$(GREEN)Docker image built!$(RESET)"

docker-run: ## Run pipeline in Docker
	@echo "$(YELLOW)Running pipeline in Docker...$(RESET)"
	docker run -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models -v $(PWD)/reports:/app/reports stock-trends
	@echo "$(GREEN)Docker run complete!$(RESET)"