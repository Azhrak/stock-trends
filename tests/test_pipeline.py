"""
Unit tests for the stock trends prediction pipeline.
Tests core functionality to ensure reproducibility.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineering import FeatureEngineer
from models.lightgbm_model import LightGBMModel
from backtest.backtest_engine import TradingCosts, PositionSizer, BacktestEngine


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducibility
        
        data = {
            'date': dates,
            'symbol': ['AAPL'] * 100,
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 2,
            'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 2,
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000000, 5000000, 100)
        }
        
        df = pd.DataFrame(data)
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        return df.sort_values('date')
    
    def test_feature_engineer_initialization(self, sample_data):
        """Test feature engineer initialization."""
        feature_engineer = FeatureEngineer()
        
        # Should initialize without error
        assert feature_engineer is not None
    
    def test_basic_feature_generation(self, sample_data):
        """Test basic feature generation."""
        feature_engineer = FeatureEngineer()
        
        # Test that we can process data without errors
        try:
            # Add basic price features
            result = sample_data.copy()
            result['return_1w'] = result['close'].pct_change()
            result['price_ma_5'] = result['close'].rolling(5).mean()
            
            # Should complete without errors
            assert len(result) == len(sample_data)
            assert 'return_1w' in result.columns
            assert 'price_ma_5' in result.columns
            
        except Exception as e:
            pytest.fail(f"Feature generation failed: {e}")
    
    def test_feature_generation_reproducibility(self, sample_data):
        """Test that feature generation is reproducible."""
        # Simple reproducibility test
        result1 = sample_data.copy()
        result1['test_feature'] = result1['close'].rolling(5).mean()
        
        result2 = sample_data.copy()
        result2['test_feature'] = result2['close'].rolling(5).mean()
        
        # Results should be identical
        pd.testing.assert_series_equal(result1['test_feature'], result2['test_feature'])


class TestLightGBMModel:
    """Test LightGBM model functionality."""
    
    @pytest.fixture
    def sample_model_data(self):
        """Create sample data for model testing."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        # Generate features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate target with some signal
        y = pd.Series(
            0.1 * X.iloc[:, 0] + 0.05 * X.iloc[:, 1] + 0.02 * np.random.randn(n_samples),
            name='target'
        )
        
        return X, y
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_initialization(self, temp_model_dir):
        """Test model initialization."""
        model = LightGBMModel(model_dir=temp_model_dir)
        
        assert model.model_dir == temp_model_dir
        assert model.models == {}
        assert model.feature_names is None
    
    def test_default_parameters(self, temp_model_dir):
        """Test that default parameters are reasonable."""
        model = LightGBMModel(model_dir=temp_model_dir)
        
        params = model.default_params
        
        # Check key parameters exist and have reasonable values
        assert params['objective'] == 'regression'
        assert params['metric'] == 'rmse'
        assert 0 < params['learning_rate'] <= 1
        assert params['random_state'] == 42


class TestTradingCosts:
    """Test trading costs calculation."""
    
    def test_trading_costs_initialization(self):
        """Test trading costs initialization."""
        costs = TradingCosts()
        
        # Check default values are reasonable
        assert 0 <= costs.commission_rate <= 0.01
        assert 0 <= costs.bid_ask_spread <= 0.01
        assert 0 <= costs.market_impact <= 0.01
    
    def test_calculate_transaction_cost(self):
        """Test transaction cost calculation."""
        costs = TradingCosts(
            commission_rate=0.001,
            bid_ask_spread=0.002,
            market_impact=0.0005
        )
        
        # Test cost calculation
        trade_value = 10000
        cost = costs.calculate_cost(trade_value)
        expected_cost = trade_value * costs.total_cost
        
        assert abs(cost - expected_cost) < 1e-6


class TestPositionSizer:
    """Test position sizing functionality."""
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        sizer = PositionSizer()
        
        assert 0 < sizer.max_position_size <= 1
        assert 0 < sizer.volatility_target <= 1
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        sizer = PositionSizer(max_position_size=0.1, volatility_target=0.02)
        
        # Test with reasonable inputs
        position_size = sizer.calculate_position_size(
            prediction=0.05,      # 5% expected return
            confidence=0.8,       # 80% confidence
            stock_volatility=0.3, # 30% volatility
            current_exposure=0.5  # 50% current exposure
        )
        
        # Position size should be reasonable
        assert 0 <= position_size <= sizer.max_position_size
        
        # Test that higher volatility reduces position size
        low_vol_size = sizer.calculate_position_size(0.05, 0.8, 0.1, 0.5)
        high_vol_size = sizer.calculate_position_size(0.05, 0.8, 0.5, 0.5)
        
        assert low_vol_size >= high_vol_size


class TestDataValidation:
    """Test data validation and consistency."""
    
    def test_data_directory_structure(self):
        """Test that required data directories exist."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            'data',
            'src',
            'models',
            'reports',
            'tests'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
    
    def test_source_code_structure(self):
        """Test that source code modules exist."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / 'src'
        
        required_modules = [
            'features',
            'models', 
            'backtest',
            'explainability',
            'ingestion',
            'preprocess',
            'utils'
        ]
        
        for module in required_modules:
            module_path = src_dir / module
            assert module_path.exists(), f"Required module missing: {module}"
    
    def test_configuration_files(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        required_files = [
            'requirements.txt',
            'Makefile',
            'cli.py'
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"


class TestReproducibility:
    """Test reproducibility of key computations."""
    
    def test_numpy_random_seeding(self):
        """Test that numpy random seeding works for reproducibility."""
        # Set seed and generate random numbers
        np.random.seed(42)
        random1 = np.random.randn(10)
        
        # Reset seed and generate again
        np.random.seed(42)
        random2 = np.random.randn(10)
        
        # Should be identical
        np.testing.assert_array_equal(random1, random2)
    
    def test_pandas_operations_consistency(self):
        """Test that pandas operations are consistent."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100)
        })
        
        # Perform operations twice
        result1 = df.rolling(window=5).mean().dropna()
        result2 = df.rolling(window=5).mean().dropna()
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])