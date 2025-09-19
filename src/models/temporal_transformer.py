"""
Temporal Transformer model for stock return prediction.
Implements a transformer-based architecture for time series forecasting.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """PyTorch Dataset for stock data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, sequence_length: int = 52):
        """
        Initialize dataset.
        
        Args:
            X: Feature dataframe
            y: Target series  
            sequence_length: Number of time steps to use for sequence
        """
        self.sequence_length = sequence_length
        
        # Convert to numpy arrays
        self.features = X.values.astype(np.float32)
        self.targets = y.values.astype(np.float32)
        
        # Group by ticker to create sequences
        self.sequences = []
        self.sequence_targets = []
        
        # Assuming ticker is in the index or a column
        if 'ticker' in X.columns:
            tickers = X['ticker'].unique()
            for ticker in tickers:
                ticker_mask = X['ticker'] == ticker
                ticker_features = self.features[ticker_mask]
                ticker_targets = self.targets[ticker_mask]
                
                # Create sequences for this ticker
                for i in range(len(ticker_features) - sequence_length + 1):
                    self.sequences.append(ticker_features[i:i + sequence_length])
                    self.sequence_targets.append(ticker_targets[i + sequence_length - 1])
        else:
            # If no ticker column, treat as single sequence
            for i in range(len(self.features) - sequence_length + 1):
                self.sequences.append(self.features[i:i + sequence_length])
                self.sequence_targets.append(self.targets[i + sequence_length - 1])
        
        self.sequences = np.array(self.sequences)
        self.sequence_targets = np.array(self.sequence_targets)
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.sequence_targets[idx])

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalTransformer(nn.Module):
    """Transformer model for temporal prediction."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Use the last time step for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Apply output layers
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch_size, 1)
        
        return x.squeeze(-1)  # (batch_size,)

class TemporalTransformerModel:
    """Wrapper class for temporal transformer model."""
    
    def __init__(
        self,
        model_dir: str = "models",
        splits_dir: str = "data/processed/splits",
        sequence_length: int = 52,
        device: str = None
    ):
        """
        Initialize the temporal transformer model.
        
        Args:
            model_dir: Directory to save model artifacts
            splits_dir: Directory containing split data
            sequence_length: Number of time steps for sequences
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_dir = model_dir
        self.splits_dir = splits_dir
        self.sequence_length = sequence_length
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model parameters
        self.model_params = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.1
        }
        
        # Training parameters
        self.training_params = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'patience': 10,
            'weight_decay': 1e-5
        }
        
        self.models = {}  # Store models for each split
        self.scalers = {}  # Store scalers for each split
        self.feature_names = None
        self.training_history = []
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, scaler: Optional[StandardScaler] = None):
        """
        Prepare data for training.
        
        Args:
            X: Feature dataframe
            y: Target series
            scaler: Optional pre-fitted scaler
            
        Returns:
            Tuple of (scaled_X, y, scaler)
        """
        # Scale features
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, y, scaler
    
    def load_split_data(self, split_id: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load data for a specific split.
        
        Args:
            split_id: Split identifier
            
        Returns:
            Tuple of (train_X, train_y, val_X, val_y, test_X, test_y)
        """
        split_dir = os.path.join(self.splits_dir, f"split_{split_id}")
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load training data
        train_X = pd.read_parquet(os.path.join(split_dir, "train_X.parquet"))
        train_y = pd.read_parquet(os.path.join(split_dir, "train_y.parquet"))['target']
        
        # Load validation data
        val_X = pd.read_parquet(os.path.join(split_dir, "val_X.parquet"))
        val_y = pd.read_parquet(os.path.join(split_dir, "val_y.parquet"))['target']
        
        # Load test data
        test_X = pd.read_parquet(os.path.join(split_dir, "test_X.parquet"))
        test_y = pd.read_parquet(os.path.join(split_dir, "test_y.parquet"))['target']
        
        # Store feature names if not already set
        if self.feature_names is None:
            self.feature_names = train_X.columns.tolist()
        
        logger.info(f"Loaded split {split_id}: Train {len(train_X)}, Val {len(val_X)}, Test {len(test_X)}")
        
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def train_single_split(
        self,
        split_id: int,
        model_params: Optional[Dict] = None,
        training_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train model on a single split.
        
        Args:
            split_id: Split identifier
            model_params: Model hyperparameters
            training_params: Training hyperparameters
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training Temporal Transformer for split {split_id}")
        
        # Load data
        train_X, train_y, val_X, val_y, test_X, test_y = self.load_split_data(split_id)
        
        # Use default parameters if none provided
        if model_params is None:
            model_params = self.model_params.copy()
        if training_params is None:
            training_params = self.training_params.copy()
        
        # Prepare data
        train_X_scaled, train_y, scaler = self.prepare_data(train_X, train_y)
        val_X_scaled, val_y, _ = self.prepare_data(val_X, val_y, scaler)
        test_X_scaled, test_y, _ = self.prepare_data(test_X, test_y, scaler)
        
        # Store scaler
        self.scalers[split_id] = scaler
        
        # Create datasets
        train_dataset = StockDataset(train_X_scaled, train_y, self.sequence_length)
        val_dataset = StockDataset(val_X_scaled, val_y, self.sequence_length)
        test_dataset = StockDataset(test_X_scaled, test_y, self.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=training_params['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        model = TemporalTransformer(
            input_dim=len(self.feature_names),
            **model_params
        ).to(self.device)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_params['learning_rate'],
            weight_decay=training_params['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(training_params['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{training_params['num_epochs']}: "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= training_params['patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        self.models[split_id] = model
        
        # Make predictions
        train_pred = self._predict_dataset(model, train_loader)
        val_pred = self._predict_dataset(model, val_loader)
        test_pred = self._predict_dataset(model, test_loader)
        
        # Get actual values (need to align with sequences)
        train_actual = train_dataset.sequence_targets
        val_actual = val_dataset.sequence_targets
        test_actual = test_dataset.sequence_targets
        
        # Calculate metrics
        results = {
            'split_id': split_id,
            'final_epoch': epoch + 1,
            'best_val_loss': float(best_val_loss),
            'train_rmse': float(np.sqrt(mean_squared_error(train_actual, train_pred))),
            'train_mae': float(mean_absolute_error(train_actual, train_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(val_actual, val_pred))),
            'val_mae': float(mean_absolute_error(val_actual, val_pred)),
            'test_rmse': float(np.sqrt(mean_squared_error(test_actual, test_pred))),
            'test_mae': float(mean_absolute_error(test_actual, test_pred)),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Calculate directional accuracy
        train_dir_acc = accuracy_score(train_actual > 0, train_pred > 0)
        val_dir_acc = accuracy_score(val_actual > 0, val_pred > 0)
        test_dir_acc = accuracy_score(test_actual > 0, test_pred > 0)
        
        results.update({
            'train_dir_accuracy': float(train_dir_acc),
            'val_dir_accuracy': float(val_dir_acc),
            'test_dir_accuracy': float(test_dir_acc)
        })
        
        logger.info(f"Split {split_id} - Val RMSE: {results['val_rmse']:.4f}, "
                   f"Test RMSE: {results['test_rmse']:.4f}, "
                   f"Test Dir Acc: {results['test_dir_accuracy']:.3f}")
        
        return results
    
    def _predict_dataset(self, model: nn.Module, data_loader: DataLoader) -> np.ndarray:
        """Make predictions on a dataset."""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def train_all_splits(
        self,
        max_splits: Optional[int] = None,
        model_params: Optional[Dict] = None,
        training_params: Optional[Dict] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Train models on all available splits.
        
        Args:
            max_splits: Maximum number of splits to train (None for all)
            model_params: Model hyperparameters
            training_params: Training hyperparameters
            save_results: Whether to save results to disk
            
        Returns:
            List of training results for each split
        """
        logger.info("Training Temporal Transformer models on all splits")
        
        # Find available splits
        available_splits = []
        for split_dir in os.listdir(self.splits_dir):
            if split_dir.startswith('split_') and split_dir[6:].isdigit():
                split_id = int(split_dir[6:])
                available_splits.append(split_id)
        
        available_splits.sort()
        
        if max_splits is not None:
            available_splits = available_splits[:max_splits]
        
        logger.info(f"Training on {len(available_splits)} splits: {available_splits}")
        
        # Train models
        all_results = []
        for split_id in available_splits:
            try:
                results = self.train_single_split(split_id, model_params, training_params)
                all_results.append(results)
                self.training_history.append(results)
            except Exception as e:
                logger.error(f"Failed to train split {split_id}: {str(e)}")
                continue
        
        # Calculate summary statistics
        if all_results:
            summary = self.calculate_summary_metrics(all_results)
            logger.info(f"Overall Results - Mean Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
            logger.info(f"Mean Test Dir Accuracy: {summary['mean_test_dir_accuracy']:.3f} ± {summary['std_test_dir_accuracy']:.3f}")
        
        # Save results
        if save_results and all_results:
            self.save_training_results(all_results, summary)
        
        return all_results
    
    def calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary statistics across all splits."""
        metrics = ['train_rmse', 'val_rmse', 'test_rmse', 'train_mae', 'val_mae', 'test_mae',
                  'train_dir_accuracy', 'val_dir_accuracy', 'test_dir_accuracy']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
        
        return summary
    
    def save_training_results(self, results: List[Dict[str, Any]], summary: Dict[str, float]):
        """Save training results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.model_dir, f"transformer_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.model_dir, f"transformer_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save models
        for split_id, model in self.models.items():
            model_file = os.path.join(self.model_dir, f"transformer_split_{split_id}_{timestamp}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_params': self.model_params,
                'scaler': self.scalers[split_id]
            }, model_file)
        
        logger.info(f"Saved training results to {self.model_dir}")
        logger.info(f"Results: {results_file}")
        logger.info(f"Summary: {summary_file}")

def main():
    """Example usage of the Temporal Transformer model."""
    
    # Initialize model
    transformer_model = TemporalTransformerModel(sequence_length=26)  # Half year sequences
    
    # Train on first split for demo
    results = transformer_model.train_single_split(0)
    
    print("\nTRANSFORMER RESULTS:")
    print("=" * 50)
    print(f"Test RMSE:        {results['test_rmse']:.4f}")
    print(f"Test Dir Accuracy: {results['test_dir_accuracy']:.3f}")
    print(f"Training Epochs:  {results['final_epoch']}")

if __name__ == "__main__":
    main()