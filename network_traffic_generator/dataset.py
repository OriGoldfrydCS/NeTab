import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random

class NetworkTrafficDataset(Dataset):
    """
    Dataset class for network traffic data with tokenization and augmentation
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        model,                              # Model for encoding network data
        max_length: int = 512,              # Maximum sequence length for tokenization
        augment_data: bool = True           # Apply data augmentation during training
    ):
        self.data = data.copy()             # Make a copy to avoid modifying the original DataFrame
        self.model = model                  # Model instance for encoding
        self.max_length = max_length        # Maximum length for token sequences
        self.augment_data = augment_data    # Whether to apply data augmentation
        
        # Prepare the data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and clean the data for training"""
        # Ensure all PPI columns are present (90 columns total: 3 groups x 30 features)
        expected_ppi_cols = [f"PPI_{i}_{j}" for i in range(3) for j in range(30)]
        
        for col in expected_ppi_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0  # Fill missing columns with zeros
        
        # Ensure APP column exists
        if "APP" not in self.data.columns:
            self.data["APP"] = "Unknown"
        
        # Convert data to list of dictionaries for easier processing
        self.samples = self.data.to_dict('records')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        
        # Apply data augmentation if enabled (30% chance during training)
        if self.augment_data and random.random() < 0.3:
            sample = self._augment_sample(sample)
        
        # Encode the sample using the model's tokenizer
        tokens = self.model.encode_network_data(sample)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Add EOS token
        tokens.append(self.model.tokenizer.eos_token_id)
        
        # Create attention mask and pad to max_length
        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.model.tokenizer.pad_token_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)  # For language modeling
        }
    
    def _augment_sample(self, sample):
        """Apply data augmentation to a sample by adding small noise"""
        augmented_sample = sample.copy()
        
        # Add small random noise to numerical PPI features only
        for key, value in augmented_sample.items():
            if key.startswith('PPI_') and isinstance(value, (int, float)):
                # Add small gaussian noise (±5% of value)
                noise_factor = 0.05
                noise = np.random.normal(0, abs(value) * noise_factor)
                augmented_sample[key] = max(0, value + noise)  # Ensure non-negative values
        
        return augmented_sample


def create_data_loader(
    data: pd.DataFrame,
    model,
    batch_size: int = 8,          # Number of samples per batch
    shuffle: bool = True,         # Randomize sample order
    max_length: int = 512,        # Maximum token sequence length
    augment_data: bool = True,    # Enable data augmentation
    num_workers: int = 0          # Number of parallel data loading processes
) -> DataLoader:
    """
    Create a DataLoader for network traffic data training
    
    Args:
        data: DataFrame containing network traffic data
        model: NetworkTrafficModel instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        augment_data: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = NetworkTrafficDataset(
        data=data,
        model=model,
        max_length=max_length,
        augment_data=augment_data
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Speed up GPU transfer
    )


def load_network_traffic_data(file_path: str) -> pd.DataFrame:
    """
    Load network traffic data from CSV file with basic validation
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the network traffic data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} samples from {file_path}")
        
        # Check for expected PPI columns (90 total: 3 directions × 30 features each)
        expected_ppi_cols = [f"PPI_{i}_{j}" for i in range(3) for j in range(30)]
        missing_cols = [col for col in expected_ppi_cols if col not in data.columns]
        
        if missing_cols:
            print(f"Warning: Missing PPI columns: {missing_cols}")
        
        # Check for APP label column
        if "APP" not in data.columns:
            print("Warning: APP column not found")
        else:
            print(f"Found {data['APP'].nunique()} unique APP labels")
        
        return data
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def split_data(
    data: pd.DataFrame, 
    train_ratio: float = 0.8,     # 80% for training
    val_ratio: float = 0.1,       # 10% for validation
    test_ratio: float = 0.1,      # 10% for testing 
    random_state: int = 42        # Seed for reproducible splits
) -> tuple:
    """
    Split data into train, validation, and test sets with stratification
    
    Args:
        data: DataFrame to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Ensure ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle the data for random split
    data_shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(data_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_data = data_shuffled[:n_train]
    val_data = data_shuffled[n_train:n_train + n_val]
    test_data = data_shuffled[n_train + n_val:]
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def analyze_data_distribution(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the statistical distribution of network traffic features
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary containing distribution statistics
    """
    analysis = {}
    
    # Analyze PPI features (90 total features across 3 packet directions)
    ppi_stats = {}
    for i in range(3):  # 3 packet directions
        for j in range(30):  # 30 features per direction
            col = f"PPI_{i}_{j}"
            if col in data.columns:
                ppi_stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'zeros': (data[col] == 0).sum(),           # Count zero values
                    'negatives': (data[col] < 0).sum()         # Count negative values
                }
    
    analysis['ppi_features'] = ppi_stats
    
    # Analyze APP labels distribution
    if "APP" in data.columns:
        app_counts = data["APP"].value_counts()
        analysis['app_labels'] = {
            'unique_count': len(app_counts),
            'distribution': app_counts.to_dict(),
            'most_common': app_counts.index[0] if len(app_counts) > 0 else None
        }
    
    # Overall dataset statistics
    analysis['overall'] = {
        'total_samples': len(data),
        'total_features': len([col for col in data.columns if col.startswith('PPI_')]),
        'missing_values': data.isnull().sum().sum()
    }
    
    return analysis