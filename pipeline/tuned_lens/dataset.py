import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class TunedLensDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 77, embedding_dim: int = 768):
        """
        Dataset for TunedLens training.
        
        Args:
            data_path: Path to the numpy file containing the embeddings
            seq_len: Length of the sequence (default: 77)
            embedding_dim: Dimension of the embeddings (default: 768)
        """
        self.data = np.load(data_path)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # Validate data shape
        if len(self.data.shape) != 2 or self.data.shape[1] != seq_len * embedding_dim:
            raise ValueError(f"Expected data shape (N, {seq_len * embedding_dim}), got {self.data.shape}")
            
        # Reshape data to (N, seq_len, embedding_dim)
        self.data = self.data.reshape(-1, seq_len, embedding_dim)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (input_vector, target_vector)
            - input_vector: Vector at position 0
            - target_vector: Vector at position i (where i is randomly chosen from 1..76)
        """
        sequence = self.data[idx]
        
        # Randomly select a target position (1 to 76)
        target_pos = np.random.randint(1, self.seq_len)
        
        # Get input and target vectors
        input_vector = torch.from_numpy(sequence[0]).float()
        target_vector = torch.from_numpy(sequence[target_pos]).float()
        
        return input_vector, target_vector 