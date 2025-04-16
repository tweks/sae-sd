import os
import torch
import json

from .utils import set_logger

_, logger = set_logger(level="INFO")


def save_dataset_efficient(dataset, dataset_path: str) -> None:
    """
    Efficiently save dataset state by separating metadata from the large memory-mapped arrays.
    
    Args:
        dataset: The dataset object (SAEDataset, SDSAEDataset, or MultiSAEDataset)
        dataset_path: Base path to save the dataset (without extension)
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(dataset_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract dataset class name
    class_name = dataset.__class__.__name__
    
    # Common attributes to save for all dataset types
    metadata = {
        "class_name": class_name,
        "dtype": str(dataset.dtype),
        "vector_size": dataset.vector_size,
        "len": dataset.len,
        "mean_center": dataset.mean_center,
        "normalize": dataset.normalize,
        "target_norm": dataset.target_norm if hasattr(dataset, "target_norm") else None,
        "scaling_factor": float(dataset.scaling_factor) if hasattr(dataset, "scaling_factor") else 1.0,
    }
    
    # Save dataset-specific attributes
    if class_name == "SAEDataset":
        # For basic SAEDataset, save the data_path
        data_path = dataset.data.filename
        metadata["data_path"] = data_path
        metadata["mean"] = dataset.mean.cpu().numpy().tolist() if hasattr(dataset, "mean") else None
        
    elif class_name == "SDSAEDataset":
        # For SDSAEDataset, save additional attributes
        data_path = dataset.data.filename
        metadata["data_path"] = data_path
        metadata["seq_len"] = dataset.seq_len
        metadata["use_lens"] = dataset.use_lens
        metadata["seq_id"] = dataset.seq_id
        metadata["mean"] = dataset.mean.cpu().numpy().tolist() if hasattr(dataset, "mean") else None
        
        # Save lenses separately if they exist
        if dataset.use_lens and dataset.lenses:
            lenses_dir = os.path.join(save_dir, "lenses")
            if not os.path.exists(lenses_dir):
                os.makedirs(lenses_dir)
            
            lens_paths = {}
            for key, lens in dataset.lenses.items():
                lens_path = os.path.join(lenses_dir, f"lens_{key}.pt")
                torch.save(lens.state_dict(), lens_path)
                lens_paths[str(key)] = os.path.relpath(lens_path, save_dir)
            
            metadata["lens_paths"] = lens_paths
            metadata["lenses_loss"] = {str(k): float(v) for k, v in dataset.lenses_loss.items()} if dataset.lenses_loss else None
            
    elif class_name == "MultiSAEDataset":
        # For MultiSAEDataset, save paths to both modalities
        data_path_first = dataset.data_first.filename
        data_path_second = dataset.data_second.filename
        metadata["data_path_first"] = data_path_first
        metadata["data_path_second"] = data_path_second
        metadata["use_first"] = dataset.use_first
        metadata["use_lens"] = dataset.use_lens
        metadata["mean_first"] = dataset.mean_first.cpu().numpy().tolist() if hasattr(dataset, "mean_first") else None
        metadata["mean_second"] = dataset.mean_second.cpu().numpy().tolist() if hasattr(dataset, "mean_second") else None
        metadata["scaling_factor_first"] = float(dataset.scaling_factor_first) if hasattr(dataset, "scaling_factor_first") else 1.0
        metadata["scaling_factor_second"] = float(dataset.scaling_factor_second) if hasattr(dataset, "scaling_factor_second") else 1.0
        
        # Save lens if it exists
        if dataset.use_lens and dataset.lenses:
            lens_path = os.path.join(save_dir, "multi_lens.pt")
            torch.save(dataset.lenses.state_dict(), lens_path)
            metadata["lens_path"] = os.path.relpath(lens_path, save_dir)
            metadata["lenses_loss"] = float(dataset.lenses_loss) if dataset.lenses_loss else 0.0
    
    # Save metadata to JSON file
    metadata_path = f"{dataset_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset metadata saved to {metadata_path}")


def load_dataset_efficient(dataset_path: str, device: torch.device | None = None):
    """
    Efficiently load a dataset from metadata and reconstruct it.
    
    Args:
        dataset_path: Base path where the dataset was saved (without extension)
        device: Optional device to load tensors to
        
    Returns:
        The reconstructed dataset object
    """
    # Load metadata
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset metadata not found at {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        metadata = json.load(f)
    
    # Import necessary classes
    from SAE.datasets import SAEDataset, SDSAEDataset, MultiSAEDataset, TunedLens
    
    # Convert dtype string back to torch dtype
    dtype_str = metadata["dtype"]
    if "float32" in dtype_str:
        dtype = torch.float32
    elif "float64" in dtype_str:
        dtype = torch.float64
    elif "float16" in dtype_str:
        dtype = torch.float16
    else:
        dtype = torch.float32  # Default
    
    # Reconstruct dataset based on class name
    class_name = metadata["class_name"]
    
    if class_name == "SAEDataset":
        # Reconstruct SAEDataset
        dataset = SAEDataset(
            data_path=metadata["data_path"],
            dtype=dtype,
            mean_center=False,
            normalize=False,
            target_norm=0.0,
        )
        dataset.mean_center = metadata["mean_center"]
        dataset.normalize = metadata["normalize"]
        dataset.target_norm = metadata["target_norm"]
        if metadata["mean"] is not None:
            dataset.mean = torch.tensor(metadata["mean"], dtype=dtype)
        if "scaling_factor" in metadata:
            dataset.scaling_factor = float(metadata["scaling_factor"])
        
    elif class_name == "SDSAEDataset":
        # Reconstruct SDSAEDataset
        dataset = SDSAEDataset(
            data_path=metadata["data_path"],
            dtype=dtype,
            mean_center=False,
            normalize=False,
            target_norm=False,
            seq_len=metadata["seq_len"],
            use_lens=False,
            seq_id=metadata["seq_id"],
            device=device
        )
        dataset.mean_center = metadata["mean_center"]
        dataset.normalize = metadata["normalize"]
        dataset.target_norm = metadata["target_norm"]
        dataset.use_lens = metadata["use_lens"]
        if metadata["mean"] is not None:
            dataset.mean = torch.tensor(metadata["mean"], dtype=dtype)
        if "scaling_factor" in metadata:
            dataset.scaling_factor = float(metadata["scaling_factor"])
        
        # Restore lenses if they exist
        if metadata["use_lens"] and "lens_paths" in metadata:
            base_dir = os.path.dirname(dataset_path)
            dataset.lenses = {}
            dataset.lenses_loss = {}
            
            for key_str, rel_path in metadata["lens_paths"].items():
                key = int(key_str)
                lens_path = os.path.join(base_dir, rel_path)
                lens = TunedLens(dataset.vector_size)
                lens.load_state_dict(torch.load(lens_path, map_location=device))
                lens.eval()
                dataset.lenses[key] = lens
                
            if "lenses_loss" in metadata and metadata["lenses_loss"]:
                for key_str, loss in metadata["lenses_loss"].items():
                    dataset.lenses_loss[int(key_str)] = loss
                    
    elif class_name == "MultiSAEDataset":
        # Reconstruct MultiSAEDataset
        dataset = MultiSAEDataset(
            data_path_first=metadata["data_path_first"],
            data_path_second=metadata["data_path_second"],
            dtype=dtype,
            mean_center=False,
            normalize=False,
            target_norm=False,
            use_first=metadata["use_first"],
            use_lens=False,
            device=device
        )
        dataset.mean_center = metadata["mean_center"]
        dataset.normalize = metadata["normalize"]
        dataset.target_norm = metadata["target_norm"]
        dataset.use_lens = metadata["use_lens"]
        if metadata["mean_first"] is not None:
            dataset.mean_first = torch.tensor(metadata["mean_first"], dtype=dtype)
        if metadata["mean_second"] is not None:
            dataset.mean_second = torch.tensor(metadata["mean_second"], dtype=dtype)
        dataset.scaling_factor_first = metadata["scaling_factor_first"]
        dataset.scaling_factor_second = metadata["scaling_factor_second"]
        
        # Restore lens if it exists
        if metadata["use_lens"] and "lens_path" in metadata:
            base_dir = os.path.dirname(dataset_path)
            lens_path = os.path.join(base_dir, metadata["lens_path"])
            dataset.lenses = TunedLens(dataset.vector_size)
            dataset.lenses.load_state_dict(torch.load(lens_path, map_location=device))
            dataset.lenses.eval()
            dataset.lenses_loss = metadata["lenses_loss"]
    
    else:
        raise ValueError(f"Unknown dataset class: {class_name}")
    
    logger.info(f"Dataset loaded from {dataset_path}")
    return dataset