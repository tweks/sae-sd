import os
import torch
import argparse
from omegaconf import OmegaConf

from utils import set_logger, set_seed, get_dtype, set_envs
from SAE.datasets import (
    SAEDataset,
    SDSAEDataset,
    MultiSAEDataset,
)
from SAE.dataset_serialization import save_dataset_efficient, load_dataset_efficient

USING_LOGURU, logger = set_logger(level="INFO")


def main(args):
    """
    Main function to prepare datasets.
    """
    # Load omegaConf config
    config = OmegaConf.load(args.config_path)
    logger.info(f"Configuration loaded from {args.config_path}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Seed set to {config.seed}")
    
    # Set up data type
    dtype = get_dtype(config.dtype)
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    logger.info(f"Data type set to {dtype}")
    
    # Prepare datasets
    logger.info(f"Preparing datasets of type: {args.dataset_type}")
    if args.dataset_type == "msae":
        dataset_name_train_ds = config.dataset_path.split("/")[-1].split(".")[0]
        dataset_name_train_ds = dataset_name_train_ds.split("_")
        dataset_name_train_ds = "_".join(dataset_name_train_ds[:-2])
        train_ds = SAEDataset(
            config.dataset_path, 
            dtype=dtype, 
            mean_center=config.mean_center, 
            normalize=config.normalize, 
            target_norm=config.target_norm
        )
        
        dataset_name_eval_ds = config.dataset_path.split("/")[-1].split(".")[0]
        dataset_name_eval_ds = dataset_name_eval_ds.split("_")
        dataset_name_eval_ds = "_".join(dataset_name_eval_ds[:-2])        
        eval_ds = SAEDataset(
            config.eval_dataset_path, 
            dtype=dtype, 
            mean_center=config.mean_center, 
            normalize=config.normalize, 
            target_norm=config.target_norm
        )
    elif args.dataset_type == "sds":
        dataset_name_train_ds = config.dataset_path.split("/")[-1].split(".")[0]
        dataset_name_train_ds = dataset_name_train_ds.split("_")
        dataset_name_train_ds = "_".join(dataset_name_train_ds[:-2])
        train_ds = SDSAEDataset(
            config.dataset_path,
            dtype=dtype,
            mean_center=config.mean_center,
            normalize=config.normalize,
            target_norm=config.target_norm,
            seq_len=config.seq_len,
            use_lens=config.use_lens,
            seq_id=config.seq_id,
            hyper_search=args.hypersearch,
        )
        eval_ds = None
    elif args.dataset_type == "multi":
        dataset_name_train_ds = config.dataset_path_first.split("/")[-1].split(".")[0]
        dataset_name_train_ds = dataset_name_train_ds.split("_")
        dataset_name_train_ds = "_".join(dataset_name_train_ds[:-2])
        train_ds = MultiSAEDataset(
            config.dataset_path_first,
            config.dataset_path_second,
            dtype=dtype,
            mean_center=config.mean_center,
            normalize=config.normalize,
            target_norm=config.target_norm,
            use_first=config.use_first,
            use_lens=config.use_lens,
            hyper_search=args.hypersearch,
        )
        
        dataset_name_eval_ds = config.dataset_eval_path_first.split("/")[-1].split(".")[0]
        dataset_name_eval_ds = dataset_name_eval_ds.split("_")
        dataset_name_eval_ds = "_".join(dataset_name_eval_ds[:-2])
        eval_ds = MultiSAEDataset(
            config.dataset_eval_path_first,
            config.dataset_eval_path_second,
            dtype=dtype,
            mean_center=config.mean_center,
            normalize=config.normalize,
            target_norm=config.target_norm,
            use_first=config.use_first,
            use_lens=config.use_lens,
            hyper_search=args.hypersearch,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    logger.info(f"Dataset prepared with {len(train_ds)} samples")
    if eval_ds is not None:
        logger.info(f"Eval dataset prepared with {len(eval_ds)} samples")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if train_ds is not None:
        logger.info(f"Train dataset with {len(train_ds)} samples and samples are the size of {train_ds.vector_size}")
        save_path = os.path.join(args.save_path, f"{dataset_name_train_ds}_{args.dataset_type}_train_dataset")
        logger.info(f"Saving train dataset to {save_path}")
        save_dataset_efficient(train_ds, save_path)
        logger.info("Train dataset saved successfully")
    if eval_ds is not None:
        logger.info(f"Eval dataset with {len(eval_ds)} samples and samples are the size of {eval_ds.vector_size}")
        save_path = os.path.join(args.save_path, f"{dataset_name_eval_ds}_{args.dataset_type}_eval_dataset")
        logger.info(f"Saving eval dataset to {save_path}")
        save_dataset_efficient(eval_ds, save_path)
        logger.info("Eval dataset saved successfully")
    
    logger.info("Dataset preparation completed successfully")


def load_dataset(dataset_path: str, device=None):
    """
    Efficiently load a dataset using the new method.
    
    Args:
        dataset_path: Path to the dataset (without extension)
        device: Optional device to load tensors to
        
    Returns:
        The loaded dataset
    """
    return load_dataset_efficient(dataset_path, device)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare datasets for SAE.")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True, choices=["msae", "sds", "multi"], help="Type of dataset to prepare."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the prepared datasets."
    )
    parser.add_argument(
        "--hypersearch", action="store_true", help="Use hypersearch mode."
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_envs()
    args = parse_args()

    if USING_LOGURU:
        main = logger.catch(main)

    main(args)