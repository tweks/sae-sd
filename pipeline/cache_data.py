# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "accelerate",
#     "clip",
#     "datasets",
#     "diffusers",
#     "loguru",
#     "numpy<2.0",
#     "open-clip-torch",
#     "python-dotenv",
#     "transformers",
# ]
#
# [tool.uv.sources]
# clip = { git = "https://github.com/openai/CLIP.git" }
# ///
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import dotenv_values

from data_utils.datasets import load as load_dataset, load_vocab, SUPPORTED_DATASETS, SUPPORTED_VOCAB
from data_utils.embedding_models import EmbeddingExtractor, embed_text, embed_image
from utils import set_seed, set_logger

USING_LOGURU, logger = set_logger(level="INFO")

def set_envs() -> tuple[str, str, str]:
    """
    Load environment variables from .env file and return paths.

    Returns:
        Tuple containing dataset path, model path, and embedding path
    """
    config = dotenv_values(".env")

    # Set HF token if present
    if "HF_TOKEN" in config:
        os.environ['HF_TOKEN'] = config['HF_TOKEN']

    # Return paths with defaults if not specified
    return (
        config.get("DATASET_PATH", "./datasets/"),
        config.get("MODEL_PATH", "./models/"),
        config.get("EMBEDDING_PATH", "./embeddings/")
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings from datasets using pre-trained models")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model to use")
    parser.add_argument("-b", "--batch-size", type=int, default=4096, help="Batch size to use")
    parser.add_argument("-t", "--train-split", action="store_true", help="Whether to use the train split")
    parser.add_argument("-v", "--vocab-size", type=int, default=-1, help="Vocabulary size to use")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def get_sample_embedding_shape(dataset: torch.utils.data.Dataset, extractor: EmbeddingExtractor) -> tuple[int, int]:
    """
    Get the shape of embeddings by processing a single sample.

    Args:
        dataset_iter: Iterator over the dataset
        extractor: Embedding extractor instance

    Returns:
        Shape of the embeddings

    Raises:
        RuntimeError: If neither image nor text can be encoded
    """
    dataset_iter = iter(dataset)
    img, target = next(dataset_iter)

    if isinstance(target, int) and hasattr(dataset, 'idx_to_class'):
        target = dataset.idx_to_class[target]

    # Choose embedding method based on available encoders and data
    if extractor.model.encode_image is not None and img is not None:
        features, _ = embed_image(extractor.model, extractor.preprocessor, img, extractor.device)
    elif extractor.model.encode_text is not None and target is not None:
        features, _ = embed_text(extractor.model, extractor.tokenizer, target, extractor.device)
    else:
        raise RuntimeError(
            f"Cannot extract embeddings: "
            f"Image encoder available: {extractor.model.encode_image is not None}, "
            f"Text encoder available: {extractor.model.encode_text is not None}, "
            f"Image available: {img is not None}, "
            f"Text available: {target is not None}"
        )

    return features.shape

def main(
    dataset_path: str,
    model_path: str,
    embedding_path: str,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    use_train: bool,
    vocab_size: int,
    seed: int = 42
) -> None:
    """
    Extract embeddings from a dataset using a specified model.

    Args:
        dataset_path: Path to dataset directory
        model_path: Path to model directory
        embedding_path: Path to save embeddings
        dataset_name: Name of dataset
        model_name: Name of model
        batch_size: Batch size for processing
        use_train: Whether to use training split
        vocab_size: Size of vocabulary (for vocab datasets)
        seed: Random seed
    """
    logger.info(f"Extracting embeddings from {dataset_name} using {model_name}")
    set_seed(seed)
    logger.info(f"Settings: train_split={use_train}, vocab_size={vocab_size}, batch_size={batch_size}, seed={seed}")

    extractor = EmbeddingExtractor(model_name, model_path)
    logger.info(f"Model loaded successfully: {model_name}")

    if dataset_name in SUPPORTED_DATASETS:
        logger.info(f"Loading {dataset_name} dataset")
        dataset = load_dataset(dataset_name, extractor.preprocessor, dataset_path, train=use_train, download=True)
        split_name = "train" if use_train else "test"
    elif dataset_name in SUPPORTED_VOCAB:
        logger.info(f"Loading {dataset_name} vocabulary")
        dataset = load_vocab(dataset_name, vocab_size)
        split_name = str(vocab_size)
    else:
        supported = ', '.join(SUPPORTED_DATASETS + SUPPORTED_VOCAB)
        raise ValueError(f"Dataset '{dataset_name}' not supported. Supported options: {supported}")

    logger.info(f"Extracting embeddings for {len(dataset)} samples")
    with torch.no_grad():
        sample_shape = get_sample_embedding_shape(dataset, extractor)
        logger.info(f"Sample embeddings shape: {sample_shape}")

        # Original dataset length (for iteration)
        dataset_len_original = len(dataset)
        # Total embedding size (for memmap)
        embedding_total_size = dataset_len_original
        seq_len = 1

        if len(sample_shape) > 2:
            seq_len = sample_shape[1]
            embedding_total_size *= seq_len  # Only for memmap allocation

        # Create memmap files for text embeddings
        if extractor.model.encode_text is not None:
            embedding_path_final_target = os.path.join(embedding_path, f"{dataset_name}_{model_name}_{split_name}_target_{embedding_total_size}_{sample_shape[-1]}.npy")
            memmap_target = np.memmap(embedding_path_final_target, dtype='float32', mode='w+', shape=(embedding_total_size, sample_shape[-1]))
            logger.info(f"Target data will be written to {embedding_path_final_target}")

            text_output_path = os.path.join(embedding_path, f"{dataset_name}_{model_name}_{split_name}_target_text_{dataset_len_original}.txt")
            logger.info(f"Text data will be written to {text_output_path}")
        else:
            text_output_path = None
            memmap_target = None
            logger.info("No target data to write")

        # Create memmap files for image embeddings
        if extractor.model.encode_image is not None and dataset_name not in SUPPORTED_VOCAB:
            embedding_path_final_img = os.path.join(embedding_path, f"{dataset_name}_{model_name}_{split_name}_img_{embedding_total_size}_{sample_shape[-1]}.npy")
            memmap_img = np.memmap(embedding_path_final_img, dtype='float32', mode='w+', shape=(embedding_total_size, sample_shape[-1]))
            logger.info(f"Image data will be written to {embedding_path_final_img}")
        else:
            memmap_img = None
            logger.info("No image data to write")

        # Process the dataset in chunks
        dataset_iter = iter(dataset)
        all_text = []
        processed_samples = 0

        # Process data in chunks
        for start_idx in tqdm(range(0, dataset_len_original, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, dataset_len_original)
            batch_size_actual = end_idx - start_idx

            # Collect batch data
            images, texts = [], []
            try:
                for _ in range(batch_size_actual):
                    try:
                        img_row, text_row = next(dataset_iter)

                        # Convert class index to text if needed
                        if isinstance(text_row, int) and hasattr(dataset, 'idx_to_class'):
                            text_row = dataset.idx_to_class[text_row]

                        images.append(img_row)
                        texts.append(text_row)
                        all_text.append(text_row)
                        processed_samples += 1
                    except StopIteration:
                        logger.warning(f"Dataset iteration stopped early at {processed_samples} out of {dataset_len_original} expected samples.")
                        break

                # If no images were collected, we're done
                if not images:
                    logger.info(f"Finished processing all available samples: {processed_samples}")
                    break

                # Process image embeddings if needed
                if memmap_img is not None:
                    with torch.amp.autocast('cuda'):  # Fixed deprecated warning
                        img_embeddings, _ = embed_image(
                            extractor.model,
                            extractor.preprocessor,
                            images,
                            extractor.device
                        )

                    # Handle 3D embeddings (sequence outputs)
                    if img_embeddings.dim() == 3:
                        # Get the middle dimension size (sequence length)
                        batch_seq_len = img_embeddings.size(1)
                        new_batch_size = img_embeddings.size(0)

                        # Calculate proper indices for the memmap
                        memmap_start = start_idx * seq_len
                        memmap_end = memmap_start + new_batch_size * batch_seq_len

                        # Reshape by flattening the batch and sequence dimensions
                        img_embeddings = img_embeddings.reshape(-1, img_embeddings.size(-1))

                        # Save to disk with corrected indices
                        memmap_img[memmap_start:memmap_end] = img_embeddings.detach().cpu().numpy().astype(np.float32)
                    else:
                        # For 2D embeddings, use the original indices
                        memmap_img[start_idx:end_idx] = img_embeddings.detach().cpu().numpy().astype(np.float32)

                    memmap_img.flush()

                # Process text embeddings if needed
                if memmap_target is not None:
                    with torch.amp.autocast('cuda'):  # Fixed deprecated warning
                        text_embeddings, _ = embed_text(
                            extractor.model,
                            extractor.tokenizer,
                            texts,
                            extractor.device
                        )

                    # Handle 3D embeddings (sequence outputs)
                    if text_embeddings.dim() == 3:
                        # Get the middle dimension size (sequence length)
                        batch_seq_len = text_embeddings.size(1)
                        new_batch_size = text_embeddings.size(0)

                        # Calculate proper indices for the memmap
                        memmap_start = start_idx * seq_len
                        memmap_end = memmap_start + new_batch_size * batch_seq_len

                        # Reshape by flattening the batch and sequence dimensions
                        text_embeddings = text_embeddings.reshape(-1, text_embeddings.size(-1))

                        # Save to disk with corrected indices
                        memmap_target[memmap_start:memmap_end] = text_embeddings.detach().cpu().numpy().astype(np.float32)
                    else:
                        # For 2D embeddings, use the original indices
                        memmap_target[start_idx:end_idx] = text_embeddings.detach().cpu().numpy().astype(np.float32)

                    memmap_target.flush()

            except Exception as e:
                logger.error(f"Error processing batch at index {start_idx}: {e}")
                continue  # Continue with the next batch

        if text_output_path is not None:
            with open(text_output_path, "w") as f:
                f.write("\n".join(all_text))

    logger.info(f"Successfully extracted embeddings for {processed_samples} samples")
    logger.info("Embeddings written to file\nAndiamo!")


if __name__ == "__main__":
    # Load environment variables and paths
    dataset_path, model_path, embedding_path = set_envs()

    # Parse command line arguments
    args = parse_args()

    if USING_LOGURU:
        main = logger.catch(main)

    # Run embedding extraction
    main(
        dataset_path,
        model_path,
        embedding_path,
        args.dataset,
        args.model,
        args.batch_size,
        args.train_split,
        args.vocab_size
    )