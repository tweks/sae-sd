import os
import random
import socket
import sys
import numpy as np
import torch
from dotenv import dotenv_values

try:
    LOGURU_AVAILABLE = True
    from loguru import logger
except ImportError:
    LOGURU_AVAILABLE = False
    import logging


def set_envs() -> dict[str, str]:
    """
    Load environment variables from .env file.
    
    Loads and sets environment variables needed for training, including
    Hydra output path, embeddings path, and WandB API key.
    
    Returns:
        Dictionary of loaded environment variables
    """
    config = dotenv_values(".env")
    
    # Set environment variables if they exist in config
    if "HYDRA_OUTPUT" in config:
        os.environ['HYDRA_OUTPUT'] = config['HYDRA_OUTPUT']
    if "EMBEDDING_PATH" in config:
        os.environ['EMBEDDING_PATH'] = config['EMBEDDING_PATH']
    if "WANDB_API_KEY" in config:
        os.environ['WANDB_API_KEY'] = config['WANDB_API_KEY']
    else:
        global USE_WANDB
        USE_WANDB = False
        
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = "cpu"
    # return torch.device(device)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "int8":
        return torch.int8
    elif dtype == "int16":
        return torch.int16
    elif dtype == "int32":
        return torch.int32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def get_hostname_ip() -> tuple[str, str]:
    return socket.gethostname(), socket.gethostbyname(socket.gethostname())


def set_logger(level: str = "INFO", log_file: str | None = None, use_hostname: bool = True) -> tuple[bool, logger]:
    if not LOGURU_AVAILABLE:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | Line %(lineno)d (%(filename)s): %(message)s",
            level=level,
        )
        logger_default = logging.getLogger()
        logger_default.info("Loguru is not available. Using standard logging.")
        return False, logger_default

    logger.remove()
    logger.configure(extra={"ip": "", "user": ""})
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <cyan>{extra[ip]}</cyan> <cyan>{extra[user]}</cyan> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <level>{message}</level>",
        level=level,
        backtrace=True,
        diagnose=True
    )
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[ip]} {extra[user]} | {level: <8} | Line {line: >4} ({file}): {message}",
        )  # , serialize=True, compression="zip")
    
    if use_hostname:
        hostname, ip = get_hostname_ip()
        context_logger = logger.bind(user=hostname, ip=ip)
    else:
        context_logger = logger
    return True, context_logger
