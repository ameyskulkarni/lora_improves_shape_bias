"""Utility functions for reproducibility."""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # PyTorch backends
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # This makes training slower but fully reproducible
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Faster but may have small variations

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def get_generator(seed: int):
    """Get torch generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g