"""
Utilities for fixing random seeds across the simulation pipeline.
Ensures reproducible results for testing and debugging.
"""

import random
import numpy as np
import torch
import os


def set_global_seed(seed: int = 42):
    """
    Fix all random seeds for reproducibility.

    Args:
        seed: The random seed to use (default: 42)

    This sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variables for hash randomization
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed (for dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ Global seed set to {seed} for reproducibility")


def get_test_seed():
    """
    Returns the standard test seed.

    Returns:
        int: The fixed seed value (42) used for all tests
    """
    return 42


if __name__ == "__main__":
    # Test the seed setting
    set_global_seed(42)

    # Verify reproducibility
    print("\nTesting reproducibility:")
    print(f"Random float: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")

    # Reset and test again
    set_global_seed(42)
    print("\nAfter reset (should be identical):")
    print(f"Random float: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
