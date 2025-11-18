"""Utilities for fixing random seeds across the simulation pipeline.

This module provides functions to ensure reproducible results for testing
and debugging by fixing all sources of randomness in the simulation.
"""

import random
import numpy as np
import torch
import os


def set_global_seed(seed: int = 42):
    """Fix all random seeds for reproducibility across the entire pipeline.

    Sets seeds for all major sources of randomness including Python's random
    module, NumPy, and PyTorch (both CPU and CUDA). Also configures PyTorch
    to use deterministic algorithms and sets Python hash seed.

    Args:
        seed: The random seed to use for all random number generators
            (default: 42)

    Side Effects:
        - Sets random.seed()
        - Sets np.random.seed()
        - Sets torch.manual_seed() and torch.cuda.manual_seed_all()
        - Enables PyTorch deterministic mode (cudnn.deterministic=True)
        - Disables PyTorch benchmark mode (cudnn.benchmark=False)
        - Sets PYTHONHASHSEED environment variable
        - Prints confirmation message to stdout

    Examples:
        >>> set_global_seed(42)
        Global seed set to 42 for reproducibility
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
    """Get the standard test seed used across all tests.

    Returns:
        The fixed seed value (42) used for all reproducibility tests.

    Examples:
        >>> get_test_seed()
        42
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
