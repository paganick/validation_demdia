"""
Test Suite

Comprehensive testing infrastructure for the simulation pipeline.

Main Components:
- run_test_quick.py: Quick test (Llama-3.1-8B only, ~2-5 min)
- run_test_full.py: Full test (all models, ~10-20 min)
- verify_reproducibility.py: Reproducibility verification
- test_seed_utils.py: Random seed fixing utilities
- save_golden_outputs.py: Save reference outputs for regression testing
- compare_with_golden.py: Compare outputs against golden reference

See TEST_SUITE_README.md for detailed documentation.
"""

__version__ = "1.0.0"

from .test_seed_utils import set_global_seed, get_test_seed

__all__ = ["set_global_seed", "get_test_seed"]
