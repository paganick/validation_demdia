"""
Validation Package

Tools for validating and analyzing simulation outputs.

Main Components:
- utils.py: Validator class for BERT and Empath validation
- validate_text.py: Text validation pipeline
- features_analysis.py: Feature computation and evaluation
- feature_utils.py: Feature extraction utilities
- LLM_judge.py: LLM-based judging and scoring
- build_validation_data.py: Validation dataset preparation
- post_process.py: Post-processing and aggregation
- compute_cosine_similarities.py: Similarity metrics
"""

__version__ = "1.0.0"

from .utils import Validator

__all__ = ["Validator"]
