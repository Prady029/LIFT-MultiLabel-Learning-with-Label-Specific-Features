"""
LIFT: Learning with Label-Specific Features for Multi-Label Classification

This package implements the LIFT algorithm, which constructs label-specific features
for multi-label classification by clustering positive and negative samples separately
for each label and transforming the feature space based on distances to cluster centroids.
"""

from .transformer import LIFTTransformer
from .classifier import LIFTClassifier

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "LIFTTransformer",
    "LIFTClassifier",
]
