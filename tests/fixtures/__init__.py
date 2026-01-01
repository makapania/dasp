"""
Test fixtures and synthetic data generators for Spectral Predict.

This package provides reusable components for testing:
- Synthetic spectral data generators
- Helper utilities for test data creation
"""

from tests.fixtures.synthetic_data import (
    generate_baseline_data,
    generate_classification_spectra,
    generate_imbalanced_data,
    generate_outlier_data,
    generate_spectral_data,
)

__all__ = [
    "generate_spectral_data",
    "generate_outlier_data",
    "generate_imbalanced_data",
    "generate_baseline_data",
    "generate_classification_spectra",
]
