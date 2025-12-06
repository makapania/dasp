"""
Generate synthetic test fixtures for imbalance and outlier detection tests.

This script creates:
- imbalanced_data.npz: Datasets with various imbalance scenarios
- outlier_data.npz: Spectral data with known outliers
"""

import numpy as np
from pathlib import Path


def generate_imbalanced_data():
    """Generate synthetic datasets with various imbalance scenarios."""
    np.random.seed(42)

    # Scenario 1: Balanced binary classification
    y_balanced = np.array([0] * 50 + [1] * 50)

    # Scenario 2: Moderate imbalance (80/20)
    y_moderate = np.array([0] * 80 + [1] * 20)

    # Scenario 3: Severe imbalance (90/10)
    y_severe = np.array([0] * 90 + [1] * 10)

    # Scenario 4: Extreme imbalance (99/1)
    y_extreme = np.array([0] * 99 + [1] * 1)

    # Scenario 5: Balanced multi-class
    y_multiclass_balanced = np.array([0] * 30 + [1] * 30 + [2] * 30)

    # Scenario 6: Imbalanced multi-class (60/30/10)
    y_multiclass_imbalanced = np.array([0] * 60 + [1] * 30 + [2] * 10)

    # Scenario 7: Balanced regression (uniform distribution)
    y_reg_balanced = np.linspace(0, 10, 100)

    # Scenario 8: Imbalanced regression (many zeros, few high values)
    y_reg_imbalanced = np.concatenate([
        np.zeros(70),
        np.random.uniform(0, 2, 15),
        np.random.uniform(8, 10, 15)
    ])

    # Scenario 9: Extreme regression imbalance (exponential distribution)
    y_reg_extreme = np.random.exponential(scale=2.0, size=100)

    fixtures_dir = Path(__file__).parent
    np.savez(
        fixtures_dir / 'imbalanced_data.npz',
        y_balanced=y_balanced,
        y_moderate=y_moderate,
        y_severe=y_severe,
        y_extreme=y_extreme,
        y_multiclass_balanced=y_multiclass_balanced,
        y_multiclass_imbalanced=y_multiclass_imbalanced,
        y_reg_balanced=y_reg_balanced,
        y_reg_imbalanced=y_reg_imbalanced,
        y_reg_extreme=y_reg_extreme
    )

    print("Generated imbalanced_data.npz")


def generate_outlier_data():
    """Generate synthetic spectral data with known outliers."""
    np.random.seed(42)

    n_samples = 100
    n_wavelengths = 200
    wavelengths = np.linspace(400, 2400, n_wavelengths)

    # Normal spectra (samples 0-94)
    X_normal = np.zeros((95, n_wavelengths))
    for i in range(95):
        # Random Gaussian peak
        center = np.random.uniform(800, 1800)
        width = np.random.uniform(100, 300)
        amplitude = np.random.uniform(0.5, 1.5)
        X_normal[i] = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
        # Add noise
        X_normal[i] += np.random.normal(0, 0.01, n_wavelengths)

    # Outlier spectra (samples 95-99)
    X_outliers = np.zeros((5, n_wavelengths))

    # Outlier 1: Very high values
    X_outliers[0] = np.ones(n_wavelengths) * 10.0

    # Outlier 2: Very low values
    X_outliers[1] = np.ones(n_wavelengths) * -5.0

    # Outlier 3: Spike
    X_outliers[2] = np.zeros(n_wavelengths)
    X_outliers[2][100] = 20.0

    # Outlier 4: Different shape (linear)
    X_outliers[4] = np.linspace(0, 5, n_wavelengths)

    # Outlier 5: Random noise
    X_outliers[4] = np.random.normal(0, 2.0, n_wavelengths)

    # Combine
    X_spectra = np.vstack([X_normal, X_outliers])

    # Reference values
    y_normal = np.random.normal(loc=10.0, scale=2.0, size=95)
    y_outliers = np.array([50.0, -20.0, 10.0, 10.0, 10.0])  # First two are Y-outliers
    y_reference = np.concatenate([y_normal, y_outliers])

    # Sample IDs
    sample_ids = [f'Sample_{i:03d}' for i in range(100)]

    # Known outlier indices
    known_outlier_indices = np.array([95, 96, 97, 98, 99])

    fixtures_dir = Path(__file__).parent
    np.savez(
        fixtures_dir / 'outlier_data.npz',
        X_spectra=X_spectra,
        y_reference=y_reference,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        known_outlier_indices=known_outlier_indices
    )

    print("Generated outlier_data.npz")


def generate_all_fixtures():
    """Generate all test fixtures."""
    generate_imbalanced_data()
    generate_outlier_data()
    print("\nAll fixtures generated successfully!")


if __name__ == '__main__':
    generate_all_fixtures()
