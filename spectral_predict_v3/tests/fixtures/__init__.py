"""Test fixtures for spectral_predict_v3."""

import numpy as np
from pathlib import Path


def ensure_fixtures_exist():
    """Generate test fixtures if they don't exist."""
    fixtures_dir = Path(__file__).parent

    # Generate imbalanced_data.npz
    imbalanced_path = fixtures_dir / 'imbalanced_data.npz'
    if not imbalanced_path.exists():
        np.random.seed(42)

        y_balanced = np.array([0] * 50 + [1] * 50)
        y_moderate = np.array([0] * 80 + [1] * 20)
        y_severe = np.array([0] * 90 + [1] * 10)
        y_extreme = np.array([0] * 99 + [1] * 1)
        y_multiclass_balanced = np.array([0] * 30 + [1] * 30 + [2] * 30)
        y_multiclass_imbalanced = np.array([0] * 60 + [1] * 30 + [2] * 10)
        y_reg_balanced = np.linspace(0, 10, 100)
        y_reg_imbalanced = np.concatenate([
            np.zeros(70),
            np.random.uniform(0, 2, 15),
            np.random.uniform(8, 10, 15)
        ])
        y_reg_extreme = np.random.exponential(scale=2.0, size=100)

        np.savez(
            imbalanced_path,
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

    # Generate outlier_data.npz
    outlier_path = fixtures_dir / 'outlier_data.npz'
    if not outlier_path.exists():
        np.random.seed(42)

        n_samples = 100
        n_wavelengths = 200
        wavelengths = np.linspace(400, 2400, n_wavelengths)

        # Normal spectra
        X_normal = np.zeros((95, n_wavelengths))
        for i in range(95):
            center = np.random.uniform(800, 1800)
            width = np.random.uniform(100, 300)
            amplitude = np.random.uniform(0.5, 1.5)
            X_normal[i] = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
            X_normal[i] += np.random.normal(0, 0.01, n_wavelengths)

        # Outlier spectra
        X_outliers = np.zeros((5, n_wavelengths))
        X_outliers[0] = np.ones(n_wavelengths) * 10.0
        X_outliers[1] = np.ones(n_wavelengths) * -5.0
        X_outliers[2] = np.zeros(n_wavelengths)
        X_outliers[2][100] = 20.0
        X_outliers[3] = np.linspace(0, 5, n_wavelengths)
        X_outliers[4] = np.random.normal(0, 2.0, n_wavelengths)

        X_spectra = np.vstack([X_normal, X_outliers])

        y_normal = np.random.normal(loc=10.0, scale=2.0, size=95)
        y_outliers = np.array([50.0, -20.0, 10.0, 10.0, 10.0])
        y_reference = np.concatenate([y_normal, y_outliers])

        sample_ids = [f'Sample_{i:03d}' for i in range(100)]
        known_outlier_indices = np.array([95, 96, 97, 98, 99])

        np.savez(
            outlier_path,
            X_spectra=X_spectra,
            y_reference=y_reference,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            known_outlier_indices=known_outlier_indices
        )


# Ensure fixtures exist on import
ensure_fixtures_exist()
