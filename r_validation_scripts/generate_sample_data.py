"""
Generate sample spectral data for R validation testing.

This script creates synthetic NIR spectral datasets with known properties
for validating that Python and R implementations produce equivalent results.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_nir_regression_data(
    n_samples=150,
    n_wavelengths=800,
    n_informative=50,
    noise_level=0.1,
    random_seed=42
):
    """
    Generate synthetic NIR spectral data for regression.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_wavelengths : int
        Number of wavelengths (features)
    n_informative : int
        Number of informative wavelengths
    noise_level : float
        Standard deviation of Gaussian noise
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    X : ndarray
        Spectral data (n_samples, n_wavelengths)
    y : ndarray
        Target values
    wavelengths : ndarray
        Wavelength values (for metadata)
    """
    np.random.seed(random_seed)

    # Generate wavelengths (typical NIR range: 1000-2500 nm)
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Generate informative features with structured patterns
    X_informative = np.zeros((n_samples, n_informative))

    # Create basis functions (different spectral patterns)
    t = np.linspace(0, 2 * np.pi, n_informative)
    basis1 = np.sin(t)
    basis2 = np.cos(t)
    basis3 = np.sin(2 * t)
    basis4 = np.exp(-((t - np.pi)**2) / 2)  # Gaussian

    # Generate samples with different combinations of basis functions
    coefficients = np.random.randn(n_samples, 4)
    for i in range(n_samples):
        X_informative[i] = (
            coefficients[i, 0] * basis1 +
            coefficients[i, 1] * basis2 +
            coefficients[i, 2] * basis3 +
            coefficients[i, 3] * basis4
        )

    # Add non-informative features
    X_noise = np.random.randn(n_samples, n_wavelengths - n_informative) * 0.5

    # Combine informative and non-informative features
    X = np.hstack([X_informative, X_noise])

    # Generate target variable as linear combination of coefficients + noise
    true_weights = np.array([2.5, -1.8, 1.2, -0.9])
    y = coefficients @ true_weights + np.random.randn(n_samples) * noise_level

    # Normalize y to have reasonable scale (e.g., 0-20 for percentage)
    y = (y - y.min()) / (y.max() - y.min()) * 20

    return X, y, wavelengths


def generate_nir_classification_data(
    n_samples=150,
    n_wavelengths=800,
    n_classes=3,
    n_informative=50,
    random_seed=42
):
    """
    Generate synthetic NIR spectral data for classification.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_wavelengths : int
        Number of wavelengths (features)
    n_classes : int
        Number of classes
    n_informative : int
        Number of informative wavelengths
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    X : ndarray
        Spectral data (n_samples, n_wavelengths)
    y : ndarray
        Class labels (0, 1, 2, ...)
    wavelengths : ndarray
        Wavelength values (for metadata)
    """
    np.random.seed(random_seed)

    # Generate wavelengths
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Generate samples for each class
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []

    for class_idx in range(n_classes):
        # Each class has different spectral signature
        n_class_samples = samples_per_class + (1 if class_idx < (n_samples % n_classes) else 0)

        # Create class-specific spectral pattern
        t = np.linspace(0, 2 * np.pi, n_informative)

        # Different patterns for different classes
        if class_idx == 0:
            pattern = np.sin(t + 0)
        elif class_idx == 1:
            pattern = np.sin(t + np.pi/2)
        else:
            pattern = np.sin(t + np.pi)

        # Generate samples with class-specific pattern + noise
        X_class_informative = np.zeros((n_class_samples, n_informative))
        for i in range(n_class_samples):
            # Add individual variation
            X_class_informative[i] = pattern + np.random.randn(n_informative) * 0.3

        # Add non-informative features
        X_class_noise = np.random.randn(n_class_samples, n_wavelengths - n_informative) * 0.5

        # Combine
        X_class = np.hstack([X_class_informative, X_class_noise])

        X_list.append(X_class)
        y_list.append(np.full(n_class_samples, class_idx))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y, wavelengths


def save_data(X, y, wavelengths, output_dir, prefix):
    """
    Save data in formats compatible with both Python and R.

    Parameters
    ----------
    X : ndarray
        Spectral data
    y : ndarray
        Target values
    wavelengths : ndarray
        Wavelength values
    output_dir : Path or str
        Output directory
    prefix : str
        File prefix (e.g., 'nir_regression' or 'nir_classification')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV (compatible with both Python and R)
    # Format: wavelength columns + target column
    columns = [f"nm_{int(w)}" for w in wavelengths] + ['target']
    df = pd.DataFrame(
        np.hstack([X, y.reshape(-1, 1)]),
        columns=columns
    )

    csv_path = output_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save wavelengths separately
    wavelengths_path = output_dir / f"{prefix}_wavelengths.csv"
    pd.DataFrame({'wavelength': wavelengths}).to_csv(wavelengths_path, index=False)
    print(f"Saved: {wavelengths_path}")

    # Save metadata
    metadata = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'target_min': y.min(),
        'target_max': y.max(),
        'target_mean': y.mean(),
        'target_std': y.std()
    }

    metadata_path = output_dir / f"{prefix}_metadata.txt"
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved: {metadata_path}")


def main():
    """Generate all sample datasets."""
    base_dir = Path(__file__).parent.parent / 'data'

    print("="*80)
    print("GENERATING SAMPLE SPECTRAL DATA FOR R VALIDATION")
    print("="*80)

    # Generate regression datasets (small, medium, large)
    print("\n1. Small regression dataset (100 samples, 500 wavelengths)")
    X_small, y_small, wavelengths_small = generate_nir_regression_data(
        n_samples=100, n_wavelengths=500, random_seed=42
    )
    save_data(X_small, y_small, wavelengths_small, base_dir, 'sample_nir_regression_small')

    print("\n2. Medium regression dataset (150 samples, 800 wavelengths)")
    X_medium, y_medium, wavelengths_medium = generate_nir_regression_data(
        n_samples=150, n_wavelengths=800, random_seed=42
    )
    save_data(X_medium, y_medium, wavelengths_medium, base_dir, 'sample_nir_regression_medium')

    print("\n3. Large regression dataset (200 samples, 1000 wavelengths)")
    X_large, y_large, wavelengths_large = generate_nir_regression_data(
        n_samples=200, n_wavelengths=1000, random_seed=42
    )
    save_data(X_large, y_large, wavelengths_large, base_dir, 'sample_nir_regression_large')

    # Generate classification dataset
    print("\n4. Classification dataset (150 samples, 800 wavelengths, 3 classes)")
    X_class, y_class, wavelengths_class = generate_nir_classification_data(
        n_samples=150, n_wavelengths=800, n_classes=3, random_seed=42
    )
    save_data(X_class, y_class, wavelengths_class, base_dir, 'sample_nir_classification')

    print("\n" + "="*80)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nData saved to: {base_dir}")
    print("\nGenerated files:")
    print("  - sample_nir_regression_small.csv (100×500)")
    print("  - sample_nir_regression_medium.csv (150×800)")
    print("  - sample_nir_regression_large.csv (200×1000)")
    print("  - sample_nir_classification.csv (150×800, 3 classes)")
    print("\nThese datasets use fixed random seed (42) for reproducibility.")
    print("Both Python and R should produce identical results when using the same seed.")


if __name__ == '__main__':
    main()
