"""
Synthetic data generators for spectral prediction testing.

This module provides deterministic synthetic data generation functions for testing
spectral prediction models. All functions accept a seed parameter for reproducibility.

Functions return properly typed pandas DataFrames and Series with appropriate
wavelength column names for spectral data.
"""

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def generate_spectral_data(
    n_samples: int,
    n_wavelengths: int,
    n_informative: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic spectral data with target values.

    Creates spectral data where y is a linear combination of specific wavelengths
    plus Gaussian noise. Wavelengths are evenly spaced from 350nm to 2500nm.

    Parameters
    ----------
    n_samples : int
        Number of spectral samples to generate
    n_wavelengths : int
        Number of wavelength measurements per spectrum
    n_informative : int, default=5
        Number of wavelengths that contribute to target variable
    noise_level : float, default=0.1
        Standard deviation of Gaussian noise added to target
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (n_samples, n_wavelengths)
        Column names are wavelengths as strings (e.g., "1000.0", "1001.0")
    y : pd.Series
        Target values with shape (n_samples,)

    Examples
    --------
    >>> X, y = generate_spectral_data(100, 200, n_informative=3, seed=42)
    >>> X.shape
    (100, 200)
    >>> y.shape
    (100,)
    >>> X.columns[0]  # First wavelength column
    '350.0'
    """
    np.random.seed(seed)

    # Generate wavelengths evenly spaced from 350nm to 2500nm
    wavelengths = np.linspace(350, 2500, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    # Generate spectral data (absorbance values typically 0-2)
    # Use realistic spectral patterns with some baseline and peaks
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Add baseline (slightly variable per sample)
        baseline = 0.3 + np.random.randn() * 0.05

        # Add broad absorption bands (simulate organic molecules)
        for _ in range(3):
            center = np.random.randint(0, n_wavelengths)
            width = np.random.randint(20, 100)
            amplitude = 0.2 + np.random.rand() * 0.3
            gaussian = amplitude * np.exp(
                -((np.arange(n_wavelengths) - center) ** 2) / (2 * width**2)
            )
            X[i] += gaussian

        X[i] += baseline

        # Add small amount of noise
        X[i] += np.random.randn(n_wavelengths) * 0.01

    # Select informative wavelengths (spread across spectrum)
    informative_indices = np.linspace(
        0, n_wavelengths - 1, n_informative, dtype=int
    )

    # Generate target as linear combination of informative wavelengths
    coefficients = np.random.randn(n_informative) * 2  # Random coefficients
    y = X[:, informative_indices] @ coefficients

    # Add noise to target
    y += np.random.randn(n_samples) * noise_level * np.std(y)

    # Create DataFrame and Series
    X_df = pd.DataFrame(X, columns=wavelength_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def generate_outlier_data(
    n_samples: int,
    n_wavelengths: int,
    n_outliers: int = 5,
    outlier_type: Literal["spectral", "leverage", "both"] = "both",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Generate spectral data with known outliers.

    Creates clean spectral data and then adds outliers of different types:
    - spectral: unusual spectral patterns (e.g., spikes, shifts)
    - leverage: extreme X values but consistent with model
    - both: outliers in both X and y space

    Parameters
    ----------
    n_samples : int
        Total number of samples including outliers
    n_wavelengths : int
        Number of wavelength measurements per spectrum
    n_outliers : int, default=5
        Number of outlier samples to inject
    outlier_type : {"spectral", "leverage", "both"}, default="both"
        Type of outliers to generate
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : pd.DataFrame
        Spectral data with outliers, shape (n_samples, n_wavelengths)
    y : pd.Series
        Target values with outliers, shape (n_samples,)
    outlier_indices : np.ndarray
        Indices of outlier samples, shape (n_outliers,)

    Examples
    --------
    >>> X, y, outliers = generate_outlier_data(100, 200, n_outliers=5, seed=42)
    >>> len(outliers)
    5
    >>> outliers[0]  # First outlier index
    0
    """
    # Generate clean base data
    X_df, y_series = generate_spectral_data(
        n_samples - n_outliers, n_wavelengths, seed=seed
    )

    X = X_df.values
    y = y_series.values

    np.random.seed(seed + 1)  # Different seed for outlier generation

    # Outlier indices will be the first n_outliers samples
    outlier_indices = np.arange(n_outliers)

    # Generate outliers
    X_outliers = np.zeros((n_outliers, n_wavelengths))
    y_outliers = np.zeros(n_outliers)

    for i in range(n_outliers):
        if outlier_type in ["spectral", "both"]:
            # Create spectral outliers with unusual patterns
            if i % 3 == 0:
                # Type 1: Random spikes
                X_outliers[i] = np.random.rand(n_wavelengths) * 0.5
                spike_positions = np.random.choice(n_wavelengths, 5, replace=False)
                X_outliers[i, spike_positions] += 2.0

            elif i % 3 == 1:
                # Type 2: Baseline shift
                X_outliers[i] = X[i % len(X)] + 3.0

            else:
                # Type 3: Inverted spectrum
                X_outliers[i] = -X[i % len(X)] + 2.0

        else:
            # Leverage outliers: extreme but consistent
            X_outliers[i] = X[i % len(X)] * 3.0

        if outlier_type in ["both"]:
            # Add y-space outliers
            y_outliers[i] = np.random.randn() * 5 * np.std(y) + np.mean(y)
        else:
            # Keep y consistent with model
            y_outliers[i] = y[i % len(y)]

    # Combine clean data with outliers (outliers at the beginning)
    X_combined = np.vstack([X_outliers, X])
    y_combined = np.hstack([y_outliers, y])

    # Create DataFrame and Series
    wavelength_names = X_df.columns
    X_df_combined = pd.DataFrame(X_combined, columns=wavelength_names)
    y_series_combined = pd.Series(y_combined, name="target")

    return X_df_combined, y_series_combined, outlier_indices


def generate_imbalanced_data(
    n_samples: int,
    n_features: int,
    imbalance_ratio: float = 10.0,
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate imbalanced classification data.

    Creates spectral-like data for classification with severe class imbalance.
    Useful for testing SMOTE and other imbalance handling techniques.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features (wavelengths)
    imbalance_ratio : float, default=10.0
        Ratio of majority to minority class (e.g., 10.0 means 10:1 ratio)
    n_classes : int, default=2
        Number of classes (currently only supports 2)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : pd.DataFrame
        Feature data with shape (n_samples, n_features)
        Column names are wavelengths as strings
    y : pd.Series
        Class labels (0 or 1) with shape (n_samples,)

    Examples
    --------
    >>> X, y = generate_imbalanced_data(110, 200, imbalance_ratio=10.0, seed=42)
    >>> y.value_counts()
    0    100
    1     10
    Name: target, dtype: int64
    """
    if n_classes != 2:
        raise NotImplementedError("Currently only supports binary classification")

    np.random.seed(seed)

    # Calculate samples per class
    minority_samples = int(n_samples / (imbalance_ratio + 1))
    majority_samples = n_samples - minority_samples

    # Generate wavelengths
    wavelengths = np.linspace(350, 2500, n_features)
    wavelength_names = [str(wl) for wl in wavelengths]

    # Generate majority class (class 0)
    X_majority = np.random.randn(majority_samples, n_features) * 0.3 + 0.5
    y_majority = np.zeros(majority_samples, dtype=int)

    # Generate minority class (class 1) with different mean
    X_minority = np.random.randn(minority_samples, n_features) * 0.3 + 0.8
    y_minority = np.ones(minority_samples, dtype=int)

    # Combine and shuffle
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Create DataFrame and Series
    X_df = pd.DataFrame(X, columns=wavelength_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def generate_baseline_data(
    n_samples: int,
    n_wavelengths: int,
    baseline_type: Literal["linear", "polynomial", "offset"] = "linear",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate spectral data with varying baseline effects.

    Creates spectral data with different baseline patterns that need correction.
    Useful for testing baseline correction algorithms (SNV, derivatives, etc.).

    Parameters
    ----------
    n_samples : int
        Number of spectral samples
    n_wavelengths : int
        Number of wavelength measurements per spectrum
    baseline_type : {"linear", "polynomial", "offset"}, default="linear"
        Type of baseline drift to simulate
        - linear: Linear baseline drift across spectrum
        - polynomial: Quadratic baseline curvature
        - offset: Random vertical offset per spectrum
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : pd.DataFrame
        Spectral data with baseline effects, shape (n_samples, n_wavelengths)
    y : pd.Series
        Target values, shape (n_samples,)

    Examples
    --------
    >>> X, y = generate_baseline_data(100, 200, baseline_type="polynomial", seed=42)
    >>> X.shape
    (100, 200)
    """
    # Generate base spectral data
    X_df, y_series = generate_spectral_data(
        n_samples, n_wavelengths, n_informative=3, seed=seed
    )

    X = X_df.values
    np.random.seed(seed + 2)

    # Add baseline effects
    wavelength_positions = np.linspace(0, 1, n_wavelengths)

    for i in range(n_samples):
        if baseline_type == "linear":
            # Linear drift
            slope = np.random.randn() * 0.5
            intercept = np.random.randn() * 0.2
            baseline = slope * wavelength_positions + intercept

        elif baseline_type == "polynomial":
            # Quadratic curvature
            a = np.random.randn() * 0.3
            b = np.random.randn() * 0.2
            c = np.random.randn() * 0.1
            baseline = (
                a * wavelength_positions**2
                + b * wavelength_positions
                + c
            )

        else:  # offset
            # Simple vertical offset
            baseline = np.random.randn() * 0.5

        X[i] += baseline

    # Update DataFrame
    X_df = pd.DataFrame(X, columns=X_df.columns)

    return X_df, y_series


def generate_classification_spectra(
    n_samples: int,
    n_wavelengths: int,
    n_classes: int = 2,
    separation: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate spectral data for classification tasks.

    Creates spectral data where different classes have distinct spectral patterns.
    The separation parameter controls how easily separable the classes are.

    Parameters
    ----------
    n_samples : int
        Total number of samples (will be divided equally among classes)
    n_wavelengths : int
        Number of wavelength measurements per spectrum
    n_classes : int, default=2
        Number of classes to generate
    separation : float, default=1.0
        Controls class separability (higher = more separated)
        Typical values: 0.5 (hard), 1.0 (medium), 2.0 (easy)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : pd.DataFrame
        Spectral data with shape (n_samples, n_wavelengths)
    y : pd.Series
        Class labels (0 to n_classes-1) with shape (n_samples,)

    Examples
    --------
    >>> X, y = generate_classification_spectra(100, 200, n_classes=2, seed=42)
    >>> y.value_counts()
    0    50
    1    50
    Name: target, dtype: int64
    """
    np.random.seed(seed)

    samples_per_class = n_samples // n_classes
    actual_samples = samples_per_class * n_classes

    # Generate wavelengths
    wavelengths = np.linspace(350, 2500, n_wavelengths)
    wavelength_names = [str(wl) for wl in wavelengths]

    X_list = []
    y_list = []

    for class_idx in range(n_classes):
        # Each class gets a unique spectral signature
        # Define class-specific peaks
        n_peaks = 2 + class_idx
        peak_centers = np.linspace(
            n_wavelengths * 0.2,
            n_wavelengths * 0.8,
            n_peaks,
            dtype=int
        )

        X_class = np.zeros((samples_per_class, n_wavelengths))

        for i in range(samples_per_class):
            # Baseline
            baseline = 0.3 + np.random.randn() * 0.05

            # Add class-specific peaks
            for center in peak_centers:
                width = 30 + np.random.randn() * 5
                amplitude = 0.3 * separation + np.random.randn() * 0.1
                gaussian = amplitude * np.exp(
                    -((np.arange(n_wavelengths) - center) ** 2) / (2 * width**2)
                )
                X_class[i] += gaussian

            X_class[i] += baseline

            # Add noise (inversely related to separation)
            noise_level = 0.05 / separation
            X_class[i] += np.random.randn(n_wavelengths) * noise_level

        X_list.append(X_class)
        y_list.append(np.full(samples_per_class, class_idx))

    # Combine all classes
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    shuffle_idx = np.random.permutation(actual_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Create DataFrame and Series
    X_df = pd.DataFrame(X, columns=wavelength_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series
