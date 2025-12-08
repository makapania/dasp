"""
SPECTRAGRYPH-INSPIRED FEATURES - READY TO USE CODE
===================================================

This file contains 3 working functions you can add to DASP immediately:
1. plot_spectra_by_group() - Visualize groups with different colors
2. subtract_spectra() / average_spectra_by_group() - Spectral operations
3. baseline_als() - Asymmetric Least Squares baseline correction

Estimated implementation time: 3-5 days
Location to add: src/spectral_predict/

Author: Analysis generated 2025-11-14
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================================
# MODULE 1: GROUP-BASED VISUALIZATION (1 day to implement)
# ============================================================================
# File location: src/spectral_predict/visualization.py

def plot_spectra_by_group(X, groups, colors=None, title="Spectra by Group",
                          show_individual=False, figsize=(12, 6)):
    """
    Plot spectra with different colors/styles per group.

    This addresses the critical gap: "viewing different groups of spectra differently"

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra (rows=samples, cols=wavelengths)
    groups : pd.Series or array-like
        Group labels for each sample
    colors : dict, optional
        {group_name: color} mapping. If None, uses default matplotlib colors
    title : str
        Plot title
    show_individual : bool
        If True, plot individual spectra; if False, only plot mean +/- std
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object

    Example:
    --------
    >>> from spectral_predict.io import read_asd_dir
    >>> X = read_asd_dir('data/spectra/')
    >>> metadata = pd.read_csv('data/metadata.csv')
    >>>
    >>> # Plot with automatic colors
    >>> fig = plot_spectra_by_group(X, groups=metadata['treatment'])
    >>> plt.show()
    >>>
    >>> # Plot with custom colors
    >>> colors = {'control': 'blue', 'treated': 'red', 'test': 'green'}
    >>> fig = plot_spectra_by_group(X, groups=metadata['treatment'], colors=colors)
    >>> plt.savefig('spectra_comparison.png', dpi=300)
    """
    fig, ax = plt.subplots(figsize=figsize)
    wavelengths = X.columns.astype(float)

    unique_groups = np.unique(groups)
    if colors is None:
        colors = {g: f'C{i}' for i, g in enumerate(unique_groups)}

    for group in unique_groups:
        mask = groups == group
        X_group = X[mask]

        if show_individual:
            # Plot all individual spectra with transparency
            for idx, row in X_group.iterrows():
                ax.plot(wavelengths, row.values, color=colors[group],
                       alpha=0.3, linewidth=0.5)

        # Plot mean with std band
        mean = X_group.mean()
        std = X_group.std()

        ax.plot(wavelengths, mean, label=f'{group} (n={len(X_group)})',
               color=colors[group], linewidth=2)
        ax.fill_between(wavelengths, mean - std, mean + std,
                        alpha=0.2, color=colors[group])

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    return fig


# ============================================================================
# MODULE 2: SPECTRAL ARITHMETIC (1 day to implement)
# ============================================================================
# File location: src/spectral_predict/operations.py

def subtract_spectra(X, reference):
    """
    Subtract a reference spectrum from all spectra.

    Common uses:
    - Background subtraction
    - Baseline correction
    - Dark current removal

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to correct (rows=samples, cols=wavelengths)
    reference : pd.Series or pd.DataFrame (single row)
        Reference spectrum to subtract

    Returns:
    --------
    pd.DataFrame
        Corrected spectra (X - reference)

    Example:
    --------
    >>> # Load sample and background spectra
    >>> sample_spectra = read_asd_dir('data/samples/')
    >>> background = read_asd_dir('data/background/').iloc[0]  # Single spectrum
    >>>
    >>> # Subtract background
    >>> corrected = subtract_spectra(sample_spectra, background)
    """
    if isinstance(reference, pd.DataFrame):
        reference = reference.iloc[0]

    return X.subtract(reference, axis=1)


def add_spectra(X, reference):
    """
    Add a reference spectrum to all spectra.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra
    reference : pd.Series or pd.DataFrame (single row)
        Reference spectrum to add

    Returns:
    --------
    pd.DataFrame
        Result (X + reference)
    """
    if isinstance(reference, pd.DataFrame):
        reference = reference.iloc[0]

    return X.add(reference, axis=1)


def average_spectra_by_group(X, groups, return_std=False):
    """
    Compute mean spectrum for each group.

    This is essential for comparing experimental conditions.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra (rows=samples, cols=wavelengths)
    groups : pd.Series or array-like
        Group labels for each sample
    return_std : bool
        If True, also return standard deviation

    Returns:
    --------
    pd.DataFrame or tuple
        Mean spectra (rows=groups, cols=wavelengths)
        If return_std=True, returns (means, stds) tuple

    Example:
    --------
    >>> # Average by treatment group
    >>> group_means = average_spectra_by_group(X, groups=metadata['treatment'])
    >>>
    >>> # With standard deviation
    >>> means, stds = average_spectra_by_group(X, groups=metadata['treatment'],
    ...                                         return_std=True)
    >>>
    >>> # Calculate coefficient of variation
    >>> cv = (stds / means) * 100
    """
    X_with_groups = X.copy()
    X_with_groups['_group'] = groups

    means = X_with_groups.groupby('_group').mean()

    if return_std:
        stds = X_with_groups.groupby('_group').std()
        return means, stds
    else:
        return means


def normalize_to_peak(X):
    """
    Normalize spectra so the highest peak = 1.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to normalize

    Returns:
    --------
    pd.DataFrame
        Normalized spectra
    """
    return X.div(X.max(axis=1), axis=0)


def normalize_to_area(X):
    """
    Normalize spectra so the total area (integral) = 1.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to normalize

    Returns:
    --------
    pd.DataFrame
        Area-normalized spectra
    """
    return X.div(X.sum(axis=1), axis=0)


# ============================================================================
# MODULE 3: BASELINE CORRECTION (2 days to implement)
# ============================================================================
# File location: src/spectral_predict/baseline.py

def baseline_als(X, lambda_=1e5, p=0.001, niter=10):
    """
    Asymmetric Least Squares baseline correction.

    This is one of the most important preprocessing steps for spectroscopy.
    The ALS algorithm fits a smooth baseline underneath peaks and subtracts it.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to correct (rows=samples, cols=wavelengths)
    lambda_ : float
        Smoothness parameter (larger = smoother baseline)
        Typical range: 1e2 to 1e9
        - 1e5-1e6: Good for most spectra
        - 1e2-1e4: Less smooth, follows signal more closely
        - 1e7-1e9: Very smooth, good for broad baselines
    p : float
        Asymmetry parameter (0.001 - 0.1)
        - 0.001-0.01: Strong asymmetry (baseline stays under peaks)
        - 0.1: Less asymmetry (baseline can rise above signal)
    niter : int
        Number of iterations (typically 10-20)

    Returns:
    --------
    pd.DataFrame
        Baseline-corrected spectra

    Example:
    --------
    >>> # Standard baseline correction
    >>> X_corrected = baseline_als(X, lambda_=1e5, p=0.001)
    >>>
    >>> # Very smooth baseline for broad background
    >>> X_corrected = baseline_als(X, lambda_=1e7, p=0.001)
    >>>
    >>> # Compare before/after
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    >>> X.iloc[0].plot(ax=ax1, title='Before Baseline Correction')
    >>> X_corrected.iloc[0].plot(ax=ax2, title='After Baseline Correction')

    Reference:
    ----------
    Eilers, P. H. C., & Boelens, H. F. M. (2005).
    Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 1(1), 5.
    """
    X_corrected = X.copy()

    for idx, row in X.iterrows():
        y = row.values
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)

        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lambda_ * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        X_corrected.loc[idx] = y - z

    return X_corrected


def baseline_polynomial(X, degree=3):
    """
    Polynomial baseline correction.

    Fits a polynomial to the spectrum and subtracts it.
    Simpler than ALS but less flexible.

    Parameters:
    -----------
    X : pd.DataFrame
        Spectra to correct
    degree : int
        Polynomial degree (1-5 typical)
        - 1: Linear baseline
        - 2-3: Gentle curved baseline (most common)
        - 4-5: Complex curved baseline

    Returns:
    --------
    pd.DataFrame
        Baseline-corrected spectra

    Example:
    --------
    >>> # Linear baseline
    >>> X_corrected = baseline_polynomial(X, degree=1)
    >>>
    >>> # Quadratic baseline
    >>> X_corrected = baseline_polynomial(X, degree=2)
    """
    X_corrected = X.copy()
    wavelengths = np.arange(len(X.columns))

    for idx, row in X.iterrows():
        y = row.values

        # Fit polynomial
        coeffs = np.polyfit(wavelengths, y, degree)
        baseline = np.polyval(coeffs, wavelengths)

        # Subtract baseline
        X_corrected.loc[idx] = y - baseline

    return X_corrected


# ============================================================================
# COMPLETE USAGE EXAMPLE
# ============================================================================

def example_workflow():
    """
    Complete example showing how to use all three modules together.

    This demonstrates the full workflow:
    1. Load spectra
    2. Baseline correction
    3. Group averaging
    4. Visualization by group
    """
    print("=" * 70)
    print("SPECTRAGRYPH-INSPIRED WORKFLOW EXAMPLE")
    print("=" * 70)

    # Step 1: Load data (using existing DASP functions)
    print("\n1. Loading spectra...")
    from spectral_predict.io import read_asd_dir
    X = read_asd_dir('data/spectra/')
    metadata = pd.read_csv('data/metadata.csv')
    print(f"   Loaded {len(X)} spectra with {len(X.columns)} wavelengths")

    # Step 2: Baseline correction
    print("\n2. Applying baseline correction (ALS)...")
    X_corrected = baseline_als(X, lambda_=1e5, p=0.001)
    print("   Baseline correction complete")

    # Step 3: Group averaging
    print("\n3. Computing group averages...")
    group_means, group_stds = average_spectra_by_group(
        X_corrected,
        groups=metadata['treatment'],
        return_std=True
    )
    print(f"   Computed means for {len(group_means)} groups")

    # Step 4: Visualize by group
    print("\n4. Creating group visualization...")
    fig = plot_spectra_by_group(
        X_corrected,
        groups=metadata['treatment'],
        title="Baseline-Corrected Spectra by Treatment Group"
    )
    plt.savefig('spectra_by_group.png', dpi=300, bbox_inches='tight')
    print("   Saved plot to: spectra_by_group.png")

    # Step 5: Background subtraction (if applicable)
    if 'background' in metadata.columns:
        print("\n5. Subtracting background...")
        background_mask = metadata['background'] == True
        if background_mask.any():
            background_spectrum = X_corrected[background_mask].mean()
            X_final = subtract_spectra(X_corrected, background_spectrum)
            print("   Background subtraction complete")

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("- Use X_corrected for machine learning models")
    print("- Apply existing DASP preprocessing (SNV, derivatives)")
    print("- Run outlier detection")
    print("- Build calibration transfer models")

    return X_corrected, group_means


# ============================================================================
# INTEGRATION WITH EXISTING DASP CODE
# ============================================================================

def integrate_with_dasp_example():
    """
    Example showing how these new functions work with existing DASP features.
    """
    from spectral_predict.io import read_asd_dir, align_xy
    from spectral_predict.preprocess import SNV, SavgolDerivative
    from spectral_predict.outlier_detection import generate_outlier_report

    # Load data
    X = read_asd_dir('data/spectra/')
    metadata = pd.read_csv('data/metadata.csv')
    y = align_xy(X, metadata, 'sample_id', 'target_value')

    # NEW: Baseline correction (SpectralGryph-inspired)
    X = baseline_als(X, lambda_=1e5)

    # EXISTING: DASP preprocessing
    snv = SNV()
    X = snv.transform(X)

    # NEW: Visualize by group (SpectralGryph-inspired)
    plot_spectra_by_group(X, groups=metadata['treatment'])

    # EXISTING: DASP outlier detection
    outliers = generate_outlier_report(X, y)

    # Continue with existing DASP workflow...
    # (wavelength selection, model building, calibration transfer, etc.)


if __name__ == '__main__':
    print(__doc__)
    print("\nThis file contains working code ready to integrate into DASP.")
    print("\nTo use:")
    print("1. Copy functions to appropriate modules in src/spectral_predict/")
    print("2. Run tests to ensure compatibility")
    print("3. Update GUI to expose new functionality")
    print("\nEstimated time: 3-5 days for core features")
