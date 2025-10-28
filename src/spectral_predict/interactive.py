"""Interactive loading and data exploration module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .preprocess import SavgolDerivative


def plot_spectra_overview(X, output_dir="outputs/plots", window=7):
    """
    Create overview plots of spectral data.

    Generates three plots:
    1. Raw reflectance/absorbance spectra
    2. First derivative spectra
    3. Second derivative spectra

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (wide format, rows = samples, columns = wavelengths)
    output_dir : str
        Directory to save plots
    window : int
        Savitzky-Golay window size for derivatives

    Returns
    -------
    dict
        Paths to the generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = X.columns.values
    n_samples = len(X)

    # Determine if we should show all spectra or sample them
    if n_samples <= 50:
        plot_all = True
        alpha = 0.3
    else:
        plot_all = False
        alpha = 0.5

    plot_paths = {}

    # 1. Raw spectra
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_all:
        for i in range(n_samples):
            ax.plot(wavelengths, X.iloc[i, :], alpha=alpha, color='blue')
    else:
        # Plot a random sample of 50 spectra
        sample_indices = np.random.choice(n_samples, size=min(50, n_samples), replace=False)
        for i in sample_indices:
            ax.plot(wavelengths, X.iloc[i, :], alpha=alpha, color='blue')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title(f'Raw Spectra (n={n_samples})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    raw_path = output_dir / 'spectra_raw.png'
    plt.savefig(raw_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['raw'] = raw_path

    # 2. First derivative
    deriv1_transformer = SavgolDerivative(deriv=1, window=window)
    X_deriv1 = deriv1_transformer.transform(X.values)

    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_all:
        for i in range(n_samples):
            ax.plot(wavelengths, X_deriv1[i, :], alpha=alpha, color='green')
    else:
        for i in sample_indices:
            ax.plot(wavelengths, X_deriv1[i, :], alpha=alpha, color='green')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('First Derivative', fontsize=12)
    ax.set_title(f'First Derivative Spectra (SG window={window}, n={n_samples})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    deriv1_path = output_dir / 'spectra_deriv1.png'
    plt.savefig(deriv1_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['deriv1'] = deriv1_path

    # 3. Second derivative
    deriv2_transformer = SavgolDerivative(deriv=2, window=window)
    X_deriv2 = deriv2_transformer.transform(X.values)

    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_all:
        for i in range(n_samples):
            ax.plot(wavelengths, X_deriv2[i, :], alpha=alpha, color='red')
    else:
        for i in sample_indices:
            ax.plot(wavelengths, X_deriv2[i, :], alpha=alpha, color='red')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Second Derivative', fontsize=12)
    ax.set_title(f'Second Derivative Spectra (SG window={window}, n={n_samples})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    deriv2_path = output_dir / 'spectra_deriv2.png'
    plt.savefig(deriv2_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['deriv2'] = deriv2_path

    return plot_paths


def show_data_preview(X, y=None, n_samples=5, n_wavelengths=10):
    """
    Show a preview of the loaded data as a table.

    Displays sample IDs, first few wavelengths, and target values (if available).

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data
    y : pd.Series, optional
        Target values
    n_samples : int
        Number of samples to show
    n_wavelengths : int
        Number of wavelengths to show

    Returns
    -------
    pd.DataFrame
        Preview table
    """
    # Select first few samples and wavelengths
    preview_X = X.iloc[:n_samples, :n_wavelengths].copy()

    # Add target column if available
    if y is not None:
        preview_X.insert(0, 'Target', y.iloc[:n_samples].values)

    # Add sample ID as a column for better display
    preview_X.insert(0, 'Sample_ID', preview_X.index.values)
    preview_X = preview_X.reset_index(drop=True)

    return preview_X


def reflectance_to_absorbance(X):
    """
    Convert reflectance to absorbance using log10(1/R).

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data in reflectance (0-1 typical range)

    Returns
    -------
    pd.DataFrame
        Spectral data in absorbance
    """
    X_abs = X.copy()

    # Avoid log(0) by clipping very small values
    X_abs = X_abs.clip(lower=1e-6)

    # A = log10(1/R)
    X_abs = np.log10(1.0 / X_abs)

    return X_abs


def compute_predictor_screening(X, y, n_top=20):
    """
    Perform predictor screening similar to JMP.

    Computes correlation between each wavelength and the target,
    and identifies the most informative wavelengths.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (rows = samples, columns = wavelengths)
    y : pd.Series
        Target values
    n_top : int
        Number of top wavelengths to identify

    Returns
    -------
    dict
        Dictionary containing:
        - 'correlations': pd.Series of correlations for each wavelength
        - 'top_wavelengths': List of top n wavelengths
        - 'top_correlations': Corresponding correlation values
    """
    # Compute correlation between each wavelength and target
    correlations = pd.Series(index=X.columns, dtype=float)

    for wl in X.columns:
        corr = np.corrcoef(X[wl].values, y.values)[0, 1]
        correlations[wl] = corr

    # Get absolute correlations for ranking
    abs_corr = correlations.abs()
    top_indices = abs_corr.nlargest(n_top).index

    return {
        'correlations': correlations,
        'top_wavelengths': list(top_indices),
        'top_correlations': correlations[top_indices].values,
        'abs_correlations': abs_corr
    }


def plot_predictor_screening(screening_results, output_dir="outputs/plots"):
    """
    Plot predictor screening results.

    Parameters
    ----------
    screening_results : dict
        Results from compute_predictor_screening()
    output_dir : str
        Directory to save plot

    Returns
    -------
    Path
        Path to the saved plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations = screening_results['correlations']
    top_wls = screening_results['top_wavelengths']
    wavelengths = correlations.index.values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: All correlations
    ax1.plot(wavelengths, correlations.values, color='blue', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Correlation with Target', fontsize=12)
    ax1.set_title('Predictor Screening: Correlation by Wavelength',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Highlight top wavelengths
    for wl in top_wls[:10]:  # Show top 10
        ax1.axvline(x=wl, color='red', alpha=0.3, linestyle=':', linewidth=1)

    # Plot 2: Absolute correlations
    abs_corr = screening_results['abs_correlations']
    ax2.plot(wavelengths, abs_corr.values, color='purple', linewidth=1)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('|Correlation|', fontsize=12)
    ax2.set_title('Absolute Correlation with Target', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight top wavelengths
    for wl in top_wls[:10]:
        ax2.axvline(x=wl, color='red', alpha=0.3, linestyle=':', linewidth=1)

    plt.tight_layout()

    screening_path = output_dir / 'predictor_screening.png'
    plt.savefig(screening_path, dpi=150, bbox_inches='tight')
    plt.close()

    return screening_path


def run_interactive_loading(X, y=None, id_column=None, target=None):
    """
    Run the interactive loading phase with user input.

    This function:
    1. Shows spectral plots (raw, 1st deriv, 2nd deriv)
    2. Shows data preview table
    3. Asks user if they want to convert to absorbance
    4. Performs predictor screening (if target available)
    5. Returns processed data

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data
    y : pd.Series, optional
        Target values
    id_column : str, optional
        ID column name (for display)
    target : str, optional
        Target name (for display)

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Processed spectral data (potentially converted to absorbance)
        - 'y': Target values (unchanged)
        - 'converted_to_absorbance': bool
        - 'screening_results': Predictor screening results (if y provided)
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE LOADING PHASE")
    print("=" * 70)
    print()

    # Step 1: Generate and show spectral plots
    print("Generating spectral plots...")
    plot_paths = plot_spectra_overview(X)
    print(f"  [OK] Raw spectra plot: {plot_paths['raw']}")
    print(f"  [OK] 1st derivative plot: {plot_paths['deriv1']}")
    print(f"  [OK] 2nd derivative plot: {plot_paths['deriv2']}")
    print()
    print("Please review the plots to verify your spectra look correct.")
    print()

    # Step 2: Show data preview
    print("Data Preview:")
    print("-" * 70)
    preview = show_data_preview(X, y)
    print(preview.to_string(index=False))
    print()
    print(f"Full dataset: {len(X)} samples × {X.shape[1]} wavelengths")
    if y is not None:
        print(f"Target '{target}': min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    print()

    # Step 3: Check data range and suggest conversion
    min_val = X.min().min()
    max_val = X.max().max()
    print(f"Spectral value range: {min_val:.4f} to {max_val:.4f}")

    # Auto-detect if data looks like reflectance
    if min_val >= 0 and max_val <= 1.5:
        print("  -> Data appears to be in reflectance format (0-1 range)")
    elif min_val >= 0 and max_val <= 100:
        print("  -> Data appears to be in percent reflectance format (0-100 range)")
        print("  -> Consider dividing by 100 to normalize to 0-1 range")
    else:
        print("  -> Data range is unusual for reflectance")
    print()

    # Step 4: Ask user about absorbance conversion
    print("Would you like to convert reflectance to absorbance?")
    print("(Absorbance is commonly used in programs like Unscrambler)")
    print()
    response = input("Convert to absorbance? [y/N]: ").strip().lower()

    converted_to_absorbance = False
    if response in ['y', 'yes']:
        print("Converting to absorbance (A = log10(1/R))...")
        X = reflectance_to_absorbance(X)
        print("  [OK] Conversion complete")
        converted_to_absorbance = True

        # Show new range
        new_min = X.min().min()
        new_max = X.max().max()
        print(f"  New range: {new_min:.4f} to {new_max:.4f}")
    else:
        print("Keeping data in reflectance format")
    print()

    # Step 5: Predictor screening (if target available)
    screening_results = None
    if y is not None:
        print("Performing predictor screening...")
        screening_results = compute_predictor_screening(X, y, n_top=20)

        print(f"  Top 10 most correlated wavelengths with '{target}':")
        for i, (wl, corr) in enumerate(zip(
            screening_results['top_wavelengths'][:10],
            screening_results['top_correlations'][:10]
        ), 1):
            print(f"    {i:2d}. {wl:8.2f} nm  ->  r = {corr:+.4f}")
        print()

        screening_path = plot_predictor_screening(screening_results)
        print(f"  [OK] Predictor screening plot: {screening_path}")
        print()

        # Interpret results
        max_abs_corr = screening_results['abs_correlations'].max()
        if max_abs_corr > 0.7:
            print(f"  -> Strong correlations detected (max |r| = {max_abs_corr:.3f})")
            print("  -> Your variables of interest are likely present in the spectra")
        elif max_abs_corr > 0.4:
            print(f"  -> Moderate correlations detected (max |r| = {max_abs_corr:.3f})")
            print("  -> Modeling may be possible but could be challenging")
        else:
            print(f"  -> Weak correlations detected (max |r| = {max_abs_corr:.3f})")
            print("  -> Warning: Target may not be well-predicted from these spectra")
        print()

    print("=" * 70)
    print("INTERACTIVE LOADING PHASE COMPLETE")
    print("=" * 70)
    print()
    print("Press Enter to continue to model search...")
    input("")
    print()

    return {
        'X': X,
        'y': y,
        'converted_to_absorbance': converted_to_absorbance,
        'screening_results': screening_results
    }
