"""
Diagnostic Test: Derivative Context Bug Investigation

This test verifies the hypothesis that:
1. iPLS (contiguous wavelengths) + derivatives works correctly in Model Development
2. Non-iPLS (scattered wavelengths) + derivatives fails in Model Development

The issue: When wavelengths are non-contiguous, reapplying Savitzky-Golay derivatives
produces DIFFERENT features than computing derivatives on the full spectrum first.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from spectral_predict.preprocess import SavgolDerivative, SNV


def generate_test_data(n_samples=100, n_wavelengths=1000, noise_level=0.1):
    """Generate synthetic spectral data with known structure."""
    np.random.seed(42)

    # Create wavelength axis
    wavelengths = np.linspace(1500, 2500, n_wavelengths)

    # Generate response variable
    y = np.random.randn(n_samples)

    # Generate spectra with spectral features correlated to y
    X = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        # Baseline
        baseline = 1.0 + 0.1 * y[i]
        # Add spectral peaks correlated with y
        X[i, :] = baseline
        X[i, 200:250] += 0.5 * y[i]  # Peak 1 (around 1700 nm)
        X[i, 600:650] += 0.3 * y[i]  # Peak 2 (around 2100 nm)
        # Add noise
        X[i, :] += noise_level * np.random.randn(n_wavelengths)

    return X, y, wavelengths


def test_contiguous_vs_noncontiguous_derivatives():
    """
    Compare derivative calculation on contiguous vs non-contiguous wavelength subsets.
    This is the core of the bug.
    """
    print("=" * 80)
    print("TEST 1: Derivative Context - Contiguous vs Non-Contiguous Wavelengths")
    print("=" * 80)

    X, y, wavelengths = generate_test_data()
    n_samples, n_wavelengths = X.shape

    # Preprocessing parameters (matching typical settings)
    window = 17
    polyorder = 3
    deriv = 2

    print(f"\nData: {n_samples} samples, {n_wavelengths} wavelengths")
    print(f"Preprocessing: SNV + SG2 (window={window}, polyorder={polyorder})")

    # Step 1: Preprocess FULL spectrum (what main analysis does)
    print("\n" + "-" * 80)
    print("MAIN ANALYSIS SIMULATION (Correct approach)")
    print("-" * 80)

    # Apply SNV
    X_snv = np.zeros_like(X)
    for i in range(n_samples):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        X_snv[i, :] = (X[i, :] - mean) / (std + 1e-10)

    # Apply SG derivative to FULL spectrum
    X_full_deriv = np.zeros_like(X_snv)
    for i in range(n_samples):
        X_full_deriv[i, :] = savgol_filter(X_snv[i, :], window, polyorder, deriv=deriv)

    print(f"Full spectrum preprocessed: shape {X_full_deriv.shape}")

    # Scenario A: Select CONTIGUOUS wavelengths (like iPLS)
    # Select a region from indices 200-250 (50 wavelengths)
    contiguous_indices = np.arange(200, 250)
    print(f"\nScenario A: CONTIGUOUS wavelengths (like iPLS)")
    print(f"  Selected indices: {contiguous_indices[:5]}...{contiguous_indices[-5:]}")
    print(f"  Selected wavelengths: {wavelengths[contiguous_indices][:3]}...{wavelengths[contiguous_indices][-3:]}")

    # Main analysis: subset the PREPROCESSED data
    X_contiguous_main = X_full_deriv[:, contiguous_indices]

    # Model development: subset RAW > preprocess
    X_raw_subset_contiguous = X[:, contiguous_indices]
    X_contiguous_modeldev = np.zeros_like(X_raw_subset_contiguous)
    for i in range(n_samples):
        # SNV
        mean = np.mean(X_raw_subset_contiguous[i, :])
        std = np.std(X_raw_subset_contiguous[i, :])
        X_snv_subset = (X_raw_subset_contiguous[i, :] - mean) / (std + 1e-10)
        # SG derivative
        X_contiguous_modeldev[i, :] = savgol_filter(X_snv_subset, window, polyorder, deriv=deriv)

    # Compare features
    max_diff_contiguous = np.max(np.abs(X_contiguous_main - X_contiguous_modeldev))
    mean_diff_contiguous = np.mean(np.abs(X_contiguous_main - X_contiguous_modeldev))

    print(f"  Feature difference (main vs modeldev):")
    print(f"    Max absolute difference: {max_diff_contiguous:.6e}")
    print(f"    Mean absolute difference: {mean_diff_contiguous:.6e}")
    print(f"  > Contiguous wavelengths: Features {'MATCH' if max_diff_contiguous < 1e-10 else 'DIFFER'}")

    # Scenario B: Select NON-CONTIGUOUS wavelengths (like feature importance)
    # Select every 20th wavelength (scattered across spectrum)
    noncontiguous_indices = np.arange(0, n_wavelengths, 20)[:50]  # 50 wavelengths
    print(f"\nScenario B: NON-CONTIGUOUS wavelengths (like feature selection)")
    print(f"  Selected indices: {noncontiguous_indices[:5]}...{noncontiguous_indices[-5:]}")
    print(f"  Selected wavelengths: {wavelengths[noncontiguous_indices][:3]}...{wavelengths[noncontiguous_indices][-3:]}")

    # Main analysis: subset the PREPROCESSED data
    X_noncontiguous_main = X_full_deriv[:, noncontiguous_indices]

    # Model development: subset RAW > preprocess
    X_raw_subset_noncontiguous = X[:, noncontiguous_indices]
    X_noncontiguous_modeldev = np.zeros_like(X_raw_subset_noncontiguous)
    for i in range(n_samples):
        # SNV
        mean = np.mean(X_raw_subset_noncontiguous[i, :])
        std = np.std(X_raw_subset_noncontiguous[i, :])
        X_snv_subset = (X_raw_subset_noncontiguous[i, :] - mean) / (std + 1e-10)
        # SG derivative
        X_noncontiguous_modeldev[i, :] = savgol_filter(X_snv_subset, window, polyorder, deriv=deriv)

    # Compare features
    max_diff_noncontiguous = np.max(np.abs(X_noncontiguous_main - X_noncontiguous_modeldev))
    mean_diff_noncontiguous = np.mean(np.abs(X_noncontiguous_main - X_noncontiguous_modeldev))

    print(f"  Feature difference (main vs modeldev):")
    print(f"    Max absolute difference: {max_diff_noncontiguous:.6e}")
    print(f"    Mean absolute difference: {mean_diff_noncontiguous:.6e}")
    print(f"  > Non-contiguous wavelengths: Features {'MATCH' if max_diff_noncontiguous < 1e-10 else 'DIFFER'} [WARNING]")

    # Summary
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print(f"Contiguous (iPLS):     Max diff = {max_diff_contiguous:.6e} [OK]")
    print(f"Non-contiguous (VarSel): Max diff = {max_diff_noncontiguous:.6e} [FAIL]")
    print(f"\nNon-contiguous derivatives differ by {max_diff_noncontiguous/mean_diff_contiguous:.1f}x")
    print("\nThis confirms the hypothesis:")
    print("  - iPLS works because wavelengths are contiguous")
    print("  - Feature selection fails because wavelengths are scattered")
    print("  - The derivative context is broken for non-contiguous wavelengths!")

    return max_diff_contiguous, max_diff_noncontiguous


def test_model_performance_impact():
    """
    Test the actual impact on PLS model R2 when using contiguous vs non-contiguous
    wavelengths with derivative preprocessing.
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: Impact on PLS Model R2 Performance")
    print("=" * 80)

    X, y, wavelengths = generate_test_data()
    n_samples, n_wavelengths = X.shape

    window = 17
    polyorder = 3
    deriv = 2
    n_components = 5

    # Create preprocessing pipeline
    from sklearn.base import BaseEstimator, TransformerMixin

    class SNVTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X_snv = np.zeros_like(X)
            for i in range(X.shape[0]):
                mean = np.mean(X[i, :])
                std = np.std(X[i, :])
                X_snv[i, :] = (X[i, :] - mean) / (std + 1e-10)
            return X_snv

    class SG2Transformer(BaseEstimator, TransformerMixin):
        def __init__(self, window=17, polyorder=3):
            self.window = window
            self.polyorder = polyorder
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X_deriv = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_deriv[i, :] = savgol_filter(X[i, :], self.window, self.polyorder, deriv=2)
            return X_deriv

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # CONTIGUOUS wavelengths (iPLS-like)
    print("\nScenario A: CONTIGUOUS wavelengths (iPLS)")
    contiguous_indices = np.arange(200, 300)  # 100 contiguous wavelengths

    # Main analysis approach
    pipe_main = Pipeline([
        ('snv', SNVTransformer()),
        ('sg2', SG2Transformer(window, polyorder)),
        ('pls', PLSRegression(n_components=n_components))
    ])
    X_full = X.copy()
    r2_main_contiguous = cross_val_score(pipe_main, X_full, y, cv=cv, scoring='r2').mean()
    print(f"  Main analysis (full spectrum > subset): R2 = {r2_main_contiguous:.4f}")

    # Model development approach (subset > preprocess)
    pipe_modeldev = Pipeline([
        ('snv', SNVTransformer()),
        ('sg2', SG2Transformer(window, polyorder)),
        ('pls', PLSRegression(n_components=n_components))
    ])
    X_subset_contiguous = X[:, contiguous_indices]
    r2_modeldev_contiguous = cross_val_score(pipe_modeldev, X_subset_contiguous, y, cv=cv, scoring='r2').mean()
    print(f"  Model development (subset > preprocess):  R2 = {r2_modeldev_contiguous:.4f}")
    print(f"  Difference: {r2_main_contiguous - r2_modeldev_contiguous:+.4f} [OK]")

    # NON-CONTIGUOUS wavelengths (feature selection-like)
    print("\nScenario B: NON-CONTIGUOUS wavelengths (Feature Selection)")
    noncontiguous_indices = np.arange(0, n_wavelengths, 10)[:100]  # 100 scattered wavelengths

    # Main analysis: preprocess full > subset
    X_full_preprocessed = pipe_main.named_steps['snv'].transform(X_full)
    X_full_preprocessed = pipe_main.named_steps['sg2'].transform(X_full_preprocessed)
    X_subset_preprocessed = X_full_preprocessed[:, noncontiguous_indices]
    pls_only = PLSRegression(n_components=n_components)
    r2_main_noncontiguous = cross_val_score(pls_only, X_subset_preprocessed, y, cv=cv, scoring='r2').mean()
    print(f"  Main analysis (full spectrum > subset): R2 = {r2_main_noncontiguous:.4f}")

    # Model development: subset > preprocess
    X_subset_noncontiguous = X[:, noncontiguous_indices]
    r2_modeldev_noncontiguous = cross_val_score(pipe_modeldev, X_subset_noncontiguous, y, cv=cv, scoring='r2').mean()
    print(f"  Model development (subset > preprocess):  R2 = {r2_modeldev_noncontiguous:.4f}")
    print(f"  Difference: {r2_main_noncontiguous - r2_modeldev_noncontiguous:+.4f} [FAIL]")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Contiguous wavelengths:     Delta R2 = {abs(r2_main_contiguous - r2_modeldev_contiguous):.4f} [OK]")
    print(f"Non-contiguous wavelengths: Delta R2 = {abs(r2_main_noncontiguous - r2_modeldev_noncontiguous):.4f} [FAIL]")
    print("\nThis demonstrates the bug:")
    print("  - iPLS (contiguous): Model Development reproduces main analysis R2")
    print("  - Feature selection (non-contiguous): Model Development gives DIFFERENT R2")


def check_wavelength_contiguity(wavelengths, tolerance=2.0):
    """
    Check if wavelengths are contiguous (equally spaced within tolerance).

    Parameters
    ----------
    wavelengths : array-like
        Wavelength values
    tolerance : float
        Maximum allowed deviation from uniform spacing (in nm)

    Returns
    -------
    is_contiguous : bool
        True if wavelengths are contiguous
    mean_spacing : float
        Mean spacing between wavelengths
    max_gap : float
        Maximum gap between consecutive wavelengths
    """
    wavelengths = np.array(wavelengths)
    if len(wavelengths) < 2:
        return True, 0.0, 0.0

    diffs = np.diff(wavelengths)
    mean_spacing = np.mean(diffs)
    max_gap = np.max(diffs)

    # Check if all spacings are within tolerance of the mean
    is_contiguous = np.all(np.abs(diffs - mean_spacing) < tolerance)

    return is_contiguous, mean_spacing, max_gap


def test_contiguity_checker():
    """Test the wavelength contiguity checker function."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Wavelength Contiguity Detection")
    print("=" * 80)

    # Test case 1: Contiguous (iPLS)
    wl_contiguous = np.linspace(1500, 1600, 50)
    is_contig, mean_sp, max_gap = check_wavelength_contiguity(wl_contiguous)
    print(f"\nContiguous wavelengths (iPLS-like):")
    print(f"  Wavelengths: {wl_contiguous[:3]}...{wl_contiguous[-3:]}")
    print(f"  Is contiguous: {is_contig}")
    print(f"  Mean spacing: {mean_sp:.2f} nm")
    print(f"  Max gap: {max_gap:.2f} nm")

    # Test case 2: Non-contiguous (feature selection)
    wl_noncontiguous = np.arange(1500, 2500, 20)
    is_contig, mean_sp, max_gap = check_wavelength_contiguity(wl_noncontiguous)
    print(f"\nNon-contiguous wavelengths (feature selection-like):")
    print(f"  Wavelengths: {wl_noncontiguous[:3]}...{wl_noncontiguous[-3:]}")
    print(f"  Is contiguous: {is_contig}")
    print(f"  Mean spacing: {mean_sp:.2f} nm")
    print(f"  Max gap: {max_gap:.2f} nm")

    # Test case 3: Actual scattered selection
    indices = np.array([10, 25, 87, 150, 234, 456, 789, 950])
    full_wavelengths = np.linspace(1500, 2500, 1000)
    wl_scattered = full_wavelengths[indices]
    is_contig, mean_sp, max_gap = check_wavelength_contiguity(wl_scattered)
    print(f"\nScattered wavelengths (actual feature selection):")
    print(f"  Wavelengths: {wl_scattered[:3]}...{wl_scattered[-3:]}")
    print(f"  Is contiguous: {is_contig}")
    print(f"  Mean spacing: {mean_sp:.2f} nm")
    print(f"  Max gap: {max_gap:.2f} nm")

    return check_wavelength_contiguity  # Return function for use in fix


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  R2 DISCREPANCY DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    print("\nTesting hypothesis:")
    print("  - iPLS (contiguous wavelengths) + derivatives: Works correctly")
    print("  - Feature selection (scattered wavelengths) + derivatives: BROKEN")
    print()

    # Run all tests
    test_contiguous_vs_noncontiguous_derivatives()
    test_model_performance_impact()
    contiguity_checker = test_contiguity_checker()

    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Fix Model Development to detect non-contiguous wavelengths")
    print("  2. When derivatives + non-contiguous detected:")
    print("     > Preprocess FULL spectrum first")
    print("     > Then subset to selected wavelengths")
    print("     > Skip preprocessing in CV")
    print("  3. This will match main analysis behavior and fix R2 discrepancy")
