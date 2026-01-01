"""Generate gold standard outputs for numerical validation.

This script generates reference outputs using validated external implementations:
- SNV: Direct formula implementation (X - mean) / std
- Savitzky-Golay derivatives: scipy.signal.savgol_filter
- PLS: sklearn.cross_decomposition.PLSRegression
- Baseline correction: Polynomial using numpy.polyfit

All outputs are deterministic (seed=42) and saved to tests/gold_standards/
for use in numerical correctness tests.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression


def generate_snv_gold_standard(output_dir: Path) -> None:
    """Generate SNV reference using validated formula: (X - mean) / std.

    Parameters
    ----------
    output_dir : Path
        Directory to save gold standard outputs
    """
    print("Generating SNV gold standard...")

    # Create deterministic input (seed=42)
    rng = np.random.RandomState(42)
    X_input = rng.randn(10, 100) * 1000 + 5000  # 10 samples, 100 features

    # Apply SNV formula manually
    means = X_input.mean(axis=1, keepdims=True)
    stds = X_input.std(axis=1, keepdims=True)

    # Avoid division by zero
    stds[stds == 0] = 1.0

    X_output = (X_input - means) / stds

    # Save to tests/gold_standards/snv_outputs.npz
    np.savez_compressed(
        output_dir / "snv_outputs.npz",
        input=X_input,
        output=X_output,
        description="SNV transformation: (X - mean) / std, per row"
    )

    print(f"  Saved: {output_dir / 'snv_outputs.npz'}")
    print(f"  Input shape: {X_input.shape}")
    print(f"  Output shape: {X_output.shape}")
    print(f"  Output mean: {X_output.mean():.2e}, std: {X_output.std():.2f}")


def generate_derivative_gold_standards(output_dir: Path) -> None:
    """Generate Savitzky-Golay derivative references using scipy.signal.

    Parameters
    ----------
    output_dir : Path
        Directory to save gold standard outputs
    """
    print("\nGenerating Savitzky-Golay derivative gold standards...")

    # Create deterministic input
    rng = np.random.RandomState(42)
    X_input = rng.randn(10, 100) * 1000 + 5000  # 10 samples, 100 features

    # Generate 1st derivative (window=7, polyorder=2, deriv=1)
    X_deriv1 = savgol_filter(X_input, window_length=7, polyorder=2, deriv=1, axis=1)

    # Generate 2nd derivative (window=7, polyorder=3, deriv=2)
    X_deriv2 = savgol_filter(X_input, window_length=7, polyorder=3, deriv=2, axis=1)

    # Also test with different parameters
    X_deriv1_alt = savgol_filter(X_input, window_length=11, polyorder=3, deriv=1, axis=1)

    # Save to tests/gold_standards/derivatives.npz
    np.savez_compressed(
        output_dir / "derivatives.npz",
        input=X_input,
        deriv1_w7_p2=X_deriv1,
        deriv2_w7_p3=X_deriv2,
        deriv1_w11_p3=X_deriv1_alt,
        description="Savitzky-Golay derivatives using scipy.signal.savgol_filter"
    )

    print(f"  Saved: {output_dir / 'derivatives.npz'}")
    print(f"  Input shape: {X_input.shape}")
    print(f"  1st derivative (w=7, p=2) shape: {X_deriv1.shape}")
    print(f"  2nd derivative (w=7, p=3) shape: {X_deriv2.shape}")
    print(f"  1st derivative (w=11, p=3) shape: {X_deriv1_alt.shape}")


def generate_pls_gold_standard(output_dir: Path) -> None:
    """Generate PLS coefficients using sklearn.

    Parameters
    ----------
    output_dir : Path
        Directory to save gold standard outputs
    """
    print("\nGenerating PLS gold standard...")

    # Create deterministic X, y data
    rng = np.random.RandomState(42)
    n_samples, n_features = 50, 100

    X = rng.randn(n_samples, n_features) * 1000 + 5000
    y = rng.randn(n_samples) * 10 + 50

    # Fit PLSRegression(n_components=5)
    pls = PLSRegression(n_components=5)
    pls.fit(X, y)

    # Get coefficients, scores, loadings
    X_scores = pls.transform(X)

    # Save to tests/gold_standards/pls_outputs.npz
    np.savez_compressed(
        output_dir / "pls_outputs.npz",
        X=X,
        y=y,
        coefficients=pls.coef_,
        x_weights=pls.x_weights_,
        y_weights=pls.y_weights_,
        x_loadings=pls.x_loadings_,
        y_loadings=pls.y_loadings_,
        x_scores=X_scores,
        n_components=5,
        description="PLS coefficients from sklearn.cross_decomposition.PLSRegression"
    )

    print(f"  Saved: {output_dir / 'pls_outputs.npz'}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Coefficients shape: {pls.coef_.shape}")
    print(f"  X scores shape: {X_scores.shape}")


def generate_baseline_gold_standard(output_dir: Path) -> None:
    """Generate baseline correction reference.

    Creates a spectrum with known polynomial baseline and saves both
    the true signal and baseline separately for validation.

    Parameters
    ----------
    output_dir : Path
        Directory to save gold standard outputs
    """
    print("\nGenerating baseline correction gold standard...")

    # Create wavelength axis
    n_points = 200
    wavelengths = np.linspace(0, n_points - 1, n_points)

    # Define true signal: Gaussian peaks
    rng = np.random.RandomState(42)
    n_peaks = 5
    peak_positions = rng.uniform(30, 170, n_peaks)
    peak_heights = rng.uniform(500, 2000, n_peaks)
    peak_widths = rng.uniform(5, 15, n_peaks)

    true_signal = np.zeros(n_points)
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        true_signal += height * np.exp(-((wavelengths - pos) / width) ** 2)

    # Define polynomial baseline (degree 2)
    baseline_coeffs = np.array([0.5, -20, 1000])  # ax^2 + bx + c
    true_baseline = np.polyval(baseline_coeffs, wavelengths)

    # Observed spectrum = signal + baseline
    observed_spectrum = true_signal + true_baseline

    # Compute corrected spectrum using numpy.polyfit (what we're validating against)
    fitted_coeffs = np.polyfit(wavelengths, observed_spectrum, deg=2)
    fitted_baseline = np.polyval(fitted_coeffs, wavelengths)
    corrected_spectrum = observed_spectrum - fitted_baseline

    # Create multi-sample test case
    n_samples = 10
    X_observed = np.tile(observed_spectrum, (n_samples, 1))
    # Add small variations
    X_observed += rng.randn(n_samples, n_points) * 10

    # Correct each spectrum
    X_corrected = np.zeros_like(X_observed)
    for i in range(n_samples):
        coeffs = np.polyfit(wavelengths, X_observed[i, :], deg=2)
        baseline = np.polyval(coeffs, wavelengths)
        X_corrected[i, :] = X_observed[i, :] - baseline

    # Save to tests/gold_standards/baseline_outputs.npz
    np.savez_compressed(
        output_dir / "baseline_outputs.npz",
        wavelengths=wavelengths,
        true_signal=true_signal,
        true_baseline=true_baseline,
        observed_spectrum=observed_spectrum,
        fitted_baseline=fitted_baseline,
        corrected_spectrum=corrected_spectrum,
        X_observed=X_observed,
        X_corrected=X_corrected,
        polynomial_degree=2,
        description="Baseline correction using numpy.polyfit (degree 2)"
    )

    print(f"  Saved: {output_dir / 'baseline_outputs.npz'}")
    print(f"  Single spectrum length: {len(observed_spectrum)}")
    print(f"  Multi-sample X shape: {X_observed.shape}")
    print(f"  Corrected X shape: {X_corrected.shape}")
    print(f"  True baseline range: [{true_baseline.min():.1f}, {true_baseline.max():.1f}]")


def generate_edge_cases(output_dir: Path) -> None:
    """Generate edge case test data.

    Parameters
    ----------
    output_dir : Path
        Directory to save gold standard outputs
    """
    print("\nGenerating edge case gold standards...")

    # Single sample
    rng = np.random.RandomState(42)
    X_single = rng.randn(1, 100) * 1000 + 5000

    # SNV on single sample
    means = X_single.mean(axis=1, keepdims=True)
    stds = X_single.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X_single_snv = (X_single - means) / stds

    # Constant values (zero variance)
    X_constant = np.ones((5, 100)) * 5000
    # SNV should handle this gracefully (std=0 -> std=1)
    means_const = X_constant.mean(axis=1, keepdims=True)
    stds_const = X_constant.std(axis=1, keepdims=True)
    stds_const[stds_const == 0] = 1.0
    X_constant_snv = (X_constant - means_const) / stds_const  # Should be all zeros

    # Very small window for derivatives
    X_small = rng.randn(5, 20) * 1000 + 5000
    X_small_deriv1 = savgol_filter(X_small, window_length=5, polyorder=2, deriv=1, axis=1)

    # Save edge cases
    np.savez_compressed(
        output_dir / "edge_cases.npz",
        X_single=X_single,
        X_single_snv=X_single_snv,
        X_constant=X_constant,
        X_constant_snv=X_constant_snv,
        X_small=X_small,
        X_small_deriv1=X_small_deriv1,
        description="Edge cases: single sample, constant values, small windows"
    )

    print(f"  Saved: {output_dir / 'edge_cases.npz'}")
    print(f"  Single sample shape: {X_single.shape}")
    print(f"  Constant values shape: {X_constant.shape}")
    print(f"  Small window shape: {X_small.shape}")


def main():
    """Main execution function."""
    # Create output directory
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "tests" / "gold_standards"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Gold Standard Reference Files")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Run all generators
    generate_snv_gold_standard(output_dir)
    generate_derivative_gold_standards(output_dir)
    generate_pls_gold_standard(output_dir)
    generate_baseline_gold_standard(output_dir)
    generate_edge_cases(output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    files = list(output_dir.glob("*.npz"))
    print(f"Generated {len(files)} gold standard files:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    print("\nGold standards ready for numerical validation!")
    print("Run: pytest tests/numerical/test_preprocessing_correctness.py")


if __name__ == "__main__":
    main()
