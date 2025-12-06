"""
Generate synthetic test fixture for calibration transfer testing.

This creates transfer_pair.npz containing simulated master and slave instrument data
with known instrumental differences.
"""

import numpy as np
from pathlib import Path


def generate_transfer_pair():
    """
    Generate synthetic master/slave instrument pair with realistic differences.
    """
    np.random.seed(42)

    # Instrument parameters
    n_samples = 100
    n_wavelengths = 200
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Generate base spectra (master instrument)
    # Use multiple Gaussian peaks to simulate realistic spectra
    X_master = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Add 3-5 random Gaussian peaks per spectrum
        n_peaks = np.random.randint(3, 6)

        for _ in range(n_peaks):
            center = np.random.uniform(1000, 2500)
            width = np.random.uniform(50, 200)
            amplitude = np.random.uniform(0.5, 2.0)

            peak = amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            X_master[i, :] += peak

        # Add baseline offset
        baseline = np.random.uniform(0.1, 0.3)
        X_master[i, :] += baseline

        # Add small random noise
        X_master[i, :] += np.random.normal(0, 0.01, n_wavelengths)

    # Simulate slave instrument with systematic differences

    # 1. Wavelength-dependent bias (offset)
    wavelength_bias = 0.05 * np.sin(2 * np.pi * (wavelengths - 1000) / 1500)

    # 2. Wavelength-dependent scale (multiplicative)
    wavelength_scale = 0.95 + 0.1 * np.cos(2 * np.pi * (wavelengths - 1000) / 1500)

    # 3. Resolution difference (slight smoothing)
    from scipy.ndimage import gaussian_filter1d
    X_slave = gaussian_filter1d(X_master, sigma=1.5, axis=1)

    # Apply bias and scale
    X_slave = X_slave * wavelength_scale + wavelength_bias

    # 4. Add independent noise
    X_slave += np.random.normal(0, 0.015, X_slave.shape)

    # Select transfer samples (use diverse subset)
    n_transfer = 20
    transfer_indices = np.linspace(0, n_samples - 1, n_transfer, dtype=int)

    # Generate reference values (for methods that need Y)
    # Create a property correlated with spectral features
    y_reference = (
        2.0 * X_master[:, 50].mean(axis=0) -
        1.5 * X_master[:, 100].mean(axis=0) +
        0.5 * X_master[:, 150].mean(axis=0) +
        np.random.normal(0, 0.1, n_samples)
    )

    # Save to fixture file
    fixture_path = Path(__file__).parent / "transfer_pair.npz"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        fixture_path,
        wavelengths=wavelengths,
        X_master=X_master,
        X_slave=X_slave,
        transfer_indices=transfer_indices,
        y_reference=y_reference,
        wavelength_bias=wavelength_bias,
        wavelength_scale=wavelength_scale,
    )

    print(f"Generated transfer pair fixture:")
    print(f"  Samples: {n_samples}")
    print(f"  Wavelengths: {n_wavelengths} ({wavelengths.min():.1f} - {wavelengths.max():.1f} nm)")
    print(f"  Transfer samples: {n_transfer}")
    print(f"  Saved to: {fixture_path}")
    print(f"\nInstrumental differences:")
    print(f"  Wavelength bias range: [{wavelength_bias.min():.4f}, {wavelength_bias.max():.4f}]")
    print(f"  Scale range: [{wavelength_scale.min():.4f}, {wavelength_scale.max():.4f}]")
    print(f"  Resolution: Slave smoothed with sigma=1.5")

    return fixture_path


if __name__ == "__main__":
    generate_transfer_pair()
