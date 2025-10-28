"""Quick test of the interactive module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy

# Load small subset
print("Loading data...")
X = read_asd_dir('example/quick_start/')
ref = read_reference_csv('example/quick_start/reference.csv', 'File Number')
X_aligned, y = align_xy(X, ref, 'File Number', '%Collagen')

print(f"Loaded {len(X_aligned)} samples")
print(f"Spectral data shape: {X_aligned.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")
print()

# Test just the plotting functions without user interaction
from spectral_predict.interactive import (
    plot_spectra_overview,
    show_data_preview,
    compute_predictor_screening,
    plot_predictor_screening
)

# Test plots
print("Testing spectral plots...")
plot_paths = plot_spectra_overview(X_aligned)
for key, path in plot_paths.items():
    print(f"  OK {key}: {path}")
print()

# Test preview
print("Testing data preview...")
preview = show_data_preview(X_aligned, y)
print(preview.to_string(index=False))
print()

# Test screening
print("Testing predictor screening...")
screening = compute_predictor_screening(X_aligned, y)
print(f"  Top 5 wavelengths:")
for i, (wl, corr) in enumerate(zip(screening['top_wavelengths'][:5],
                                     screening['top_correlations'][:5]), 1):
    print(f"    {i}. {wl:.2f} nm -> r = {corr:+.4f}")
print()

print("Generating screening plot...")
screening_path = plot_predictor_screening(screening)
print(f"  OK {screening_path}")
print()

print("All tests passed!")
