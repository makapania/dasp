"""Test the GUI interactive loading phase."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
from spectral_predict.interactive_gui import run_interactive_loading_gui

print("Loading data from example/quick_start/...")
X = read_asd_dir('example/quick_start/')
ref = read_reference_csv('example/quick_start/reference.csv', 'File Number')
X_aligned, y = align_xy(X, ref, 'File Number', '%Collagen')

print(f"Loaded {len(X_aligned)} samples")
print()
print("Opening GUI...")
print()

# Run GUI
result = run_interactive_loading_gui(X_aligned, y, 'File Number', '%Collagen')

print()
print("=" * 70)
print("GUI RESULTS")
print("=" * 70)
print(f"User continued: {result['user_continue']}")
print(f"Converted to absorbance: {result['converted_to_absorbance']}")
print(f"Final data shape: {result['X'].shape}")
if result['screening_results']:
    max_corr = result['screening_results']['abs_correlations'].max()
    print(f"Max correlation: |r| = {max_corr:.3f}")
print()
