"""Run interactive demo with automatic plot display."""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
from spectral_predict.interactive import run_interactive_loading

print("=" * 70)
print("RUNNING INTERACTIVE DEMO")
print("=" * 70)
print()

# Load data
print("Loading spectral data from example/quick_start/...")
X = read_asd_dir('example/quick_start/')
print(f"  Loaded {len(X)} spectra with {X.shape[1]} wavelengths")
print()

print("Loading reference data...")
ref = read_reference_csv('example/quick_start/reference.csv', 'File Number')
print(f"  Loaded reference with {len(ref)} samples")
print()

print("Aligning data...")
X_aligned, y = align_xy(X, ref, 'File Number', '%Collagen')
print(f"  Aligned {len(X_aligned)} samples for target '%Collagen'")
print()

# Run interactive loading (simulating user responses)
# We'll use a mock for input to auto-respond
import builtins
original_input = builtins.input

def mock_input(prompt=""):
    """Mock input function that auto-responds."""
    print(prompt, end='')
    if "Convert to absorbance" in prompt:
        response = 'n'
        print(response)
        return response
    elif "Press Enter" in prompt or prompt == "":
        response = ''
        print()
        return response
    else:
        print('<auto>')
        return ''

builtins.input = mock_input

try:
    result = run_interactive_loading(X_aligned, y, 'File Number', '%Collagen')
finally:
    builtins.input = original_input

print()
print("=" * 70)
print("OPENING PLOTS...")
print("=" * 70)

# Open the plots
import matplotlib.pyplot as plt
from matplotlib.image import imread

plot_files = [
    'outputs/plots/spectra_raw.png',
    'outputs/plots/spectra_deriv1.png',
    'outputs/plots/spectra_deriv2.png',
    'outputs/plots/predictor_screening.png'
]

for plot_file in plot_files:
    if Path(plot_file).exists():
        print(f"Opening {plot_file}...")
        # Open with default system viewer
        if sys.platform == 'win32':
            subprocess.run(['start', plot_file], shell=True)
        elif sys.platform == 'darwin':
            subprocess.run(['open', plot_file])
        else:
            subprocess.run(['xdg-open', plot_file])

print()
print("All plots opened in your default image viewer!")
print()
print("Summary:")
print(f"  - Data converted to absorbance: {result['converted_to_absorbance']}")
print(f"  - Final data shape: {result['X'].shape}")
if result['screening_results']:
    max_corr = result['screening_results']['abs_correlations'].max()
    print(f"  - Max correlation: |r| = {max_corr:.3f}")
