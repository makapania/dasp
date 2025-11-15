"""
Export Spectral Data for R Comparison
======================================

This script loads the ASD spectral files and exports them to CSV format
that R can easily read, paired with the appropriate train/test splits.

Outputs:
- X_train.csv, X_test.csv: Spectral matrices (rows=samples, cols=wavelengths)
- wavelengths.csv: Wavelength values
- Separate files for regression, binary, 4class, 7class tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import spectral_predict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spectral_predict.io import read_asd_dir

# Paths
DATA_DIR = Path(__file__).parent / "data"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"
R_DATA_DIR = Path(__file__).parent / "r_data"

# Create output directory
R_DATA_DIR.mkdir(exist_ok=True)

# Load all spectra once
print("Loading all spectra from example directory...")
SPECTRA_DF, METADATA = read_asd_dir(EXAMPLE_DIR, reader_mode="auto")
print(f"Loaded {len(SPECTRA_DF)} spectra with {SPECTRA_DF.shape[1]} wavelengths")
print(f"Wavelength range: {METADATA['wavelength_range'][0]:.1f} - {METADATA['wavelength_range'][1]:.1f} nm")

def get_spectra_for_file_numbers(file_numbers):
    """Extract spectra for given file numbers from the loaded data."""
    # Format: "Spectrum 00001" -> "Spectrum00001"
    file_stems = [f"Spectrum{fn.split()[-1]}" for fn in file_numbers]

    # Check for missing files
    missing = [stem for stem in file_stems if stem not in SPECTRA_DF.index]
    if missing:
        raise ValueError(f"Missing spectra: {missing}")

    # Extract rows
    X = SPECTRA_DF.loc[file_stems].values
    wavelengths = SPECTRA_DF.columns.values

    return X, wavelengths

def export_task_data(task_name, train_csv, test_csv):
    """Export spectral data for a specific task."""
    print(f"\nProcessing {task_name} task...")

    # Load train and test splits
    train_df = pd.read_csv(DATA_DIR / train_csv)
    test_df = pd.read_csv(DATA_DIR / test_csv)

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Load spectra
    print("  Extracting train spectra...")
    X_train, wavelengths = get_spectra_for_file_numbers(train_df['File Number'].values)

    print("  Extracting test spectra...")
    X_test, _ = get_spectra_for_file_numbers(test_df['File Number'].values)

    # Create DataFrames
    # Column names: WL_350.0, WL_351.0, etc.
    col_names = [f"WL_{wl:.1f}" for wl in wavelengths]

    X_train_df = pd.DataFrame(X_train, columns=col_names)
    X_test_df = pd.DataFrame(X_test, columns=col_names)

    # Export to CSV
    output_dir = R_DATA_DIR / task_name
    output_dir.mkdir(exist_ok=True)

    X_train_df.to_csv(output_dir / "X_train.csv", index=False)
    X_test_df.to_csv(output_dir / "X_test.csv", index=False)

    # Export wavelengths
    wl_df = pd.DataFrame({'wavelength': wavelengths})
    wl_df.to_csv(output_dir / "wavelengths.csv", index=False)

    # Export reference data (without File Number column for convenience)
    y_cols = [col for col in train_df.columns if col != 'File Number']
    train_df[y_cols].to_csv(output_dir / "y_train.csv", index=False)
    test_df[y_cols].to_csv(output_dir / "y_test.csv", index=False)

    print(f"  Exported to: {output_dir}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Wavelengths: {len(wavelengths)} ({wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm)")

    return {
        'task': task_name,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_wavelengths': len(wavelengths),
        'wavelength_range': [float(wavelengths[0]), float(wavelengths[-1])]
    }

def main():
    """Main execution function."""
    print("=" * 80)
    print("Exporting Spectral Data for R Comparison")
    print("=" * 80)

    # Export for all tasks
    tasks = [
        ('regression', 'regression_train.csv', 'regression_test.csv'),
        ('binary', 'binary_train.csv', 'binary_test.csv'),
        ('4class', '4class_train.csv', '4class_test.csv'),
        ('7class', '7class_train.csv', '7class_test.csv')
    ]

    metadata = {}

    for task_name, train_csv, test_csv in tasks:
        task_metadata = export_task_data(task_name, train_csv, test_csv)
        metadata[task_name] = task_metadata

    # Export metadata
    import json
    with open(R_DATA_DIR / "export_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nAll data exported to: {R_DATA_DIR}")
    print("\nEach task directory contains:")
    print("  - X_train.csv: Training spectra matrix")
    print("  - X_test.csv: Test spectra matrix")
    print("  - y_train.csv: Training reference values")
    print("  - y_test.csv: Test reference values")
    print("  - wavelengths.csv: Wavelength values")
    print("\nReady for R comparison scripts!")

if __name__ == "__main__":
    main()
