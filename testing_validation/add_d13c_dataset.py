"""
Add d13C Enamel Dataset to Testing Framework
=============================================

This script integrates the d13C enamel dataset into the testing validation framework.

Dataset details:
- Source: Desktop/ellie/
- Samples: ~152 enamel samples
- Target: d13C (carbon isotope ratio)
- Spectral files: ASD format (NIR)
- CSV: Ellie_NIR_Data.csv

Outputs:
- Copies data to testing_validation/data_sources/d13c/
- Creates train/test splits
- Exports spectral matrices for R
- Generates metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.spectral_predict.io import read_asd_dir

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
SOURCE_DIR = Path(r"C:\Users\sponheim\Desktop\ellie")
BASE_DIR = Path(__file__).parent
DATA_SOURCES_DIR = BASE_DIR / "data_sources" / "d13c"
DATA_DIR = BASE_DIR / "data"
R_DATA_DIR = BASE_DIR / "r_data"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
DATA_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
(R_DATA_DIR / "d13c").mkdir(parents=True, exist_ok=True)

def copy_source_data():
    """Copy source data to testing directory."""
    print("=" * 80)
    print("Copying d13C Enamel Data to Testing Directory")
    print("=" * 80)

    # Copy CSV
    source_csv = SOURCE_DIR / "Ellie_NIR_Data.csv"
    dest_csv = DATA_SOURCES_DIR / "Ellie_NIR_Data.csv"

    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    shutil.copy2(source_csv, dest_csv)
    print(f"\nCopied CSV: {dest_csv}")

    # Copy ASD files
    asd_files = list(SOURCE_DIR.glob("*.asd"))
    print(f"Found {len(asd_files)} ASD files")

    if len(asd_files) == 0:
        raise ValueError("No ASD files found in source directory")

    # Create spectra directory
    spectra_dir = DATA_SOURCES_DIR / "spectra"
    spectra_dir.mkdir(exist_ok=True)

    print("Copying spectral files...")
    for i, asd_file in enumerate(asd_files):
        shutil.copy2(asd_file, spectra_dir / asd_file.name)
        if (i + 1) % 20 == 0:
            print(f"  Copied {i + 1}/{len(asd_files)} files...")

    print(f"  Copied all {len(asd_files)} files to {spectra_dir}")

    return dest_csv, spectra_dir

def load_and_clean_reference_data(csv_path):
    """Load and clean the reference CSV data."""
    print("\n" + "=" * 80)
    print("Loading and Cleaning Reference Data")
    print("=" * 80)

    # Read CSV with proper handling of whitespace in column names
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    print(f"\nLoaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Check for d13C column
    if 'd13C' not in df.columns:
        raise ValueError(f"d13C column not found. Available columns: {df.columns.tolist()}")

    # Extract relevant columns
    # Sample identifier, d13C target, and other metadata
    df_clean = df[['sample number', 'd13C']].copy()

    # Rename for consistency
    df_clean.columns = ['Sample_ID', 'd13C']

    # Remove rows with missing d13C values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['d13C'])
    final_count = len(df_clean)

    if final_count < initial_count:
        print(f"\nWarning: Removed {initial_count - final_count} samples with missing d13C values")

    print(f"\nFinal dataset: {final_count} samples")
    print(f"d13C range: {df_clean['d13C'].min():.2f} to {df_clean['d13C'].max():.2f}")
    print(f"d13C mean: {df_clean['d13C'].mean():.2f}")
    print(f"d13C std: {df_clean['d13C'].std():.2f}")

    return df_clean

def match_spectra_to_reference(df_reference, spectra_dir):
    """Match spectral files to reference data."""
    print("\n" + "=" * 80)
    print("Matching Spectral Files to Reference Data")
    print("=" * 80)

    # Load all spectra
    print(f"\nLoading spectra from {spectra_dir}...")
    spectra_df, metadata = read_asd_dir(spectra_dir, reader_mode="auto")

    print(f"Loaded {len(spectra_df)} spectra")
    print(f"Wavelength range: {metadata['wavelength_range'][0]:.1f} - {metadata['wavelength_range'][1]:.1f} nm")
    print(f"Number of wavelengths: {spectra_df.shape[1]}")

    # Match sample IDs
    # Spectral filenames are like "04-TSV-101" (without .asd extension)
    # Reference sample IDs are like "04-TSV-101"

    matched_samples = []
    unmatched_ref = []
    unmatched_spec = []

    for sample_id in df_reference['Sample_ID']:
        if sample_id in spectra_df.index:
            matched_samples.append(sample_id)
        else:
            unmatched_ref.append(sample_id)

    unmatched_spec = [idx for idx in spectra_df.index if idx not in df_reference['Sample_ID'].values]

    print(f"\nMatching results:")
    print(f"  Matched samples: {len(matched_samples)}")
    print(f"  Unmatched in reference: {len(unmatched_ref)}")
    print(f"  Unmatched in spectra: {len(unmatched_spec)}")

    if len(unmatched_ref) > 0:
        print(f"\n  First few unmatched reference samples: {unmatched_ref[:5]}")

    if len(unmatched_spec) > 0:
        print(f"  First few unmatched spectral files: {unmatched_spec[:5]}")

    # Create matched dataset
    df_matched = df_reference[df_reference['Sample_ID'].isin(matched_samples)].copy()

    print(f"\nFinal matched dataset: {len(df_matched)} samples")

    return df_matched, spectra_df, metadata

def create_train_test_split(df_matched, test_size=0.25):
    """Create train/test split stratified by d13C quartiles."""
    print("\n" + "=" * 80)
    print("Creating Train/Test Split")
    print("=" * 80)

    # Create quartiles for stratification
    df_matched['d13C_quartile'] = pd.qcut(df_matched['d13C'], q=4, labels=False, duplicates='drop')

    # Split
    train_idx, test_idx = train_test_split(
        df_matched.index,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df_matched['d13C_quartile']
    )

    train_df = df_matched.loc[train_idx].copy()
    test_df = df_matched.loc[test_idx].copy()

    print(f"\nTrain samples: {len(train_df)}")
    print(f"  d13C range: {train_df['d13C'].min():.2f} to {train_df['d13C'].max():.2f}")
    print(f"  d13C mean: {train_df['d13C'].mean():.2f}")

    print(f"\nTest samples: {len(test_df)}")
    print(f"  d13C range: {test_df['d13C'].min():.2f} to {test_df['d13C'].max():.2f}")
    print(f"  d13C mean: {test_df['d13C'].mean():.2f}")

    # Save splits
    train_csv = DATA_DIR / "d13c_train.csv"
    test_csv = DATA_DIR / "d13c_test.csv"

    train_df[['Sample_ID', 'd13C']].to_csv(train_csv, index=False)
    test_df[['Sample_ID', 'd13C']].to_csv(test_csv, index=False)

    print(f"\nSaved train split: {train_csv}")
    print(f"Saved test split: {test_csv}")

    return train_df, test_df

def export_spectral_data_for_r(train_df, test_df, spectra_df):
    """Export spectral matrices for R."""
    print("\n" + "=" * 80)
    print("Exporting Spectral Data for R")
    print("=" * 80)

    # Extract spectral matrices
    X_train = spectra_df.loc[train_df['Sample_ID']].values
    X_test = spectra_df.loc[test_df['Sample_ID']].values
    wavelengths = spectra_df.columns.values

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Wavelengths: {len(wavelengths)} ({wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm)")

    # Create DataFrames
    col_names = [f"WL_{wl:.1f}" for wl in wavelengths]
    X_train_df = pd.DataFrame(X_train, columns=col_names)
    X_test_df = pd.DataFrame(X_test, columns=col_names)

    # Output directory
    output_dir = R_DATA_DIR / "d13c"
    output_dir.mkdir(exist_ok=True)

    # Export
    X_train_df.to_csv(output_dir / "X_train.csv", index=False)
    X_test_df.to_csv(output_dir / "X_test.csv", index=False)

    # Export reference values
    train_df[['Sample_ID', 'd13C']].to_csv(output_dir / "y_train.csv", index=False)
    test_df[['Sample_ID', 'd13C']].to_csv(output_dir / "y_test.csv", index=False)

    # Export wavelengths
    wl_df = pd.DataFrame({'wavelength': wavelengths})
    wl_df.to_csv(output_dir / "wavelengths.csv", index=False)

    print(f"\nExported to: {output_dir}")
    print("Files:")
    print("  - X_train.csv")
    print("  - X_test.csv")
    print("  - y_train.csv")
    print("  - y_test.csv")
    print("  - wavelengths.csv")

    return output_dir

def generate_metadata(df_matched, train_df, test_df, spectra_metadata):
    """Generate metadata about the d13C dataset."""
    print("\n" + "=" * 80)
    print("Generating Metadata")
    print("=" * 80)

    metadata = {
        'dataset_name': 'd13C_enamel',
        'description': 'Enamel d13C (carbon isotope ratio) prediction from NIR spectra',
        'random_seed': RANDOM_SEED,
        'total_samples': len(df_matched),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'test_size': 0.25,
        'd13C_range': {
            'min': float(df_matched['d13C'].min()),
            'max': float(df_matched['d13C'].max()),
            'mean': float(df_matched['d13C'].mean()),
            'std': float(df_matched['d13C'].std())
        },
        'spectral_info': {
            'n_wavelengths': spectra_metadata['n_spectra'],
            'wavelength_range': [float(x) for x in spectra_metadata['wavelength_range']],
            'data_type': spectra_metadata['data_type'],
            'file_format': 'asd'
        },
        'source_location': str(SOURCE_DIR),
        'date_added': pd.Timestamp.now().isoformat()
    }

    # Save metadata
    metadata_path = DATA_DIR / "d13c_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("D13C Dataset Summary")
    print("=" * 80)
    print(f"\nDataset: {metadata['dataset_name']}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Train: {metadata['train_samples']}, Test: {metadata['test_samples']}")
    print(f"\nd13C range: {metadata['d13C_range']['min']:.2f} to {metadata['d13C_range']['max']:.2f}")
    print(f"d13C mean: {metadata['d13C_range']['mean']:.2f} +/- {metadata['d13C_range']['std']:.2f}")
    print(f"\nWavelengths: {metadata['spectral_info']['n_wavelengths']}")
    print(f"Range: {metadata['spectral_info']['wavelength_range'][0]:.1f} - {metadata['spectral_info']['wavelength_range'][1]:.1f} nm")
    print(f"Data type: {metadata['spectral_info']['data_type']}")

    return metadata

def main():
    """Main execution function."""
    print("=" * 80)
    print("Adding d13C Enamel Dataset to Testing Framework")
    print("=" * 80)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Destination: {DATA_SOURCES_DIR}")

    # Step 1: Copy source data
    csv_path, spectra_dir = copy_source_data()

    # Step 2: Load and clean reference data
    df_reference = load_and_clean_reference_data(csv_path)

    # Step 3: Match spectra to reference
    df_matched, spectra_df, spectra_metadata = match_spectra_to_reference(df_reference, spectra_dir)

    # Step 4: Create train/test split
    train_df, test_df = create_train_test_split(df_matched)

    # Step 5: Export for R
    r_output_dir = export_spectral_data_for_r(train_df, test_df, spectra_df)

    # Step 6: Generate metadata
    metadata = generate_metadata(df_matched, train_df, test_df, spectra_metadata)

    print("\n" + "=" * 80)
    print("[SUCCESS] d13C Dataset Successfully Added!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Update R regression scripts to include d13c task")
    print("  2. Run regression testing on d13C dataset")
    print("  3. Compare DASP vs. R results")
    print("=" * 80)

if __name__ == "__main__":
    main()
