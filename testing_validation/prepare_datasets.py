"""
Dataset Preparation for DASP Validation Testing
================================================

This script prepares the bone collagen dataset for comprehensive testing:
1. Regression: Predict %Collagen (continuous)
2. Binary Classification: High (>10%) vs. Low (≤10%) collagen
3. Multi-class Classification: 4 classes (A, F, G, H) and 7 classes (all)

Outputs:
- CSV files with train/test splits
- Documentation of sample distributions
- Metadata for reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = Path(__file__).parent / "data"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"
RESULTS_DIR = Path(__file__).parent / "results"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def load_bone_collagen_data():
    """Load the bone collagen reference data."""
    csv_path = EXAMPLE_DIR / "BoneCollagen.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def extract_categorical_labels(df):
    """Extract categorical labels from Sample no. column."""
    # Extract first letter (category)
    df['Category'] = df['Sample no.'].str[0]

    # Count samples per category
    category_counts = df['Category'].value_counts().sort_index()
    print("\nCategory distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} samples")

    return df

def create_binary_classification_target(df, threshold=10.0):
    """Create binary classification target: High vs. Low collagen."""
    df['Binary_Class'] = (df['%Collagen'] > threshold).astype(int)
    df['Binary_Label'] = df['Binary_Class'].map({0: 'Low', 1: 'High'})

    # Statistics
    high_count = (df['Binary_Class'] == 1).sum()
    low_count = (df['Binary_Class'] == 0).sum()
    print(f"\nBinary classification (threshold={threshold}%):")
    print(f"  Low (<={threshold}%): {low_count} samples")
    print(f"  High (>{threshold}%): {high_count} samples")
    print(f"  Balance ratio: {min(high_count, low_count) / max(high_count, low_count):.2f}")

    return df

def create_balanced_4class_dataset(df):
    """Create balanced 4-class dataset using categories A, F, G, H."""
    # Select only the 4 largest categories
    selected_categories = ['A', 'F', 'G', 'H']
    df_4class = df[df['Category'].isin(selected_categories)].copy()

    print(f"\n4-class dataset (A, F, G, H):")
    print(f"  Total samples: {len(df_4class)}")
    for cat in selected_categories:
        count = (df_4class['Category'] == cat).sum()
        print(f"  {cat}: {count} samples")

    return df_4class

def create_train_test_splits(df, test_size=0.25):
    """Create stratified train/test splits for all task types."""
    splits = {}

    # Regression split (stratify by collagen quartiles for balanced distribution)
    collagen_quartiles = pd.qcut(df['%Collagen'], q=4, labels=False, duplicates='drop')
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=collagen_quartiles
    )
    splits['regression'] = {
        'train': df.loc[train_idx].copy(),
        'test': df.loc[test_idx].copy()
    }

    # Binary classification split (stratify by binary class)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df['Binary_Class']
    )
    splits['binary'] = {
        'train': df.loc[train_idx].copy(),
        'test': df.loc[test_idx].copy()
    }

    # 4-class split
    df_4class = create_balanced_4class_dataset(df)
    train_idx, test_idx = train_test_split(
        df_4class.index,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df_4class['Category']
    )
    splits['4class'] = {
        'train': df_4class.loc[train_idx].copy(),
        'test': df_4class.loc[test_idx].copy()
    }

    # 7-class split (all categories - may be imbalanced)
    # For very small classes, we may not be able to stratify
    try:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=df['Category']
        )
        splits['7class'] = {
            'train': df.loc[train_idx].copy(),
            'test': df.loc[test_idx].copy()
        }
    except ValueError as e:
        print(f"\nWarning: Cannot stratify 7-class split due to small classes: {e}")
        print("Creating non-stratified split instead.")
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=RANDOM_SEED
        )
        splits['7class'] = {
            'train': df.loc[train_idx].copy(),
            'test': df.loc[test_idx].copy()
        }

    return splits

def export_splits_to_csv(splits):
    """Export all train/test splits to CSV files."""
    print("\nExporting train/test splits to CSV...")

    for task_type, split_data in splits.items():
        for split_name, data in split_data.items():
            filename = f"{task_type}_{split_name}.csv"
            filepath = DATA_DIR / filename
            data.to_csv(filepath, index=False)
            print(f"  Saved: {filename} ({len(data)} samples)")

def generate_metadata(df, splits):
    """Generate metadata about the datasets."""
    metadata = {
        'random_seed': RANDOM_SEED,
        'total_samples': len(df),
        'test_size': 0.25,
        'collagen_range': {
            'min': float(df['%Collagen'].min()),
            'max': float(df['%Collagen'].max()),
            'mean': float(df['%Collagen'].mean()),
            'std': float(df['%Collagen'].std())
        },
        'categories': {
            cat: int(count) for cat, count in df['Category'].value_counts().sort_index().items()
        },
        'splits': {}
    }

    # Add split information
    for task_type, split_data in splits.items():
        metadata['splits'][task_type] = {
            'train_samples': len(split_data['train']),
            'test_samples': len(split_data['test'])
        }

        # Add class distributions for classification tasks
        if task_type != 'regression':
            target_col = 'Binary_Class' if task_type == 'binary' else 'Category'
            metadata['splits'][task_type]['train_distribution'] = {
                str(k): int(v) for k, v in split_data['train'][target_col].value_counts().sort_index().items()
            }
            metadata['splits'][task_type]['test_distribution'] = {
                str(k): int(v) for k, v in split_data['test'][target_col].value_counts().sort_index().items()
            }

    # Save metadata
    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    return metadata

def generate_summary_report(df, splits, metadata):
    """Generate a human-readable summary report."""
    report_lines = [
        "=" * 80,
        "DASP Validation Testing - Dataset Preparation Summary",
        "=" * 80,
        "",
        f"Random Seed: {RANDOM_SEED}",
        f"Total Samples: {len(df)}",
        f"Collagen Range: {metadata['collagen_range']['min']:.1f}% - {metadata['collagen_range']['max']:.1f}%",
        "",
        "Category Distribution:",
    ]

    for cat, count in sorted(metadata['categories'].items()):
        pct = 100 * count / len(df)
        report_lines.append(f"  {cat}: {count:2d} samples ({pct:5.1f}%)")

    report_lines.extend([
        "",
        "=" * 80,
        "Dataset Splits (75% train / 25% test)",
        "=" * 80,
        ""
    ])

    for task_type, split_info in metadata['splits'].items():
        report_lines.append(f"\n{task_type.upper()} Task:")
        report_lines.append(f"  Train: {split_info['train_samples']} samples")
        report_lines.append(f"  Test:  {split_info['test_samples']} samples")

        if 'train_distribution' in split_info:
            report_lines.append("  Train class distribution:")
            for cls, count in sorted(split_info['train_distribution'].items()):
                report_lines.append(f"    {cls}: {count} samples")
            report_lines.append("  Test class distribution:")
            for cls, count in sorted(split_info['test_distribution'].items()):
                report_lines.append(f"    {cls}: {count} samples")

    report_lines.extend([
        "",
        "=" * 80,
        "Output Files",
        "=" * 80,
        "",
        "CSV Files (in testing_validation/data/):",
        "  - regression_train.csv / regression_test.csv",
        "  - binary_train.csv / binary_test.csv",
        "  - 4class_train.csv / 4class_test.csv",
        "  - 7class_train.csv / 7class_test.csv",
        "  - metadata.json",
        "",
        "All files are ready for use in DASP and R comparisons.",
        "=" * 80
    ])

    report = "\n".join(report_lines)

    # Save report
    report_path = RESULTS_DIR / "dataset_preparation_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSummary report saved to: {report_path}")

    # Also print to console
    print("\n" + report)

    return report

def main():
    """Main execution function."""
    print("=" * 80)
    print("DASP Validation Testing - Dataset Preparation")
    print("=" * 80)

    # Load data
    df = load_bone_collagen_data()

    # Extract categorical labels
    df = extract_categorical_labels(df)

    # Create binary classification target
    df = create_binary_classification_target(df, threshold=10.0)

    # Create train/test splits
    splits = create_train_test_splits(df)

    # Export to CSV
    export_splits_to_csv(splits)

    # Generate metadata
    metadata = generate_metadata(df, splits)

    # Generate summary report
    generate_summary_report(df, splits, metadata)

    print("\n[SUCCESS] Dataset preparation complete!")
    print(f"[SUCCESS] All files saved to: {DATA_DIR}")

if __name__ == "__main__":
    main()
