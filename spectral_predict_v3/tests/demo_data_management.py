"""
Demo script for data_management module.

Shows examples of merging spectral data sources with different strategies.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_management import DataSource, merge_sources, print_merge_summary


def create_example_source(name, wl_start, wl_end, n_samples=10):
    """Create example spectral data source."""
    wavelengths = np.arange(wl_start, wl_end, 10, dtype=np.float64)
    n_wavelengths = len(wavelengths)

    # Simulate spectral data with some structure
    X = np.random.randn(n_samples, n_wavelengths) + np.linspace(0, 2, n_wavelengths)

    # Create sample IDs
    sample_ids = [f"{name}_sample_{i:03d}" for i in range(n_samples)]

    # Create target values
    y = np.random.randn(n_samples) * 10 + 50  # Around 50 +/- 10

    return DataSource(
        source_id=f"src_{name}",
        name=name,
        path=f"/data/{name}.csv",
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        y=y,
        target_name="concentration"
    )


def demo_intersection():
    """Demonstrate intersection merge strategy."""
    print("\n" + "="*80)
    print("DEMO 1: INTERSECTION MERGE")
    print("="*80)
    print("Scenario: Two labs measured samples with partially overlapping wavelength ranges")
    print("Strategy: Keep only common wavelengths\n")

    # Lab 1: Measured 400-600 nm
    source1 = create_example_source("lab1", 400, 600, n_samples=15)

    # Lab 2: Measured 500-700 nm
    source2 = create_example_source("lab2", 500, 700, n_samples=12)

    result = merge_sources([source1, source2], strategy='intersection', dup_handling='rename')

    print_merge_summary(result)
    print(f"Common wavelength range preserved: {result.wavelengths[0]:.1f} to {result.wavelengths[-1]:.1f} nm")


def demo_union():
    """Demonstrate union merge strategy."""
    print("\n" + "="*80)
    print("DEMO 2: UNION MERGE")
    print("="*80)
    print("Scenario: Combine data from different instruments with different wavelength ranges")
    print("Strategy: Keep all wavelengths, fill missing with NaN\n")

    # NIR instrument: 1000-2000 nm
    source1 = create_example_source("nir", 1000, 1200, n_samples=8)

    # Visible instrument: 400-700 nm
    source2 = create_example_source("vis", 400, 700, n_samples=10)

    result = merge_sources([source1, source2], strategy='union', dup_handling='rename')

    print_merge_summary(result)
    print(f"Total wavelength range: {result.wavelengths[0]:.1f} to {result.wavelengths[-1]:.1f} nm")
    print(f"NaN content: {result.report['nan_percent']:.1f}% (expected due to non-overlapping ranges)")


def demo_interpolation():
    """Demonstrate interpolation merge strategy."""
    print("\n" + "="*80)
    print("DEMO 3: INTERPOLATION MERGE")
    print("="*80)
    print("Scenario: Combine data from instruments with different wavelength spacings")
    print("Strategy: Interpolate all to common uniform grid\n")

    # Source 1: 5 nm spacing
    wl1 = np.arange(400, 700, 5, dtype=np.float64)
    X1 = np.random.randn(10, len(wl1))
    source1 = DataSource(
        source_id="src_fine",
        name="fine_spacing",
        path="/data/fine.csv",
        X=X1,
        wavelengths=wl1,
        sample_ids=[f"fine_{i}" for i in range(10)],
        y=np.random.randn(10) * 10 + 50,
        target_name="concentration"
    )

    # Source 2: 15 nm spacing
    wl2 = np.arange(405, 705, 15, dtype=np.float64)
    X2 = np.random.randn(8, len(wl2))
    source2 = DataSource(
        source_id="src_coarse",
        name="coarse_spacing",
        path="/data/coarse.csv",
        X=X2,
        wavelengths=wl2,
        sample_ids=[f"coarse_{i}" for i in range(8)],
        y=np.random.randn(8) * 10 + 50,
        target_name="concentration"
    )

    result = merge_sources([source1, source2], strategy='interpolation',
                          wavelength_step=10.0, dup_handling='rename')

    print_merge_summary(result)
    print(f"Uniform grid: {result.wavelengths[0]:.1f} to {result.wavelengths[-1]:.1f} nm "
          f"with {result.report['wavelength_step']:.1f} nm spacing")


def demo_duplicate_handling():
    """Demonstrate duplicate sample ID handling."""
    print("\n" + "="*80)
    print("DEMO 4: DUPLICATE SAMPLE ID HANDLING")
    print("="*80)
    print("Scenario: Two sources have overlapping sample IDs")
    print("Strategy comparison: rename vs keep_first\n")

    # Both sources have samples named "sample_001", "sample_002", etc.
    wl = np.arange(400, 500, 10, dtype=np.float64)

    X1 = np.random.randn(5, len(wl))
    source1 = DataSource(
        source_id="src1",
        name="dataset_A",
        path="/data/A.csv",
        X=X1,
        wavelengths=wl,
        sample_ids=[f"sample_{i:03d}" for i in range(5)]
    )

    X2 = np.random.randn(5, len(wl))
    source2 = DataSource(
        source_id="src2",
        name="dataset_B",
        path="/data/B.csv",
        X=X2,
        wavelengths=wl,
        sample_ids=[f"sample_{i:03d}" for i in range(5)]  # Same IDs!
    )

    print("Strategy 1: RENAME duplicates")
    result1 = merge_sources([source1, source2], strategy='intersection', dup_handling='rename')
    print(f"  Result: {len(result1.sample_ids)} samples")
    print(f"  Sample IDs: {result1.sample_ids[:3]} ... {result1.sample_ids[-2:]}")

    print("\nStrategy 2: KEEP_FIRST (skip duplicates)")
    result2 = merge_sources([source1, source2], strategy='intersection', dup_handling='keep_first')
    print(f"  Result: {len(result2.sample_ids)} samples (only from dataset_A)")
    print(f"  Sample IDs: {result2.sample_ids}")


if __name__ == '__main__':
    print("\n")
    print("*" * 80)
    print(" " * 20 + "DATA MANAGEMENT MODULE DEMO")
    print("*" * 80)

    demo_intersection()
    demo_union()
    demo_interpolation()
    demo_duplicate_handling()

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nKey takeaways:")
    print("  - Intersection: Only common wavelengths (safest for analysis)")
    print("  - Union: All wavelengths with NaN fill (preserves all data)")
    print("  - Interpolation: Uniform grid via interpolation (enables comparison)")
    print("  - Duplicate handling: Rename, keep_first, keep_last, or error")
    print("\n")
