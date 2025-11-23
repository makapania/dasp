"""
Demonstration of Reproducibility Toggle Mechanism

This script demonstrates how to toggle between fast (exploration) mode
and reproducible (publication) mode for scientific research.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_predict.search import run_search
from spectral_predict.reproducibility import (
    check_reproducibility_status,
    restore_default_threads,
    reproducible_context
)

def create_sample_data():
    """Create sample spectral data for testing."""
    np.random.seed(99999)
    n_samples = 40
    n_wavelengths = 80

    # Create synthetic spectral data
    X = np.random.randn(n_samples, n_wavelengths)
    y = np.random.randn(n_samples)

    # Convert to DataFrame with wavelength column names
    wavelengths = [f"{400 + i*5}" for i in range(n_wavelengths)]
    X_df = pd.DataFrame(X, columns=wavelengths)
    y_series = pd.Series(y, name='target')

    return X_df, y_series


def demo_toggle_mechanism():
    """
    Demonstrate how reproducibility settings toggle on and off.
    """
    print("="*90)
    print("REPRODUCIBILITY TOGGLE DEMONSTRATION")
    print("="*90)
    print()

    X, y = create_sample_data()

    # =========================================================================
    # PART 1: Fast Mode (Exploration)
    # =========================================================================
    print("\n" + "="*90)
    print("PART 1: FAST MODE (for exploration)")
    print("="*90)
    print("Running with reproducible=False (default)...")
    print()

    results_fast1, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=False,  # Fast mode
        tier='quick'
    )

    print(f"\n✓ Fast mode completed. Top model R² = {results_fast1.iloc[0]['R2']:.6f}")
    print(f"  Settings after fast run:")
    status = check_reproducibility_status()
    print(f"  BLAS threads: {status['blas_threads_env']}")

    # =========================================================================
    # PART 2: Reproducible Mode (Publications)
    # =========================================================================
    print("\n" + "="*90)
    print("PART 2: REPRODUCIBLE MODE (for publications)")
    print("="*90)
    print("Running with reproducible=True...")
    print()

    results_repro1, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,  # Reproducible mode
        random_state=42,
        tier='quick'
    )

    print(f"\n✓ Reproducible mode completed. Top model R² = {results_repro1.iloc[0]['R2']:.6f}")
    print(f"  Settings after reproducible run (should be RESTORED):")
    status = check_reproducibility_status()
    print(f"  BLAS threads: {status['blas_threads_env']}")

    # =========================================================================
    # PART 3: Verify Toggle - Run Reproducible Again
    # =========================================================================
    print("\n" + "="*90)
    print("PART 3: VERIFY REPRODUCIBILITY - Second reproducible run")
    print("="*90)
    print("Running again with reproducible=True and same random_state...")
    print()

    results_repro2, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=True,  # Reproducible mode
        random_state=42,  # Same seed
        tier='quick'
    )

    print(f"\n✓ Second reproducible run completed. Top model R² = {results_repro2.iloc[0]['R2']:.6f}")

    # =========================================================================
    # PART 4: Verify Results are Identical
    # =========================================================================
    print("\n" + "="*90)
    print("PART 4: VERIFICATION OF REPRODUCIBILITY")
    print("="*90)

    # Compare the two reproducible runs
    try:
        pd.testing.assert_frame_equal(results_repro1, results_repro2)
        print("✅ SUCCESS: Both reproducible runs produced IDENTICAL results!")
        print(f"   Run 1 top R² = {results_repro1.iloc[0]['R2']:.10f}")
        print(f"   Run 2 top R² = {results_repro2.iloc[0]['R2']:.10f}")
        print(f"   Difference   = {abs(results_repro1.iloc[0]['R2'] - results_repro2.iloc[0]['R2']):.2e}")
    except AssertionError as e:
        print(f"❌ ERROR: Results differ between runs!")
        print(f"   This should not happen in reproducible mode.")
        print(f"   Run 1 top R² = {results_repro1.iloc[0]['R2']:.10f}")
        print(f"   Run 2 top R² = {results_repro2.iloc[0]['R2']:.10f}")
        raise

    # =========================================================================
    # PART 5: Back to Fast Mode
    # =========================================================================
    print("\n" + "="*90)
    print("PART 5: TOGGLE BACK TO FAST MODE")
    print("="*90)
    print("Running again with reproducible=False...")
    print()

    results_fast2, _ = run_search(
        X, y,
        task_type='regression',
        folds=3,
        max_n_components=3,
        models_to_test=['PLS'],
        preprocessing_methods={'raw': True, 'snv': False, 'deriv': False, 'deriv_snv': False},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        reproducible=False,  # Back to fast mode
        tier='quick'
    )

    print(f"\n✓ Fast mode completed. Top model R² = {results_fast2.iloc[0]['R2']:.6f}")
    print(f"  Settings after fast run (should use all cores):")
    status = check_reproducibility_status()
    print(f"  BLAS threads: {status['blas_threads_env']}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*90)
    print("SUMMARY: HOW TO TOGGLE REPRODUCIBILITY")
    print("="*90)
    print()
    print("1️⃣  FAST MODE (Exploration)")
    print("   results, _ = run_search(X, y, ..., reproducible=False)")
    print("   → Uses all CPU cores")
    print("   → Fast but results may vary slightly between runs")
    print("   → Good for exploring different models and preprocessing")
    print()
    print("2️⃣  REPRODUCIBLE MODE (Publications)")
    print("   results, _ = run_search(X, y, ..., reproducible=True, random_state=42)")
    print("   → Uses 1 thread (3-5x slower)")
    print("   → Bit-identical results across runs")
    print("   → Required for scientific papers and regulatory submissions")
    print()
    print("3️⃣  AUTOMATIC TOGGLE")
    print("   → Settings automatically restore after each run")
    print("   → No manual cleanup needed")
    print("   → Switch between modes as often as you want")
    print()
    print("="*90)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*90)


def demo_context_manager():
    """
    Demonstrate the reproducible_context() context manager.
    """
    print("\n" + "="*90)
    print("BONUS: Using reproducible_context() directly")
    print("="*90)
    print()
    print("You can also use the context manager for fine-grained control:")
    print()
    print("```python")
    print("with reproducible_context(n_threads=1, random_state=42):")
    print("    # Your code runs with BLAS=1 and seeded RNG")
    print("    results = some_analysis(data)")
    print("# Settings automatically restored here")
    print("```")
    print()
    print("This is useful if you want to control reproducibility")
    print("at a more granular level than run_search().")
    print("="*90)


if __name__ == "__main__":
    # Run the main demonstration
    demo_toggle_mechanism()

    # Show the context manager approach
    demo_context_manager()

    print("\n" + "="*90)
    print("For more information, see:")
    print("  - src/spectral_predict/reproducibility.py")
    print("  - tests/test_reproducibility.py")
    print("="*90)
