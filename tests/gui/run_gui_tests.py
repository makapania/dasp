#!/usr/bin/env python
"""
Standalone GUI test runner for Spectral Predict V1.

Provides multiple ways to run GUI tests:
1. Via pytest (recommended)
2. Interactive mode with step-by-step prompts
3. Quick smoke test

Usage:
    python tests/gui/run_gui_tests.py                # Run all tests headless
    python tests/gui/run_gui_tests.py --visible      # Run with visible window
    python tests/gui/run_gui_tests.py --interactive  # Interactive mode
    python tests/gui/run_gui_tests.py --smoke        # Quick smoke test only
    python tests/gui/run_gui_tests.py --data-path C:/mydata  # Custom data
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_pytest(args):
    """Run tests via pytest."""
    import pytest

    pytest_args = [
        str(PROJECT_ROOT / "tests" / "gui"),
        "-v",
    ]

    if args.visible:
        pytest_args.append("--visible")

    if args.data_path:
        pytest_args.extend(["--data-path", args.data_path])

    if args.smoke:
        pytest_args.extend(["-m", "smoke"])
    elif args.slow:
        pytest_args.extend(["-m", "slow"])
    else:
        pytest_args.extend(["-m", "gui"])

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    print(f"\nRunning: pytest {' '.join(pytest_args)}\n")
    return pytest.main(pytest_args)


def run_interactive(args):
    """Run interactive step-by-step tests."""
    import tkinter as tk
    import pandas as pd

    print("\n" + "=" * 60)
    print("INTERACTIVE GUI TEST MODE")
    print("=" * 60)
    print("\nThis will walk through key functionality step by step.")
    print("Press Enter after each step to continue, or 'q' to quit.\n")

    # Import app
    from spectral_predict_gui_optimized import SpectralPredictApp
    from tests.gui.harness import GUITestHarness

    # Create app
    root = tk.Tk()
    root.title("GUI Test - Interactive Mode")

    if not args.visible:
        root.withdraw()
        print("(Running in headless mode - use --visible to see window)")

    app = SpectralPredictApp(root)
    if hasattr(app, '_on_tier_changed'):
        app._on_tier_changed()

    harness = GUITestHarness(app, visible=args.visible)

    def step(description):
        """Prompt user to continue."""
        response = input(f"\n[STEP] {description}\n       Press Enter to continue (q to quit): ")
        if response.lower() == 'q':
            print("\nTest aborted by user.")
            root.destroy()
            sys.exit(0)

    def check(condition, message):
        """Check a condition and print result."""
        if condition:
            print(f"  [OK] PASS: {message}")
            return True
        else:
            print(f"  [X]  FAIL: {message}")
            return False

    try:
        # Step 1: Check app created
        step("Checking app initialization...")
        check(app is not None, "App created")
        check(hasattr(app, 'notebook'), "Notebook widget exists")
        check(app.X is None, "X is None initially")
        check(app.y is None, "y is None initially")

        # Step 2: Load data
        step("Loading example data...")
        data_path = Path(args.data_path) if args.data_path else PROJECT_ROOT / "example"

        # Load reference data
        csv_path = data_path / "BoneCollagen.csv"
        if not csv_path.exists():
            print(f"  ! ERROR: CSV file not found: {csv_path}")
            return 1

        ref_df = pd.read_csv(csv_path)
        print(f"  Loaded reference: {len(ref_df)} samples")

        # Load spectral data
        from spectral_predict.io import read_asd_dir
        asd_files = list(data_path.glob("*.asd")) + list(data_path.glob("*.sig"))

        if not asd_files:
            print(f"  ! ERROR: No ASD files found in {data_path}")
            return 1

        try:
            result = read_asd_dir(str(data_path))
            # read_asd_dir returns (DataFrame, metadata_dict)
            X = result[0] if isinstance(result, tuple) else result
            # Adjust index to match reference format
            new_index = [idx.replace("Spectrum", "Spectrum ") if idx.startswith("Spectrum") else idx
                         for idx in X.index]
            X.index = new_index
            print(f"  Loaded spectra: {X.shape[0]} samples x {X.shape[1]} wavelengths")
        except Exception as e:
            print(f"  ! ERROR loading spectra: {e}")
            return 1

        # Match with reference
        ref_df['File Number'] = ref_df['File Number'].str.strip()
        X.index = X.index.str.strip()
        common_ids = X.index.intersection(ref_df.set_index('File Number').index)
        X = X.loc[common_ids]
        ref_subset = ref_df.set_index('File Number').loc[common_ids]
        y = ref_subset['%Collagen']

        # Set in app
        app.X = X
        app.X_original = X.copy()
        app.y = y
        app.task_type.set("regression")
        harness.wait_for_idle()

        check(app.X is not None, f"X loaded: {app.X.shape}")
        check(app.y is not None, f"y loaded: {len(app.y)} values")
        check(len(app.X) == len(app.y), "X and y have same length")

        # Step 3: Configure analysis
        step("Configuring quick analysis (PLS, SNV, 3-fold CV)...")
        harness.configure_quick_analysis(
            models=['PLS'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=3
        )
        check(app.use_pls.get(), "PLS enabled")
        check(app.use_snv.get(), "SNV enabled")
        check(app.folds.get() == 3, "CV folds = 3")

        # Step 4: Run analysis
        step("Running analysis (this may take a minute)...")
        print("  Running...")

        # Use run_analysis_direct to bypass GUI threading issues
        success = harness.run_analysis_direct(
            models=['PLS'],
            preprocessing=['Raw', 'SNV'],
            cv_folds=3
        )

        check(success, "Analysis completed")
        check(app.results_df is not None, "Results generated")

        if app.results_df is not None:
            print(f"  Results: {len(app.results_df)} rows")
            print(f"  Columns: {list(app.results_df.columns)}")

            # Find R2 column
            r2_col = None
            for col in app.results_df.columns:
                if 'r2' in col.lower():
                    r2_col = col
                    break

            if r2_col:
                best_r2 = app.results_df[r2_col].max()
                print(f"  Best R2: {best_r2:.4f}")
                check(0 <= best_r2 <= 1, f"R2 in valid range")

        # Step 5: Validate results
        step("Validating results...")
        validation = harness.validate_results()
        check(validation['has_results'], "Has results")
        check(validation['r2_valid'], "R2 values valid")

        if validation['errors']:
            print(f"  Warnings: {validation['errors']}")

        print("\n" + "=" * 60)
        print("INTERACTIVE TEST COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\n! ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass

    return 0


def run_smoke_test(args):
    """Run quick smoke test."""
    print("\n" + "=" * 60)
    print("QUICK SMOKE TEST")
    print("=" * 60 + "\n")

    import tkinter as tk

    try:
        # Test 1: App creates
        print("[1/4] Creating app...")
        from spectral_predict_gui_optimized import SpectralPredictApp

        root = tk.Tk()
        root.withdraw()
        app = SpectralPredictApp(root)
        print("      [OK] App created successfully")

        # Test 2: Core modules import
        print("[2/4] Importing core modules...")
        from spectral_predict.search import run_search
        from spectral_predict.models import get_model
        from spectral_predict.preprocess import build_preprocessing_pipeline
        print("      [OK] Core modules imported")

        # Test 3: Example data exists
        print("[3/4] Checking example data...")
        data_path = Path(args.data_path) if args.data_path else PROJECT_ROOT / "example"
        csv_path = data_path / "BoneCollagen.csv"
        asd_count = len(list(data_path.glob("*.asd")))

        if csv_path.exists():
            print(f"      [OK] CSV file exists: {csv_path.name}")
        else:
            print(f"      [X]  CSV file missing: {csv_path}")

        print(f"      [OK] Found {asd_count} ASD files")

        # Test 4: Test harness works
        print("[4/4] Testing harness...")
        from tests.gui.harness import GUITestHarness
        harness = GUITestHarness(app)
        harness.wait_for_idle(0.5)
        print("      [OK] Harness works")

        root.quit()
        root.destroy()

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\n[X] SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Spectral Predict GUI tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_gui_tests.py                  # Run all GUI tests
    python run_gui_tests.py --visible        # Show GUI window
    python run_gui_tests.py --smoke          # Quick smoke test
    python run_gui_tests.py --interactive    # Step-by-step mode
    python run_gui_tests.py -k "regression"  # Run tests matching keyword
        """
    )

    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show GUI window during tests"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to test data folder (default: example/)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive step-by-step mode"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test only"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        help="Only run tests matching keyword"
    )

    args = parser.parse_args()

    if args.interactive:
        return run_interactive(args)
    elif args.smoke:
        return run_smoke_test(args)
    else:
        return run_pytest(args)


if __name__ == "__main__":
    sys.exit(main())
