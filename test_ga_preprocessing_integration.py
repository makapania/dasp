"""
Test script to verify GA Preprocessing integration.

This script checks:
1. GA preprocessing module imports correctly
2. search.py recognizes HAS_GA_PREPROCESS
3. GUI has all required variables
4. Integration is complete
"""

import sys
import os
import numpy as np

# Fix Windows console encoding for checkmarks
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

def test_module_import():
    """Test that ga_preprocessing module imports."""
    try:
        from src.spectral_predict.ga_preprocessing import optimize_preprocessing
        print("✓ GA preprocessing module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ga_preprocessing: {e}")
        return False

def test_search_integration():
    """Test that search.py recognizes GA preprocessing."""
    try:
        from src.spectral_predict import search
        if hasattr(search, 'HAS_GA_PREPROCESS'):
            print(f"✓ search.py HAS_GA_PREPROCESS = {search.HAS_GA_PREPROCESS}")
            return search.HAS_GA_PREPROCESS
        else:
            print("✗ search.py missing HAS_GA_PREPROCESS flag")
            return False
    except Exception as e:
        print(f"✗ Error checking search.py: {e}")
        return False

def test_gui_variables():
    """Test that GUI has all required variables."""
    try:
        # Import just enough to check the __init__ creates variables
        # (can't actually run GUI in headless test)
        import tkinter as tk
        from spectral_predict_gui_optimized import SpectralPredictApp

        # Create a test root
        root = tk.Tk()
        root.withdraw()  # Hide window

        # Initialize GUI (this will set up all variables)
        gui = SpectralPredictApp(root)

        # Check for required variables
        required_vars = [
            'enable_ga_preprocessing',
            'ga_preprocess_population',
            'ga_preprocess_generations',
            'ga_preprocess_cv_folds'
        ]

        missing = []
        for var in required_vars:
            if not hasattr(gui, var):
                missing.append(var)

        if missing:
            print(f"✗ GUI missing variables: {missing}")
            root.destroy()
            return False
        else:
            print("✓ GUI has all required GA preprocessing variables")
            # Check defaults
            print(f"  - enable_ga_preprocessing: {gui.enable_ga_preprocessing.get()}")
            print(f"  - ga_preprocess_population: {gui.ga_preprocess_population.get()}")
            print(f"  - ga_preprocess_generations: {gui.ga_preprocess_generations.get()}")
            print(f"  - ga_preprocess_cv_folds: {gui.ga_preprocess_cv_folds.get()}")
            root.destroy()
            return True

    except Exception as e:
        print(f"✗ Error testing GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_functional_integration():
    """Test that GA preprocessing actually works with a tiny dataset."""
    try:
        from src.spectral_predict.ga_preprocessing import optimize_preprocessing

        # Create tiny synthetic dataset
        np.random.seed(42)
        X = np.random.randn(20, 50)  # 20 samples, 50 features
        y = np.random.randn(20)  # Random targets

        # Run GA with minimal settings
        print("\n  Running GA preprocessing with minimal settings...")
        result = optimize_preprocessing(
            X, y,
            population_size=8,
            n_generations=3,
            cv_folds=3,
            task_type='regression',
            verbose=0
        )

        # Check result structure
        required_keys = ['best_genes', 'best_name', 'best_transform', 'best_rmsecv', 'best_config', 'history']
        missing_keys = [k for k in required_keys if k not in result]

        if missing_keys:
            print(f"✗ GA result missing keys: {missing_keys}")
            return False

        print(f"✓ GA preprocessing functional test passed")
        print(f"  - Best config: {result['best_config']}")
        print(f"  - Best RMSECV: {result['best_rmsecv']:.4f}")
        print(f"  - Generations completed: {len(result['history'])}")

        return True

    except Exception as e:
        print(f"✗ Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("GA PREPROCESSING INTEGRATION TEST")
    print("=" * 70)
    print()

    tests = [
        ("Module Import", test_module_import),
        ("search.py Integration", test_search_integration),
        ("GUI Variables", test_gui_variables),
        ("Functional Test", test_functional_integration),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print()
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! GA Preprocessing is fully integrated.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
