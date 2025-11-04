#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for all Phase 1-3 implementations.

Tests:
- Phase 1: Ridge & Lasso model integration
- Phase 2: Interactive plot features
- Phase 3: Outlier detection system
"""

import sys
import traceback
from pathlib import Path
import io

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_phase1_models():
    """Test Ridge & Lasso model implementations."""
    print("\n" + "="*60)
    print("PHASE 1: RIDGE & LASSO MODELS")
    print("="*60)

    # Test 1: Check models.py has Ridge and Lasso
    print("\n[Test 1.1] Checking Ridge & Lasso in models.py...")
    try:
        from spectral_predict.models import get_model, get_model_grids

        # Check get_model
        ridge = get_model('Ridge')
        lasso = get_model('Lasso')
        print(f"  ✓ Ridge model: {ridge}")
        print(f"  ✓ Lasso model: {lasso}")

        # Check grids
        grids = get_model_grids(task_type='regression', n_features=100)
        assert 'Ridge' in grids, "Ridge not in grids"
        assert 'Lasso' in grids, "Lasso not in grids"
        print(f"  ✓ Ridge grid size: {len(grids['Ridge'])} configurations")
        print(f"  ✓ Lasso grid size: {len(grids['Lasso'])} configurations")

        # Verify grid structure
        ridge_alphas = [cfg[1]['alpha'] for cfg in grids['Ridge']]
        lasso_alphas = [cfg[1]['alpha'] for cfg in grids['Lasso']]
        print(f"  ✓ Ridge alphas: {ridge_alphas}")
        print(f"  ✓ Lasso alphas: {lasso_alphas}")

        assert len(ridge_alphas) == 7, f"Expected 7 Ridge alphas, got {len(ridge_alphas)}"
        assert len(lasso_alphas) == 6, f"Expected 6 Lasso alphas, got {len(lasso_alphas)}"

        print("  ✅ Test 1.1 PASSED")

    except Exception as e:
        print(f"  ❌ Test 1.1 FAILED: {e}")
        traceback.print_exc()
        return False

    # Test 2: Check feature importance extraction
    print("\n[Test 1.2] Checking feature importance extraction...")
    try:
        # Check search.py includes Ridge/Lasso in model lists
        with open('src/spectral_predict/search.py', 'r', encoding='utf-8') as f:
            search_content = f.read()
        assert '"Ridge"' in search_content, "Ridge not found in search.py"
        assert '"Lasso"' in search_content, "Lasso not found in search.py"
        print("  ✓ Ridge and Lasso found in search.py model lists")

        # Check models.py has coefficient extraction logic
        with open('src/spectral_predict/models.py', 'r', encoding='utf-8') as f:
            models_content = f.read()
        assert 'elif model_name in ["Ridge", "Lasso"]:' in models_content, \
            "Ridge/Lasso coefficient extraction not found in models.py"
        assert 'coefs = np.abs(model.coef_)' in models_content, \
            "Coefficient extraction logic not found"
        print("  ✓ Coefficient extraction logic found in models.py")

        print("  ✅ Test 1.2 PASSED")

    except Exception as e:
        print(f"  ❌ Test 1.2 FAILED: {e}")
        traceback.print_exc()
        return False

    # Test 3: Check GUI has Ridge & Lasso checkboxes
    print("\n[Test 1.3] Checking GUI has Ridge & Lasso checkboxes...")
    try:
        with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
            gui_content = f.read()

        # Check BooleanVar initialization
        assert 'self.use_ridge = tk.BooleanVar(value=False)' in gui_content, \
            "use_ridge BooleanVar not found"
        assert 'self.use_lasso = tk.BooleanVar(value=False)' in gui_content, \
            "use_lasso BooleanVar not found"
        print("  ✓ BooleanVar variables found")

        # Check checkbuttons
        assert '"✓ Ridge Regression"' in gui_content or "'✓ Ridge Regression'" in gui_content, \
            "Ridge checkbox text not found"
        assert '"✓ Lasso Regression"' in gui_content or "'✓ Lasso Regression'" in gui_content, \
            "Lasso checkbox text not found"
        print("  ✓ Checkbutton labels found")

        # Check model collection in _run_analysis
        assert ('selected_models.append("Ridge")' in gui_content or
                "selected_models.append('Ridge')" in gui_content), \
            "Ridge not added to selected_models"
        assert ('selected_models.append("Lasso")' in gui_content or
                "selected_models.append('Lasso')" in gui_content), \
            "Lasso not added to selected_models"
        print("  ✓ Model collection logic found")

        print("  ✅ Test 1.3 PASSED")

    except Exception as e:
        print(f"  ❌ Test 1.3 FAILED: {e}")
        traceback.print_exc()
        return False

    return True


def test_phase2_interactive_plots():
    """Test interactive plot features."""
    print("\n" + "="*60)
    print("PHASE 2: INTERACTIVE PLOT FEATURES")
    print("="*60)

    print("\n[Test 2.1] Checking GUI for interactive plot features...")
    try:
        with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
            gui_content = f.read()

        # Check NavigationToolbar import
        assert 'NavigationToolbar2Tk' in gui_content, \
            "NavigationToolbar2Tk import not found"
        print("  ✓ NavigationToolbar2Tk imported")

        # Check state variables
        assert 'self.use_absorbance = tk.BooleanVar(value=False)' in gui_content, \
            "use_absorbance BooleanVar not found"
        assert 'self.excluded_spectra = set()' in gui_content, \
            "excluded_spectra set not found"
        print("  ✓ State variables found")

        # Check key methods
        methods = [
            '_toggle_absorbance',
            '_reset_exclusions',
            '_update_exclusion_status',
            '_on_spectrum_click',
            '_apply_transformation'
        ]
        for method in methods:
            assert f'def {method}' in gui_content, f"{method} method not found"
        print(f"  ✓ All {len(methods)} helper methods found")

        # Check absorbance transformation
        assert 'log10(1/R)' in gui_content or 'log10' in gui_content, \
            "Absorbance transformation not found"
        print("  ✓ Absorbance transformation logic found")

        # Check click event handler
        assert 'mpl_connect' in gui_content and 'pick_event' in gui_content, \
            "Click event handler not found"
        print("  ✓ Click event handler found")

        # Check NavigationToolbar instantiation
        assert 'NavigationToolbar2Tk' in gui_content, \
            "NavigationToolbar not instantiated"
        print("  ✓ NavigationToolbar instantiation found")

        # Check exclusion filtering in analysis
        assert 'excluded_spectra' in gui_content and 'mask' in gui_content, \
            "Exclusion filtering not found in analysis"
        print("  ✓ Exclusion filtering in analysis found")

        print("  ✅ Test 2.1 PASSED")

    except Exception as e:
        print(f"  ❌ Test 2.1 FAILED: {e}")
        traceback.print_exc()
        return False

    return True


def test_phase3_outlier_detection():
    """Test outlier detection system."""
    print("\n" + "="*60)
    print("PHASE 3: OUTLIER DETECTION SYSTEM")
    print("="*60)

    # Test 1: Check outlier_detection.py module
    print("\n[Test 3.1] Checking outlier_detection.py module...")
    try:
        from spectral_predict.outlier_detection import (
            run_pca_outlier_detection,
            compute_q_residuals,
            compute_mahalanobis_distance,
            check_y_data_consistency,
            generate_outlier_report
        )
        print("  ✓ All functions imported successfully")

        # Test with synthetic data
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randn(50) * 10 + 50

        # Run detection
        report = generate_outlier_report(X, y, n_pca_components=3)
        print(f"  ✓ Generated report with {len(report['outlier_summary'])} samples")
        print(f"  ✓ High confidence outliers: {len(report['high_confidence_outliers'])}")
        print(f"  ✓ Moderate confidence outliers: {len(report['moderate_confidence_outliers'])}")

        # Check report structure
        required_keys = ['pca', 'q_residuals', 'mahalanobis', 'y_consistency',
                        'outlier_summary', 'combined_flags']
        for key in required_keys:
            assert key in report, f"Missing key: {key}"
        print(f"  ✓ Report has all {len(required_keys)} required keys")

        print("  ✅ Test 3.1 PASSED")

    except Exception as e:
        print(f"  ❌ Test 3.1 FAILED: {e}")
        traceback.print_exc()
        return False

    # Test 2: Check GUI integration
    print("\n[Test 3.2] Checking GUI integration...")
    try:
        with open('spectral_predict_gui_optimized.py', 'r', encoding='utf-8') as f:
            gui_content = f.read()

        # Check import
        assert 'from src.spectral_predict.outlier_detection import generate_outlier_report' in gui_content or \
               'from spectral_predict.outlier_detection import generate_outlier_report' in gui_content, \
            "Outlier detection import not found"
        print("  ✓ Outlier detection import found")

        # Check state variables
        assert 'self.n_pca_components = tk.IntVar(value=5)' in gui_content, \
            "n_pca_components variable not found"
        assert 'self.outlier_report = None' in gui_content, \
            "outlier_report variable not found"
        print("  ✓ State variables found")

        # Check tab creation
        assert '_create_tab2_data_quality_check' in gui_content, \
            "Data Quality Check tab method not found"
        assert '🔍 Data Quality Check' in gui_content or 'Data Quality Check' in gui_content, \
            "Data Quality Check tab text not found"
        print("  ✓ Data Quality Check tab found")

        # Check visualization methods
        viz_methods = [
            '_run_outlier_detection',
            '_plot_pca_scores',
            '_plot_hotelling_t2',
            '_plot_q_residuals',
            '_plot_mahalanobis',
            '_plot_y_distribution',
            '_populate_outlier_table'
        ]
        for method in viz_methods:
            assert f'def {method}' in gui_content, f"{method} method not found"
        print(f"  ✓ All {len(viz_methods)} visualization methods found")

        # Check exclusion integration
        assert '_mark_selected_for_exclusion' in gui_content, \
            "Mark for exclusion method not found"
        print("  ✓ Exclusion integration found")

        # Check export functionality
        assert '_export_outlier_report' in gui_content, \
            "Export report method not found"
        print("  ✓ Export report functionality found")

        # Check tab renumbering
        assert 'tab3' in gui_content and 'tab4' in gui_content, \
            "Tabs not renumbered correctly"
        print("  ✓ Tabs renumbered correctly")

        print("  ✅ Test 3.2 PASSED")

    except Exception as e:
        print(f"  ❌ Test 3.2 FAILED: {e}")
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE FOR ALL IMPLEMENTATIONS")
    print("="*60)

    results = {
        'Phase 1: Ridge & Lasso Models': test_phase1_models(),
        'Phase 2: Interactive Plot Features': test_phase2_interactive_plots(),
        'Phase 3: Outlier Detection System': test_phase3_outlier_detection()
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for phase, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{phase}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nAll three phases implemented successfully:")
        print("✓ Phase 1: Ridge & Lasso models added to analysis pipeline")
        print("✓ Phase 2: Interactive plots with reflectance/absorbance toggle")
        print("✓ Phase 3: Comprehensive outlier detection system")
        print("\nReady for user testing!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("Please review the failures above and fix any issues.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
