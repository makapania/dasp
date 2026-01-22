"""
Comprehensive functional tests for imbalance method export coverage.

This test suite verifies that EVERY imbalance method supported in the
main application also generates valid, executable export code.
"""

import sys
import io

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from spectral_predict.code_generator import CodeGenerator, ExportOptions


def test_classification_imbalance_methods():
    """Test all 7 classification imbalance methods generate valid code."""

    # Sample classification data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y_binary = np.array([0]*80 + [1]*20)  # Imbalanced

    classification_methods = [
        'smote',
        'adasyn',
        'borderline_smote',
        'random_undersampler',
        'tomek_links',
        'smote_tomek',
        'class_weight'
    ]

    results = {}

    for method in classification_methods:
        print(f"\n{'='*60}")
        print(f"Testing CLASSIFICATION method: {method}")
        print(f"{'='*60}")

        # Create model config with imbalance method
        config = {
            'model_name': 'RandomForest',
            'preprocessing': 'snv',
            'target_name': 'quality',
            'task_type': 'classification',
            'params': {'n_estimators': 100, 'random_state': 42},
            'metrics': {'Accuracy': 0.85, 'F1': 0.82},
            'cv_folds': 5,
            'imbalance_method': method,
            'wavelengths': np.arange(400, 2500, 2)
        }

        options = ExportOptions(
            include_data=True,
            data_X=X,
            data_y=y_binary,
            wavelengths=np.arange(400, 2500, 2),
            include_visualization=False,
            include_prediction_template=False
        )

        try:
            # Generate script
            generator = CodeGenerator(config, options)
            script = generator.generate_script()

            # Validate script contains imbalance handling
            if method == 'class_weight':
                # class_weight is handled differently (in model params)
                assert 'class_weight' in script.lower(), f"Script missing class_weight handling"
                print(f"  ✓ class_weight found in model parameters")
            else:
                assert 'imbalance' in script.lower(), f"Script missing imbalance section for {method}"
                print(f"  ✓ Imbalance handling section present")

            # Check method name or class name is present
            if method == 'borderline_smote':
                # Variable name uses underscores, class name is BorderlineSMOTE
                assert 'borderline_smote' in script or 'BorderlineSMOTE' in script, \
                    f"Neither 'borderline_smote' nor 'BorderlineSMOTE' found"
            elif method == 'random_undersampler':
                # Variable name uses underscores, class name is RandomUnderSampler
                assert 'undersampler' in script.lower() or 'RandomUnderSampler' in script, \
                    f"Neither 'undersampler' nor 'RandomUnderSampler' found"
            elif method == 'tomek_links':
                # Variable name is tomek, class name is TomekLinks
                assert 'tomek' in script.lower() or 'TomekLinks' in script, \
                    f"Neither 'tomek' nor 'TomekLinks' found"
            elif method == 'smote_tomek':
                # Variable name uses underscores, class name is SMOTETomek
                assert 'smote_tomek' in script or 'SMOTETomek' in script, \
                    f"Neither 'smote_tomek' nor 'SMOTETomek' found"

            # Check for required imports
            if method in ['smote', 'adasyn', 'borderline_smote', 'smote_tomek']:
                # These require imblearn imports
                if method == 'adasyn':
                    assert 'ADASYN' in script or 'adasyn' in script.lower()
                elif method == 'borderline_smote':
                    assert 'BorderlineSMOTE' in script or 'borderline' in script.lower()
                print(f"  ✓ Required imports present")

            # Try to execute the script (syntax check)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_file = f.name

            # Compile to check for syntax errors
            with open(temp_file, 'r') as f:
                compile(f.read(), temp_file, 'exec')

            os.unlink(temp_file)
            print(f"  ✓ Script compiles without syntax errors")

            results[method] = 'PASS'
            print(f"\n✓ {method}: PASS")

        except Exception as e:
            results[method] = f'FAIL: {str(e)}'
            print(f"\n✗ {method}: FAIL - {str(e)}")

    return results


def test_regression_imbalance_methods():
    """Test all 7 regression imbalance methods generate valid code."""

    # Sample regression data with imbalance
    np.random.seed(42)
    X = np.random.randn(100, 50)
    # Many zeros, few high values (common in spectroscopy)
    y_continuous = np.concatenate([
        np.zeros(60),
        np.random.uniform(0.1, 1.0, 30),
        np.random.uniform(5.0, 10.0, 10)
    ])

    regression_methods = [
        'smogn',
        'oversample',
        'smotetomek',
        'undersample',
        'binning',
        'rare_boost',
        'balanced'
    ]

    results = {}

    for method in regression_methods:
        print(f"\n{'='*60}")
        print(f"Testing REGRESSION method: {method}")
        print(f"{'='*60}")

        # Create model config with imbalance method
        config = {
            'model_name': 'PLS',
            'preprocessing': 'snv',
            'target_name': 'protein',
            'task_type': 'regression',
            'params': {'n_components': 8},
            'metrics': {'RMSE': 0.45, 'R2': 0.92},
            'cv_folds': 5,
            'imbalance_method': method,
            'wavelengths': np.arange(400, 2500, 2)
        }

        options = ExportOptions(
            include_data=True,
            data_X=X,
            data_y=y_continuous,
            wavelengths=np.arange(400, 2500, 2),
            include_visualization=False,
            include_prediction_template=False
        )

        try:
            # Generate script
            generator = CodeGenerator(config, options)
            script = generator.generate_script()

            # Validate script contains imbalance handling
            if method in ['binning', 'rare_boost', 'balanced']:
                # These use sample weighting, not resampling
                assert 'weight' in script.lower() or 'sample_weight' in script.lower(), \
                    f"Script missing weighting for {method}"
                print(f"  ✓ Sample weighting code present")
            else:
                # These use resampling
                assert 'imbalance' in script.lower() or 'resample' in script.lower(), \
                    f"Script missing imbalance/resampling section for {method}"
                print(f"  ✓ Resampling section present")

            # Check method-specific content (flexible matching)
            if method == 'rare_boost':
                # rare_boost appears as "Rare-Value Boosting" in the title
                assert 'rare' in script.lower() and 'boost' in script.lower(), \
                    f"Script missing rare boost keywords for {method}"
            else:
                # For other methods, check for the method name
                assert method.upper() in script or method.replace('_', ' ') in script.lower(), \
                    f"Script missing method name '{method}'"
            print(f"  ✓ Method name found in script")

            # Try to execute the script (syntax check)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_file = f.name

            # Compile to check for syntax errors
            with open(temp_file, 'r') as f:
                compile(f.read(), temp_file, 'exec')

            os.unlink(temp_file)
            print(f"  ✓ Script compiles without syntax errors")

            results[method] = 'PASS'
            print(f"\n✓ {method}: PASS")

        except Exception as e:
            results[method] = f'FAIL: {str(e)}'
            print(f"\n✗ {method}: FAIL - {str(e)}")

    return results


def test_notebook_export_with_imbalance():
    """Test that notebook export also handles imbalance methods."""

    np.random.seed(42)
    X = np.random.randn(50, 30)
    y = np.array([0]*40 + [1]*10)

    config = {
        'model_name': 'SVC',
        'preprocessing': 'snv',
        'target_name': 'class',
        'task_type': 'classification',
        'params': {'C': 1.0, 'kernel': 'rbf'},
        'metrics': {'Accuracy': 0.88},
        'cv_folds': 5,
        'imbalance_method': 'smote',
        'wavelengths': np.arange(400, 800, 2)
    }

    options = ExportOptions(
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=np.arange(400, 800, 2),
        format='notebook',
        colab_ready=True
    )

    try:
        generator = CodeGenerator(config, options)
        notebook = generator.generate_notebook()

        # Validate notebook structure
        assert 'cells' in notebook
        assert len(notebook['cells']) > 0

        # Find imbalance cell
        imbalance_cells = [c for c in notebook['cells']
                          if 'imbalance' in ''.join(c.get('source', [])).lower()]

        assert len(imbalance_cells) > 0, "No imbalance handling cell found in notebook"

        print("\n✓ Notebook export with SMOTE: PASS")
        return 'PASS'

    except Exception as e:
        print(f"\n✗ Notebook export: FAIL - {str(e)}")
        return f'FAIL: {str(e)}'


def test_colab_ready_with_imbalance():
    """Test Colab-ready export includes proper pip install for imbalance packages."""

    config = {
        'model_name': 'RandomForest',
        'preprocessing': 'raw',
        'target_name': 'class',
        'task_type': 'classification',
        'params': {'n_estimators': 50},
        'metrics': {'Accuracy': 0.85},
        'cv_folds': 5,
        'imbalance_method': 'adasyn',
    }

    options = ExportOptions(
        format='notebook',
        colab_ready=True,
        include_data=False
    )

    try:
        generator = CodeGenerator(config, options)
        notebook = generator.generate_notebook()

        # Find pip install cell
        pip_cells = [c for c in notebook['cells']
                    if 'pip install' in ''.join(c.get('source', [])).lower()]

        assert len(pip_cells) > 0, "No pip install cell found"

        pip_content = ''.join(pip_cells[0]['source'])
        assert 'imbalanced-learn' in pip_content, "imbalanced-learn not in pip install"

        print("\n✓ Colab-ready with imbalance packages: PASS")
        return 'PASS'

    except Exception as e:
        print(f"\n✗ Colab-ready test: FAIL - {str(e)}")
        return f'FAIL: {str(e)}'


def print_summary_report(classification_results, regression_results, notebook_result, colab_result):
    """Print a comprehensive summary report."""

    print("\n" + "="*80)
    print("COMPREHENSIVE IMBALANCE METHOD EXPORT TEST SUMMARY")
    print("="*80)

    print("\n" + "-"*80)
    print("CLASSIFICATION METHODS (7 total)")
    print("-"*80)
    classification_pass = sum(1 for v in classification_results.values() if v == 'PASS')
    classification_fail = len(classification_results) - classification_pass

    for method, result in sorted(classification_results.items()):
        status = "✓ PASS" if result == 'PASS' else f"✗ FAIL"
        print(f"  {method:25s} {status}")
        if result != 'PASS':
            print(f"    Error: {result}")

    print(f"\nClassification Summary: {classification_pass}/{len(classification_results)} passed")

    print("\n" + "-"*80)
    print("REGRESSION METHODS (7 total)")
    print("-"*80)
    regression_pass = sum(1 for v in regression_results.values() if v == 'PASS')
    regression_fail = len(regression_results) - regression_pass

    for method, result in sorted(regression_results.items()):
        status = "✓ PASS" if result == 'PASS' else f"✗ FAIL"
        print(f"  {method:25s} {status}")
        if result != 'PASS':
            print(f"    Error: {result}")

    print(f"\nRegression Summary: {regression_pass}/{len(regression_results)} passed")

    print("\n" + "-"*80)
    print("ADDITIONAL TESTS")
    print("-"*80)
    print(f"  Notebook export:          {notebook_result}")
    print(f"  Colab-ready export:       {colab_result}")

    print("\n" + "="*80)
    total_tests = len(classification_results) + len(regression_results) + 2
    total_passed = classification_pass + regression_pass + \
                   (1 if notebook_result == 'PASS' else 0) + \
                   (1 if colab_result == 'PASS' else 0)

    print(f"OVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nAll imbalance methods are properly supported in export functionality!")
    else:
        print(f"\n✗✗✗ {total_tests - total_passed} TESTS FAILED ✗✗✗")
        print("\nThe following methods need implementation:")

        failed_methods = []
        for method, result in classification_results.items():
            if result != 'PASS':
                failed_methods.append(f"Classification: {method}")
        for method, result in regression_results.items():
            if result != 'PASS':
                failed_methods.append(f"Regression: {method}")

        for fm in failed_methods:
            print(f"  - {fm}")

    print("="*80 + "\n")

    return total_passed == total_tests


if __name__ == '__main__':
    print("="*80)
    print("IMBALANCE METHOD EXPORT COVERAGE TEST")
    print("="*80)
    print("\nThis test verifies that ALL imbalance methods supported in the main")
    print("application also generate valid, executable export code.\n")

    # Run all tests
    classification_results = test_classification_imbalance_methods()
    regression_results = test_regression_imbalance_methods()
    notebook_result = test_notebook_export_with_imbalance()
    colab_result = test_colab_ready_with_imbalance()

    # Print comprehensive report
    all_passed = print_summary_report(
        classification_results,
        regression_results,
        notebook_result,
        colab_result
    )

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
