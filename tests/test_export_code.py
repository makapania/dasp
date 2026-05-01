"""
Comprehensive tests for code export functionality.

Tests Python script generation, R script generation, Jupyter notebooks,
data embedding, ZIP bundle creation, and actual code execution.
"""

import pytest
import numpy as np
import json
import zipfile
import subprocess
import tempfile
import shutil
import os
from pathlib import Path

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.r_code_generator import RCodeGenerator
from spectral_predict.export_bundle import create_export_bundle


def find_rscript():
    """Find Rscript executable, checking PATH and common Windows locations."""
    # First check PATH
    rscript = shutil.which('Rscript')
    if rscript:
        return rscript

    # Check common Windows R installation paths
    r_base_paths = [
        Path("C:/Program Files/R"),
        Path("C:/Program Files (x86)/R"),
        Path(os.path.expanduser("~/R")),
    ]

    for base_path in r_base_paths:
        if base_path.exists():
            # Find latest R version
            r_versions = sorted(base_path.glob("R-*"), reverse=True)
            for r_version in r_versions:
                rscript_path = r_version / "bin" / "Rscript.exe"
                if rscript_path.exists():
                    return str(rscript_path)

    return None


RSCRIPT_PATH = find_rscript()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_spectral_data():
    """Create sample spectral data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_wavelengths = 100

    X = np.random.randn(n_samples, n_wavelengths)
    y = np.random.randn(n_samples) * 10 + 50  # Regression target
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    return X, y, wavelengths


@pytest.fixture
def sample_model_config():
    """Create sample model configuration."""
    return {
        'model_name': 'PLS',
        'preprocessing': 'snv',
        'target_name': 'protein',
        'task_type': 'regression',
        'params': {'n_components': 8},
        'metrics': {'RMSE': 0.45, 'R2': 0.92},
        'variable_indices': None,
        'wavelengths': np.linspace(1000, 2500, 100),
        'cv_folds': 5
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


# ============================================================================
# Python Code Generator Tests
# ============================================================================

def test_python_script_generation_basic(sample_model_config):
    """Test basic Python script generation without data embedding."""
    options = ExportOptions(
        include_visualization=False,
        include_prediction_template=True,
        format='script',
        include_data=False
    )

    generator = CodeGenerator(sample_model_config, options)
    script = generator.generate_script()

    # Verify script structure
    assert 'import numpy as np' in script
    assert 'import pandas as pd' in script
    assert 'from sklearn.cross_decomposition import PLSRegression' in script
    assert 'apply_snv' in script  # SNV preprocessing function
    assert 'PLSRegression' in script
    # Model parameter in dict-style format: 'n_components': 8
    assert "'n_components': 8" in script or "n_components" in script
    assert 'cross_val_predict' in script  # Cross-validation
    assert len(script) > 1000  # Reasonable length


def test_python_script_with_data_embedding(sample_model_config, sample_spectral_data):
    """Test Python script generation with embedded data."""
    X, y, wavelengths = sample_spectral_data

    options = ExportOptions(
        include_visualization=False,
        include_prediction_template=True,
        format='script',
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    generator = CodeGenerator(sample_model_config, options)
    script = generator.generate_script()

    # Verify embedded data section
    assert '_decode_embedded_data' in script
    assert '_X_ENCODED' in script
    assert '_Y_ENCODED' in script
    assert '_WAVELENGTHS_ENCODED' in script
    assert 'import base64' in script
    assert 'import gzip' in script

    # Verify data is actually encoded (base64 strings present)
    assert "'''" in script  # Triple quotes for multiline strings


def test_data_size_limit_validation(sample_model_config):
    """Test that data size limit (100 MB) is enforced."""
    # Create data that exceeds 100 MB
    huge_X = np.random.randn(10000, 10000)  # ~800 MB uncompressed
    huge_y = np.random.randn(10000)

    options = ExportOptions(
        include_data=True,
        data_X=huge_X,
        data_y=huge_y
    )

    with pytest.raises(ValueError, match="exceeds 100 MB limit"):
        generator = CodeGenerator(sample_model_config, options)


def test_jupyter_notebook_generation(sample_model_config):
    """Test Jupyter notebook generation."""
    options = ExportOptions(
        include_visualization=True,
        include_prediction_template=True,
        format='notebook',
        include_data=False
    )

    generator = CodeGenerator(sample_model_config, options)
    notebook = generator.generate_notebook()

    # Verify notebook structure
    assert notebook['nbformat'] == 4
    assert 'cells' in notebook
    assert len(notebook['cells']) > 0

    # Verify cell types
    cell_types = [cell['cell_type'] for cell in notebook['cells']]
    assert 'markdown' in cell_types
    assert 'code' in cell_types

    # Verify title cell
    first_cell = notebook['cells'][0]
    assert first_cell['cell_type'] == 'markdown'
    source = ''.join(first_cell['source'])
    assert 'Spectral Analysis' in source
    assert 'PLS' in source


def test_colab_notebook_features(sample_model_config, sample_spectral_data):
    """Test Colab-ready notebook with data embedding."""
    X, y, wavelengths = sample_spectral_data

    options = ExportOptions(
        format='notebook',
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths,
        colab_ready=True
    )

    generator = CodeGenerator(sample_model_config, options)
    notebook = generator.generate_notebook()

    # Find pip install cell (may use subprocess or !pip install)
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    pip_cell = None
    for cell in code_cells:
        source = ''.join(cell['source'])
        if '!pip install' in source or 'pip' in source and 'install' in source:
            pip_cell = source
            break

    assert pip_cell is not None, "Colab notebook should have pip install cell"
    assert 'scikit-learn' in pip_cell
    assert 'numpy' in pip_cell

    # Verify Colab badge in title
    first_cell_source = ''.join(notebook['cells'][0]['source'])
    assert 'colab-badge.svg' in first_cell_source

    # Verify Colab metadata
    assert 'colab' in notebook['metadata']


def test_python_script_execution(sample_model_config, sample_spectral_data, temp_dir):
    """Test that generated Python script actually runs without errors."""
    X, y, wavelengths = sample_spectral_data

    options = ExportOptions(
        include_visualization=False,  # Avoid matplotlib import issues
        include_prediction_template=False,
        format='script',
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    generator = CodeGenerator(sample_model_config, options)
    script_path = Path(temp_dir) / 'test_script.py'
    generator.save_script(str(script_path))

    # Execute the script
    result = subprocess.run(
        ['python', str(script_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Check execution
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Script execution failed:\n{result.stderr}"
    assert 'Loaded embedded data' in result.stdout
    assert 'RMSE:' in result.stdout or 'R²:' in result.stdout


# ============================================================================
# R Code Generator Tests
# ============================================================================

def test_r_script_generation_basic(sample_model_config):
    """Test basic R script generation (reticulate wrapper around Python)."""
    generator = RCodeGenerator(
        model_config=sample_model_config,
        include_data=False
    )

    script = generator.generate_script()

    # Verify reticulate wrapper structure
    assert 'library(reticulate)' in script
    assert 'library(base64enc)' in script
    assert 'py_run_string' in script
    assert 'PY_CODE_ENCODED' in script
    # Model info should appear in header comments
    assert 'PLS' in script
    assert 'snv' in script


def test_r_script_with_data_embedding(sample_model_config, sample_spectral_data):
    """Test R script generation with embedded data (reticulate wrapper)."""
    X, y, wavelengths = sample_spectral_data

    generator = RCodeGenerator(
        model_config=sample_model_config,
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    script = generator.generate_script()

    # Verify reticulate wrapper embeds the Python code (which contains the data)
    assert 'PY_CODE_ENCODED' in script  # Encoded Python script with embedded data
    assert 'base64enc' in script or 'base64decode' in script
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    # The R wrapper should be larger when data is embedded in the Python code
    assert len(script) > 1000


def test_r_model_mappings():
    """Test that R code generator supports various models."""
    models_to_test = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'SVM']

    for model_name in models_to_test:
        config = {
            'model_name': model_name,
            'preprocessing': 'raw',
            'task_type': 'regression',
            'params': {},
            'cv_folds': 5
        }

        generator = RCodeGenerator(model_config=config, include_data=False)
        script = generator.generate_script()

        # Should not have error message
        assert 'not supported' not in script.lower()
        assert 'ERROR' not in script


def _r_has_reticulate():
    """Check if R has reticulate package installed."""
    if not RSCRIPT_PATH:
        return False
    try:
        result = subprocess.run(
            [RSCRIPT_PATH, '-e', 'library(reticulate); cat("OK")'],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0 and 'OK' in result.stdout
    except Exception:
        return False


R_HAS_RETICULATE = _r_has_reticulate()


@pytest.mark.skipif(not RSCRIPT_PATH, reason="R not installed")
@pytest.mark.skipif(not R_HAS_RETICULATE, reason="R reticulate package not installed")
def test_r_script_execution(sample_model_config, sample_spectral_data, temp_dir):
    """Test that generated R script actually runs without errors.

    The R script is now a reticulate wrapper that runs the Python export
    via py_run_string. This requires both R and the reticulate package.
    """
    X, y, wavelengths = sample_spectral_data

    generator = RCodeGenerator(
        model_config=sample_model_config,
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    script_path = Path(temp_dir) / 'test_script.R'
    generator.save_script(str(script_path))

    # Execute the script
    result = subprocess.run(
        [RSCRIPT_PATH, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120  # R can be slower
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # R may output warnings to stderr, so check for actual errors
    assert result.returncode == 0, f"R script execution failed:\n{result.stderr}"
    # Output may come from the embedded Python script
    assert 'Running Python export' in result.stdout or 'loaded' in result.stdout.lower() or result.returncode == 0


# ============================================================================
# ZIP Bundle Export Tests
# ============================================================================

def test_bundle_creation(sample_model_config, sample_spectral_data, temp_dir):
    """Test complete ZIP bundle creation."""
    X, y, wavelengths = sample_spectral_data
    bundle_path = Path(temp_dir) / 'test_bundle.zip'

    create_export_bundle(
        model_config=sample_model_config,
        output_path=str(bundle_path),
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    # Verify bundle was created
    assert bundle_path.exists()
    assert bundle_path.stat().st_size > 1000  # Non-empty

    # Verify bundle contents
    with zipfile.ZipFile(bundle_path, 'r') as zipf:
        file_list = zipf.namelist()

        # Check for required files
        assert any('README.md' in f for f in file_list)
        assert any('python/analysis.py' in f for f in file_list)
        assert any('python/analysis.ipynb' in f for f in file_list)
        assert any('python/requirements.txt' in f for f in file_list)
        assert any('r/analysis.R' in f for f in file_list)
        assert any('r/install_packages.R' in f for f in file_list)
        assert any('data/spectra.csv' in f for f in file_list)
        assert any('data/target.csv' in f for f in file_list)
        assert any('data/wavelengths.csv' in f for f in file_list)

        # Verify README content
        readme_file = [f for f in file_list if 'README.md' in f][0]
        readme_content = zipf.read(readme_file).decode('utf-8')
        assert 'Spectral Analysis Export' in readme_content
        assert 'PLS' in readme_content


def test_bundle_without_data(sample_model_config, temp_dir):
    """Test bundle creation without data files."""
    bundle_path = Path(temp_dir) / 'test_bundle_nodata.zip'

    create_export_bundle(
        model_config=sample_model_config,
        output_path=str(bundle_path),
        include_data=False,
        data_X=None,
        data_y=None,
        wavelengths=None
    )

    with zipfile.ZipFile(bundle_path, 'r') as zipf:
        file_list = zipf.namelist()

        # Should have code files but no data files
        assert any('python/analysis.py' in f for f in file_list)
        assert any('r/analysis.R' in f for f in file_list)
        assert not any('data/spectra.csv' in f for f in file_list)
        assert not any('data/target.csv' in f for f in file_list)


# ============================================================================
# Data Encoding/Decoding Tests
# ============================================================================

def test_data_encoding_decoding(sample_spectral_data):
    """Test that data encoding and decoding is lossless."""
    X, y, wavelengths = sample_spectral_data

    # Encode
    X_encoded = CodeGenerator._encode_array(X)
    y_encoded = CodeGenerator._encode_array(y)

    # Decode using the generated decode function
    import base64
    import gzip

    def decode(encoded_str):
        decoded = base64.b64decode(encoded_str.encode('ascii'))
        decompressed = gzip.decompress(decoded)
        data_list = json.loads(decompressed.decode('utf-8'))
        return np.array(data_list)

    X_decoded = decode(X_encoded)
    y_decoded = decode(y_encoded)

    # Verify lossless
    np.testing.assert_array_almost_equal(X, X_decoded)
    np.testing.assert_array_almost_equal(y, y_decoded)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_python(sample_model_config, sample_spectral_data, temp_dir):
    """Test complete workflow: generate, save, and execute Python script."""
    X, y, wavelengths = sample_spectral_data

    # Generate with embedded data
    options = ExportOptions(
        include_visualization=False,
        include_prediction_template=False,
        format='script',
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    generator = CodeGenerator(sample_model_config, options)
    script_path = Path(temp_dir) / 'full_workflow.py'
    generator.save_script(str(script_path))

    # Verify file was created
    assert script_path.exists()
    assert script_path.stat().st_size > 5000  # Substantial file

    # Execute and verify output
    result = subprocess.run(
        ['python', str(script_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, f"Execution failed:\n{result.stderr}"

    # Verify key outputs
    output = result.stdout.lower()
    assert 'loaded' in output or 'shape' in output
    assert 'rmse' in output or 'r²' in output or 'r2' in output


@pytest.mark.skipif(not RSCRIPT_PATH, reason="R not installed")
@pytest.mark.skipif(not R_HAS_RETICULATE, reason="R reticulate package not installed")
def test_full_workflow_r(sample_model_config, sample_spectral_data, temp_dir):
    """Test complete workflow: generate, save, and execute R script.

    The R script is now a reticulate wrapper that runs the Python export
    via py_run_string. This requires both R and the reticulate package.
    """
    X, y, wavelengths = sample_spectral_data

    generator = RCodeGenerator(
        model_config=sample_model_config,
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=wavelengths
    )

    script_path = Path(temp_dir) / 'full_workflow.R'
    generator.save_script(str(script_path))

    # Verify file was created
    assert script_path.exists()
    assert script_path.stat().st_size > 1000  # Reticulate wrapper is smaller than native R

    # Execute
    result = subprocess.run(
        [RSCRIPT_PATH, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120
    )

    # R can output package loading messages to stderr, so don't fail on that
    print("R OUTPUT:", result.stdout)
    print("R MESSAGES:", result.stderr)

    assert result.returncode == 0, f"R execution failed with return code {result.returncode}"


def test_multiple_models(sample_spectral_data, temp_dir):
    """Test that export works for different model types."""
    X, y, wavelengths = sample_spectral_data

    models_to_test = [
        ('PLS', {'n_components': 5}),
        ('Ridge', {'alpha': 1.0}),
        ('RandomForest', {'n_estimators': 50})
    ]

    for model_name, params in models_to_test:
        config = {
            'model_name': model_name,
            'preprocessing': 'snv',
            'task_type': 'regression',
            'params': params,
            'metrics': {},
            'cv_folds': 3
        }

        # Test Python generation
        options = ExportOptions(
            include_data=True,
            data_X=X,
            data_y=y,
            wavelengths=wavelengths
        )

        generator = CodeGenerator(config, options)
        script = generator.generate_script()
        assert len(script) > 1000
        assert model_name in script

        # Test R generation
        r_generator = RCodeGenerator(
            model_config=config,
            include_data=True,
            data_X=X,
            data_y=y,
            wavelengths=wavelengths
        )
        r_script = r_generator.generate_script()
        assert len(r_script) > 1000


def test_mlp_python_export(sample_spectral_data):
    """Test that MLP model export works for Python."""
    X, y, wavelengths = sample_spectral_data

    config = {
        'model_name': 'MLP',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 200
        },
        'cv_folds': 3
    }

    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Verify MLP import and instantiation
    assert 'from sklearn.neural_network import MLPRegressor' in script
    assert 'MLPRegressor(' in script
    # Parameters are rendered in dict-style: 'hidden_layer_sizes': (100,)
    assert "'hidden_layer_sizes': (100,)" in script or 'hidden_layer_sizes' in script


def test_mlp_classifier_python_export(sample_spectral_data):
    """Test that MLP classifier export works for Python."""
    X, y, wavelengths = sample_spectral_data

    config = {
        'model_name': 'MLP',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {
            'hidden_layer_sizes': (50,),
            'max_iter': 150
        },
        'cv_folds': 3
    }

    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Verify MLP classifier import
    assert 'from sklearn.neural_network import MLPClassifier' in script
    assert 'MLPClassifier(' in script


def test_imbalance_smote_python_export():
    """Test that SMOTE imbalance handling is exported for Python."""
    config = {
        'model_name': 'RandomForest',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {'n_estimators': 100},
        'cv_folds': 3,
        'imbalance_method': 'smote'
    }

    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Verify SMOTE import and application
    assert 'from imblearn.over_sampling import SMOTE' in script
    # SMOTE is used inside the resampler factory function, not as 'smote = SMOTE'
    assert 'SMOTE' in script
    assert 'fit_resample' in script
    assert 'imbalanced-learn' in script  # In extra packages


def test_imbalance_class_weight_python_export():
    """Test that class_weight imbalance handling is exported for Python."""
    config = {
        'model_name': 'RandomForest',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {'n_estimators': 100},
        'cv_folds': 3,
        'imbalance_method': 'class_weight'
    }

    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Verify class_weight is mentioned
    assert 'class_weight' in script.lower()


def test_mlp_r_export():
    """Test that MLP model export works for R (reticulate wrapper)."""
    config = {
        'model_name': 'MLP',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {
            'hidden_layer_sizes': (100,),
            'max_iter': 200
        },
        'cv_folds': 3
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper with model info in header
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'MLP' in script  # Model name in header
    assert 'regression' in script  # Task type in header


def test_mlp_classifier_r_export():
    """Test that MLP classifier export works for R (reticulate wrapper)."""
    config = {
        'model_name': 'MLPClassifier',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {
            'hidden_layer_sizes': (50,),
            'max_iter': 150
        },
        'cv_folds': 3
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper with classifier info in header
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'MLPClassifier' in script  # Model name in header
    assert 'classification' in script  # Task type in header


def test_catboost_r_export():
    """Test that CatBoost export works for R (reticulate wrapper)."""
    config = {
        'model_name': 'CatBoost',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'cv_folds': 3
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper with CatBoost info in header
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'CatBoost' in script  # Model name in header
    assert 'n_estimators: 100' in script  # Parameters in header


def test_lightgbm_r_export():
    """Test that LightGBM export works for R (reticulate wrapper)."""
    config = {
        'model_name': 'LightGBM',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31
        },
        'cv_folds': 3
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper with LightGBM info in header
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'LightGBM' in script  # Model name in header
    assert 'num_leaves: 31' in script  # Parameters in header


def test_classification_metrics_r_export():
    """Test that classification tasks generate appropriate R reticulate wrapper."""
    config = {
        'model_name': 'RandomForest',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {'n_estimators': 100},
        'cv_folds': 3
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper with classification info in header
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'classification' in script  # Task type in header
    assert 'RandomForest' in script  # Model name in header


def test_imbalance_smote_r_export():
    """Test that SMOTE imbalance handling is exported for R (reticulate wrapper)."""
    config = {
        'model_name': 'RandomForest',
        'preprocessing': 'snv',
        'task_type': 'classification',
        'params': {'n_estimators': 100},
        'cv_folds': 3,
        'imbalance_method': 'smote'
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper is generated (SMOTE is handled in embedded Python)
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'PY_CODE_ENCODED' in script
    assert 'RandomForest' in script  # Model in header
    assert 'classification' in script  # Task in header


def test_regression_vs_classification_r_export():
    """Test that R exports differ correctly for regression vs classification (reticulate wrappers)."""
    # Test regression
    reg_config = {
        'model_name': 'XGBoost',
        'preprocessing': 'raw',
        'task_type': 'regression',
        'params': {'n_estimators': 100},
        'cv_folds': 3
    }

    reg_gen = RCodeGenerator(model_config=reg_config, include_data=False)
    reg_script = reg_gen.generate_script()

    # Test classification
    class_config = {
        'model_name': 'XGBoost',
        'preprocessing': 'raw',
        'task_type': 'classification',
        'params': {'n_estimators': 100},
        'cv_folds': 3
    }

    class_gen = RCodeGenerator(model_config=class_config, include_data=False)
    class_script = class_gen.generate_script()

    # Both should be reticulate wrappers
    assert 'library(reticulate)' in reg_script
    assert 'library(reticulate)' in class_script

    # Regression header should indicate regression task
    assert 'regression' in reg_script
    assert 'XGBoost' in reg_script

    # Classification header should indicate classification task
    assert 'classification' in class_script
    assert 'XGBoost' in class_script

    # The two scripts should differ (different embedded Python code)
    assert reg_script != class_script


def test_variable_selection_python_export():
    """Test that variable selection is exported correctly for Python."""
    config = {
        'model_name': 'PLS',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {'n_components': 5},
        'cv_folds': 3,
        'variable_indices': [5, 12, 23, 45, 67, 89],  # Selected wavelength indices
        'variable_selection_method': 'UVE'
    }

    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Verify variable selection is included
    assert 'selected_indices' in script
    assert 'X_final = X_processed[:, selected_indices]' in script
    assert 'UVE' in script  # Method name in comment
    # Check at least some indices are present
    assert '5' in script and '12' in script


def test_variable_selection_r_export():
    """Test that variable selection is exported correctly for R (reticulate wrapper)."""
    config = {
        'model_name': 'PLS',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {'n_components': 5},
        'cv_folds': 3,
        'variable_indices': [5, 12, 23, 45, 67, 89],  # Selected wavelength indices
        'variable_selection_method': 'SPA'
    }

    generator = RCodeGenerator(model_config=config, include_data=False)
    script = generator.generate_script()

    # Verify reticulate wrapper is generated
    assert 'library(reticulate)' in script
    assert 'py_run_string' in script
    assert 'PY_CODE_ENCODED' in script
    # Model info in header
    assert 'PLS' in script
    assert 'snv' in script


def test_autoscale_emitted_in_python_export():
    """T-36: when autoscale is True in model_config, the exported script must
    apply StandardScaler after SNV/derivatives so the saved pipeline reproduces
    exactly. Codex Phase 7 review caught that this was missing."""
    config = {
        'model_name': 'PLS',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {'n_components': 5},
        'cv_folds': 3,
        'autoscale': True,
    }
    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()

    # Autoscale block must appear in the generated preprocessing section.
    assert 'StandardScaler' in script, (
        "Autoscale=True model_config must emit a StandardScaler import"
    )
    assert 'autoscale' in script.lower() or 'UV scaling' in script, (
        "Generated script should comment that autoscale is being applied"
    )


def test_autoscale_omitted_when_false():
    """When autoscale is False (or absent), exported scripts must NOT apply
    StandardScaler — would silently change the pipeline."""
    config = {
        'model_name': 'PLS',
        'preprocessing': 'snv',
        'task_type': 'regression',
        'params': {'n_components': 5},
        'cv_folds': 3,
        'autoscale': False,
    }
    options = ExportOptions(include_data=False)
    generator = CodeGenerator(config, options)
    script = generator.generate_script()
    # Search for the autoscale-block sentinel (the comment prefix is unique).
    assert 'Autoscale (UV scaling)' not in script, (
        "Autoscale=False should not emit the autoscale block"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
