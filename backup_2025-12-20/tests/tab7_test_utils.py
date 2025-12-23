"""
Test utilities for Tab 7 Model Development testing.

Provides helper functions and fixtures for testing Tab 7 functionality:
- Data loading helpers
- Analysis execution helpers
- GUI interaction helpers
- Validation helpers
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.search import run_search
from spectral_predict.model_io import save_model, load_model


def load_quick_start_data():
    """
    Load quick_start example data.

    Returns
    -------
    X : pd.DataFrame
        Spectral data
    y : pd.Series
        Target values
    ref : pd.DataFrame
        Reference dataframe

    Raises
    ------
    FileNotFoundError
        If example data is not found
    """
    example_dir = Path(__file__).parent.parent / "example"
    ref_file = example_dir / "reference.csv"

    if not ref_file.exists():
        raise FileNotFoundError(f"Quick start data not found at {example_dir}")

    # Load reference
    ref = pd.read_csv(ref_file)

    # Load spectral files
    from spectral_predict.io import read_asd

    spectral_files = sorted(example_dir.glob("Spectrum*.asd"))
    if not spectral_files:
        raise FileNotFoundError("No spectral files found in example directory")

    # Load all spectra
    spectra_list = []
    sample_ids = []
    for f in spectral_files:
        wl, refl = read_asd(str(f))
        spectra_list.append(refl)
        sample_ids.append(f.stem)

    # Create DataFrame
    X = pd.DataFrame(spectra_list, columns=[f"{w:.1f}" for w in wl], index=sample_ids)

    # Get target from reference (assume first numeric column)
    numeric_cols = ref.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in reference file")

    y = ref[numeric_cols[0]]

    return X, y, ref


def create_minimal_synthetic_data(n_samples=30, n_wavelengths=100, seed=42):
    """
    Create minimal synthetic spectral data for fast testing.

    Parameters
    ----------
    n_samples : int
        Number of samples (default: 30)
    n_wavelengths : int
        Number of wavelengths (default: 100)
    seed : int
        Random seed

    Returns
    -------
    X : pd.DataFrame
        Spectral data
    y : pd.Series
        Target values
    """
    np.random.seed(seed)

    # Generate wavelengths (1500-2500 nm range)
    wavelengths = np.linspace(1500, 2500, n_wavelengths)

    # Generate realistic spectral data
    X = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        # Base spectrum with smooth variation
        base = 0.5 + 0.2 * np.sin(np.linspace(0, 3*np.pi, n_wavelengths))
        noise = np.random.normal(0, 0.05, n_wavelengths)
        X[i] = base + noise

    # Create target correlated with first few wavelengths
    signal = X[:, :5].mean(axis=1)
    y = 10 + 5 * signal + np.random.normal(0, 0.3, n_samples)

    # Convert to DataFrame/Series
    X_df = pd.DataFrame(X, columns=[f"{wl:.1f}" for wl in wavelengths])
    y_series = pd.Series(y, name='target')

    return X_df, y_series


def run_minimal_analysis(X, y, models=None, preprocessing=None, n_folds=3, verbose=False):
    """
    Run a minimal analysis for fast testing.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data
    y : pd.Series
        Target values
    models : list, optional
        Models to test (default: ['PLS', 'Ridge'])
    preprocessing : list, optional
        Preprocessing methods (default: ['raw', 'sg1'])
    n_folds : int
        Number of CV folds (default: 3)
    verbose : bool
        Print progress (default: False)

    Returns
    -------
    results : pd.DataFrame
        Analysis results
    """
    if models is None:
        models = ['PLS', 'Ridge']

    if preprocessing is None:
        preprocessing = ['raw', 'sg1']

    # Run analysis
    results = run_search(
        X=X,
        y=y,
        task_type='regression',
        models=models,
        preprocessing_methods=preprocessing,
        n_folds=n_folds,
        subset_methods=['full'],  # No subsets for speed
        max_n_components=5,
        verbose=verbose
    )

    return results


def run_analysis_with_subsets(X, y, subset_sizes=[50], n_folds=3, verbose=False):
    """
    Run analysis with wavelength subsets for testing subset loading.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data
    y : pd.Series
        Target values
    subset_sizes : list
        Subset sizes to test (default: [50])
    n_folds : int
        Number of CV folds
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        Analysis results with subset models
    """
    results = run_search(
        X=X,
        y=y,
        task_type='regression',
        models=['PLS', 'Ridge'],
        preprocessing_methods=['raw', 'sg1'],
        n_folds=n_folds,
        subset_methods=['top', 'forward'],
        subset_sizes=subset_sizes,
        max_n_components=5,
        verbose=verbose
    )

    return results


def wait_for_analysis_completion(gui, timeout=120):
    """
    Wait for analysis to complete in GUI.

    Parameters
    ----------
    gui : SpectralPredictApp
        GUI instance
    timeout : int
        Maximum wait time in seconds

    Returns
    -------
    bool
        True if analysis completed, False if timeout

    Raises
    ------
    TimeoutError
        If analysis doesn't complete within timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if analysis is complete
        # (Implementation depends on GUI state tracking)
        if hasattr(gui, 'analysis_complete') and gui.analysis_complete:
            return True

        # Check if results are available
        if gui.results_df is not None and len(gui.results_df) > 0:
            return True

        # Wait a bit before checking again
        time.sleep(0.5)

    raise TimeoutError(f"Analysis did not complete within {timeout} seconds")


def get_top_result(results_df, model_type=None, preprocessing=None):
    """
    Get the top-ranked result from analysis results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Analysis results
    model_type : str, optional
        Filter by model type (e.g., 'PLS', 'Ridge')
    preprocessing : str, optional
        Filter by preprocessing (e.g., 'raw', 'sg1')

    Returns
    -------
    result : pd.Series
        Top result configuration
    """
    filtered = results_df.copy()

    if model_type is not None:
        filtered = filtered[filtered['Model'] == model_type]

    if preprocessing is not None:
        filtered = filtered[filtered['Preprocess'] == preprocessing]

    if len(filtered) == 0:
        raise ValueError(f"No results found matching criteria: model={model_type}, preprocess={preprocessing}")

    # Sort by R2 (descending) and get top
    filtered = filtered.sort_values('R2', ascending=False)
    return filtered.iloc[0]


def get_subset_result(results_df, subset_tag):
    """
    Get a result with specific subset tag.

    Parameters
    ----------
    results_df : pd.DataFrame
        Analysis results
    subset_tag : str
        Subset tag to filter by (e.g., 'top50', 'forward50')

    Returns
    -------
    result : pd.Series
        Result configuration with specified subset
    """
    subset_results = results_df[results_df['SubsetTag'] == subset_tag]

    if len(subset_results) == 0:
        raise ValueError(f"No results found with SubsetTag='{subset_tag}'")

    # Return top result
    subset_results = subset_results.sort_values('R2', ascending=False)
    return subset_results.iloc[0]


def validate_result_fields(result, expected_fields=None):
    """
    Validate that a result has all required fields.

    Parameters
    ----------
    result : pd.Series
        Result configuration to validate
    expected_fields : list, optional
        List of expected field names

    Returns
    -------
    bool
        True if all fields present, False otherwise

    Raises
    ------
    AssertionError
        If required fields are missing
    """
    if expected_fields is None:
        # Default required fields
        expected_fields = [
            'Model', 'Rank', 'Preprocess', 'n_vars', 'R2', 'RMSE',
            'SubsetTag', 'Window'
        ]

    missing_fields = [f for f in expected_fields if f not in result.index]

    if missing_fields:
        raise AssertionError(f"Missing required fields: {missing_fields}")

    return True


def validate_subset_result_fields(result):
    """
    Validate that a subset result has all required fields including 'all_vars'.

    Parameters
    ----------
    result : pd.Series
        Subset result configuration

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    AssertionError
        If validation fails
    """
    # First check basic fields
    validate_result_fields(result)

    # Check subset-specific fields
    assert 'all_vars' in result.index, "Missing 'all_vars' field for subset model"
    assert not pd.isna(result['all_vars']), "'all_vars' field is NaN"
    assert result['all_vars'] != 'N/A', "'all_vars' field is N/A"

    # Validate n_vars matches all_vars count
    n_vars = int(result['n_vars'])
    all_vars_str = str(result['all_vars'])
    all_vars_list = [w.strip() for w in all_vars_str.split(',') if w.strip()]

    assert len(all_vars_list) == n_vars, \
        f"Wavelength count mismatch: n_vars={n_vars} but all_vars has {len(all_vars_list)} wavelengths"

    return True


def extract_hyperparameters(result):
    """
    Extract hyperparameters from a result configuration.

    Parameters
    ----------
    result : pd.Series
        Result configuration

    Returns
    -------
    params : dict
        Extracted hyperparameters
    """
    params = {}
    model_name = result['Model']

    if model_name == 'PLS':
        if 'LVs' in result.index and not pd.isna(result['LVs']):
            params['n_components'] = int(result['LVs'])

    elif model_name in ['Ridge', 'Lasso']:
        if 'Alpha' in result.index and not pd.isna(result['Alpha']):
            params['alpha'] = float(result['Alpha'])

    elif model_name == 'RandomForest':
        if 'n_estimators' in result.index and not pd.isna(result['n_estimators']):
            params['n_estimators'] = int(result['n_estimators'])
        if 'max_depth' in result.index and not pd.isna(result['max_depth']):
            params['max_depth'] = int(result['max_depth'])
        if 'max_features' in result.index and not pd.isna(result['max_features']):
            params['max_features'] = result['max_features']

    elif model_name == 'MLP':
        if 'LR_init' in result.index and not pd.isna(result['LR_init']):
            params['learning_rate_init'] = float(result['LR_init'])
        if 'Hidden' in result.index and not pd.isna(result['Hidden']):
            params['hidden_layer_sizes'] = result['Hidden']

    elif model_name == 'NeuralBoosted':
        if 'n_estimators' in result.index and not pd.isna(result['n_estimators']):
            params['n_estimators'] = int(result['n_estimators'])
        if 'LearningRate' in result.index and not pd.isna(result['LearningRate']):
            params['learning_rate'] = float(result['LearningRate'])
        if 'HiddenSize' in result.index and not pd.isna(result['HiddenSize']):
            params['hidden_layer_size'] = int(result['HiddenSize'])
        if 'Activation' in result.index and not pd.isna(result['Activation']):
            params['activation'] = result['Activation']

    return params


def compare_r2_values(original_r2, reproduced_r2, tolerance=0.001):
    """
    Compare two R² values and return difference.

    Parameters
    ----------
    original_r2 : float
        Original R² from Results tab
    reproduced_r2 : float
        Reproduced R² from Tab 7
    tolerance : float
        Maximum allowed difference (default: 0.001)

    Returns
    -------
    diff : float
        Absolute difference

    Raises
    ------
    AssertionError
        If difference exceeds tolerance
    """
    diff = abs(original_r2 - reproduced_r2)

    assert diff < tolerance, \
        f"R² mismatch exceeds tolerance: original={original_r2:.6f}, reproduced={reproduced_r2:.6f}, " \
        f"diff={diff:.6f} (tolerance={tolerance})"

    return diff


def simulate_gui_load_model(gui, result_config):
    """
    Simulate loading a model from Results tab into Tab 7.

    This is a test helper that calls the same method as double-clicking in the GUI.

    Parameters
    ----------
    gui : SpectralPredictApp
        GUI instance
    result_config : pd.Series or dict
        Result configuration to load

    Returns
    -------
    success : bool
        True if loading succeeded
    """
    if isinstance(result_config, pd.Series):
        result_config = result_config.to_dict()

    try:
        gui._load_model_to_tab7(result_config)
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False


def verify_wavelength_loading(gui, expected_wavelengths):
    """
    Verify that wavelengths were loaded correctly into Tab 7.

    Parameters
    ----------
    gui : SpectralPredictApp
        GUI instance
    expected_wavelengths : list
        Expected wavelength list

    Returns
    -------
    bool
        True if wavelengths match

    Raises
    ------
    AssertionError
        If wavelengths don't match
    """
    # Get wavelength spec from GUI
    wl_spec_text = gui.refine_wl_spec.get('1.0', 'end-1c')

    # Parse wavelengths
    available_wl = gui.X_original.columns.astype(float).values
    parsed_wl = gui._parse_wavelength_spec(wl_spec_text, available_wl)

    # Compare
    assert len(parsed_wl) == len(expected_wavelengths), \
        f"Wavelength count mismatch: expected {len(expected_wavelengths)}, got {len(parsed_wl)}"

    # Check that all expected wavelengths are present
    expected_set = set(expected_wavelengths)
    parsed_set = set(parsed_wl)

    missing = expected_set - parsed_set
    extra = parsed_set - expected_set

    if missing:
        raise AssertionError(f"Missing wavelengths: {sorted(missing)[:10]}...")
    if extra:
        raise AssertionError(f"Extra wavelengths: {sorted(extra)[:10]}...")

    return True


def create_test_model_file(output_path, model_name='PLS', task_type='regression'):
    """
    Create a test .dasp model file for testing model loading.

    Parameters
    ----------
    output_path : str or Path
        Output path for .dasp file
    model_name : str
        Model type (default: 'PLS')
    task_type : str
        Task type (default: 'regression')

    Returns
    -------
    model_path : Path
        Path to created model file
    """
    from spectral_predict.models import get_model

    # Create synthetic data
    X, y = create_minimal_synthetic_data()

    # Train model
    model = get_model(model_name, task_type=task_type, n_components=5)
    model.fit(X.values, y.values)

    # Create metadata
    metadata = {
        'model_name': model_name,
        'task_type': task_type,
        'preprocessing': 'raw',
        'n_vars': X.shape[1],
        'wavelengths': X.columns.tolist(),
        'performance': {
            'R2': 0.85,
            'RMSE': 1.25
        }
    }

    # Save model
    save_model(
        model=model,
        preprocessor=None,
        metadata=metadata,
        filepath=str(output_path)
    )

    return Path(output_path)
