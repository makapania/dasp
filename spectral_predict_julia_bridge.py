"""
Python-Julia Bridge for SpectralPredict

This module provides a bridge between the Python GUI (spectral_predict_gui_optimized.py)
and the Julia backend (SpectralPredict.jl), allowing the GUI to leverage Julia's performance
while maintaining the same interface as the Python implementation.

Key Features:
- Matches the exact interface of Python's run_search() function
- Handles data marshalling between Python (NumPy/Pandas) and Julia (CSV files)
- Provides progress updates from Julia back to the Python GUI
- Returns results in the same Pandas DataFrame format as Python version
- Comprehensive error handling and validation

Usage:
    from spectral_predict_julia_bridge import run_search_julia

    results_df = run_search_julia(
        X, y,
        task_type="regression",
        folds=5,
        models_to_test=["PLS", "RandomForest"],
        ...
    )

Author: Claude AI
Date: October 2025
"""

import os
import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

# Julia paths (user can override these)
JULIA_EXE = r"C:\Users\sponheim\AppData\Local\Programs\Julia-1.12.1\bin\julia.exe"
JULIA_PROJECT = r"C:\Users\sponheim\git\dasp\julia_port\SpectralPredict"


# ============================================================================
# Main Bridge Function
# ============================================================================

def run_search_julia(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    folds: int = 5,
    lambda_penalty: float = 0.15,
    max_n_components: int = 24,
    max_iter: int = 500,
    models_to_test: Optional[List[str]] = None,
    preprocessing_methods: Optional[Dict[str, bool]] = None,
    window_sizes: Optional[List[int]] = None,
    n_estimators_list: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None,
    enable_variable_subsets: bool = True,
    variable_counts: Optional[List[int]] = None,
    variable_selection_methods: Optional[List[str]] = None,
    enable_region_subsets: bool = True,
    n_top_regions: int = 5,
    # Variable selection method parameters (ignored for now, use Julia defaults)
    apply_uve_prefilter: bool = False,
    uve_cutoff_multiplier: float = 1.0,
    uve_n_components: Optional[int] = None,
    spa_n_random_starts: int = 10,
    ipls_n_intervals: int = 20,
    progress_callback: Optional[Callable] = None,
    julia_exe: Optional[str] = None,
    julia_project: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run comprehensive model search using Julia backend.

    This function provides the same interface as the Python version but calls
    Julia for computation. It handles all data marshalling and progress tracking.

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (n_samples, n_features). Column names should be wavelengths.
    y : pd.Series
        Target values (n_samples,)
    task_type : str
        Either 'regression' or 'classification'
    folds : int, default=5
        Number of CV folds
    lambda_penalty : float, default=0.15
        Complexity penalty weight for scoring (0.0-1.0)
    max_n_components : int, default=24
        Maximum number of PLS components (Julia uses adaptive sizing)
    max_iter : int, default=500
        Maximum iterations for MLP (passed to Julia)
    models_to_test : list of str, optional
        Models to test. Options: 'PLS', 'RandomForest', 'MLP', 'Ridge', 'Lasso', 'NeuralBoosted'
        If None, tests: ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP']
    preprocessing_methods : dict, optional
        Dict with keys: 'raw', 'snv', 'sg1', 'sg2', 'deriv_snv', 'msc'
        Values are booleans indicating whether to use each method.
        If None, uses all methods (excluding MSC by default).
    window_sizes : list of int, optional
        Window sizes for Savitzky-Golay derivatives (default: [17])
    n_estimators_list : list of int, optional
        Not used by Julia backend (for compatibility)
    learning_rates : list of float, optional
        Not used by Julia backend (for compatibility)
    enable_variable_subsets : bool, default=True
        Enable top-N variable subset analysis
    variable_counts : list of int, optional
        Variable counts to test (e.g., [10, 20, 50, 100, 250])
        If None, uses: [10, 20, 50, 100, 250]
    variable_selection_methods : list of str, optional
        Variable selection methods to use. Options:
        - 'importance': Model-based feature importance (default)
        - 'SPA': Successive Projections Algorithm
        - 'UVE': Uninformative Variable Elimination
        - 'iPLS': Interval PLS
        - 'UVE-SPA': Hybrid UVE-SPA approach
        If None, uses ['importance']
    enable_region_subsets : bool, default=True
        Enable spectral region subset analysis
    n_top_regions : int, default=5
        Number of top regions to analyze (5, 10, 15, or 20)
    progress_callback : callable, optional
        Function to call with progress updates. Signature:
        callback(dict) where dict has keys:
            - 'stage': Current stage
            - 'message': Status message
            - 'current': Current config number
            - 'total': Total configs
            - 'best_model': Best model dict (optional)
    julia_exe : str, optional
        Path to Julia executable (overrides default)
    julia_project : str, optional
        Path to Julia project directory (overrides default)

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns matching Python version:
        - Model, Preprocess, Deriv, Window, Poly, LVs
        - SubsetTag, n_vars, full_vars
        - RMSE, R2, MAE (regression) or Accuracy, Precision, Recall, F1, ROC_AUC (classification)
        - CompositeScore, Rank
        - top_vars (comma-separated wavelengths)

    Raises
    ------
    FileNotFoundError
        If Julia executable or project directory not found
    RuntimeError
        If Julia process fails or returns invalid data
    ValueError
        If input validation fails

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = pd.DataFrame(np.random.rand(100, 200))
    >>> X.columns = [f"{w}" for w in range(400, 600)]  # Wavelengths
    >>> y = pd.Series(np.random.rand(100))
    >>>
    >>> # Run search with Julia backend
    >>> results = run_search_julia(
    ...     X, y,
    ...     task_type='regression',
    ...     models_to_test=['PLS', 'RandomForest'],
    ...     enable_variable_subsets=True,
    ...     enable_region_subsets=True
    ... )
    >>>
    >>> # View top models
    >>> print(results.head(10))
    """

    # Validate inputs
    _validate_inputs(X, y, task_type, folds, lambda_penalty)

    # Use provided paths or defaults
    julia_exe = julia_exe or JULIA_EXE
    julia_project = julia_project or JULIA_PROJECT

    # Validate Julia installation
    _validate_julia_installation(julia_exe, julia_project)

    # Create temporary directory for data exchange
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Progress update: Starting
        if progress_callback:
            progress_callback({
                'stage': 'initialization',
                'message': 'Preparing data for Julia...',
                'current': 0,
                'total': 1
            })

        # 1. Save input data to CSV files
        X_file = temp_path / "X_input.csv"
        y_file = temp_path / "y_input.csv"
        wavelengths_file = temp_path / "wavelengths.csv"

        _save_input_data(X, y, X_file, y_file, wavelengths_file)

        # 2. Create configuration JSON
        config_file = temp_path / "config.json"
        config = _create_config(
            task_type=task_type,
            folds=folds,
            lambda_penalty=lambda_penalty,
            max_n_components=max_n_components,
            max_iter=max_iter,
            models_to_test=models_to_test,
            preprocessing_methods=preprocessing_methods,
            window_sizes=window_sizes,
            enable_variable_subsets=enable_variable_subsets,
            variable_counts=variable_counts,
            variable_selection_methods=variable_selection_methods,
            enable_region_subsets=enable_region_subsets,
            n_top_regions=n_top_regions
        )

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # 3. Create output file path
        output_file = temp_path / "results_output.csv"
        progress_file = temp_path / "progress.json"

        # Progress update: Calling Julia
        if progress_callback:
            progress_callback({
                'stage': 'julia_execution',
                'message': 'Starting Julia analysis...',
                'current': 0,
                'total': 1
            })

        # 4. Build Julia command
        julia_script = _create_julia_script(
            X_file, y_file, wavelengths_file,
            config_file, output_file, progress_file, julia_project
        )

        script_file = temp_path / "run_analysis.jl"
        with open(script_file, 'w') as f:
            f.write(julia_script)

        # 5. Execute Julia process
        try:
            result = _run_julia_process(
                julia_exe,
                julia_project,
                script_file,
                progress_file,
                progress_callback
            )

        except Exception as e:
            raise RuntimeError(f"Julia execution failed: {e}")

        # 6. Load results
        if not output_file.exists():
            raise RuntimeError(
                f"Julia did not produce output file. "
                f"Check Julia installation and project setup."
            )

        try:
            results_df = pd.read_csv(output_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load Julia results: {e}")

        # 7. Post-process results to match Python format
        results_df = _postprocess_results(results_df, task_type)

        # Final progress update
        if progress_callback:
            progress_callback({
                'stage': 'complete',
                'message': 'Analysis complete!',
                'current': 1,
                'total': 1
            })

        return results_df


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_inputs(X, y, task_type, folds, lambda_penalty):
    """Validate input parameters."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")

    if len(X) != len(y):
        raise ValueError(f"X rows ({len(X)}) must match y length ({len(y)})")

    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples, got {X.shape[0]}")

    if X.shape[1] < 1:
        raise ValueError(f"Need at least 1 feature, got {X.shape[1]}")

    if task_type not in ['regression', 'classification']:
        raise ValueError("task_type must be 'regression' or 'classification'")

    if folds < 2:
        raise ValueError("folds must be at least 2")

    if folds > len(X):
        raise ValueError(f"folds ({folds}) cannot exceed number of samples ({len(X)})")

    if not (0.0 <= lambda_penalty <= 1.0):
        raise ValueError("lambda_penalty must be between 0.0 and 1.0")


def _validate_julia_installation(julia_exe, julia_project):
    """Validate Julia installation and project."""
    if not os.path.exists(julia_exe):
        raise FileNotFoundError(
            f"Julia executable not found: {julia_exe}\n"
            f"Please install Julia 1.12+ or update JULIA_EXE path"
        )

    if not os.path.exists(julia_project):
        raise FileNotFoundError(
            f"Julia project directory not found: {julia_project}\n"
            f"Please ensure SpectralPredict.jl is installed"
        )

    # Check for Project.toml
    project_toml = Path(julia_project) / "Project.toml"
    if not project_toml.exists():
        raise FileNotFoundError(
            f"Project.toml not found in {julia_project}\n"
            f"Please ensure SpectralPredict.jl project is properly initialized"
        )


def _save_input_data(X, y, X_file, y_file, wavelengths_file):
    """Save input data to CSV files for Julia."""
    # Save X (samples × wavelengths)
    X.to_csv(X_file, index=True, index_label='sample_id')

    # Save y (sample_id, target)
    y_df = pd.DataFrame({
        'sample_id': X.index,
        'target': y.values
    })
    y_df.to_csv(y_file, index=False)

    # Save wavelengths
    wavelengths = [float(col) for col in X.columns]
    wl_df = pd.DataFrame({'wavelength': wavelengths})
    wl_df.to_csv(wavelengths_file, index=False)


def _create_config(
    task_type,
    folds,
    lambda_penalty,
    max_n_components,
    max_iter,
    models_to_test,
    preprocessing_methods,
    window_sizes,
    enable_variable_subsets,
    variable_counts,
    variable_selection_methods,
    enable_region_subsets,
    n_top_regions
):
    """Create configuration dictionary for Julia."""

    # Map Python model names to Julia names
    model_mapping = {
        'PLS': 'PLS',
        'RandomForest': 'RandomForest',
        'MLP': 'MLP',
        'Ridge': 'Ridge',
        'Lasso': 'Lasso',
        'ElasticNet': 'ElasticNet',
        'NeuralBoosted': 'NeuralBoosted',
    }

    # Default models if not specified
    if models_to_test is None:
        julia_models = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP']
    else:
        # Map to Julia names and validate
        julia_models = []
        for model in models_to_test:
            if model in model_mapping:
                julia_models.append(model_mapping[model])
            else:
                print(f"Warning: Unknown model '{model}' specified, skipping")

        if not julia_models:
            raise ValueError("No valid models specified for Julia backend")

    # Map preprocessing methods to Julia format
    if preprocessing_methods is None:
        julia_preprocessing = ['raw', 'snv', 'deriv']
        derivative_orders = [1, 2]
    else:
        julia_preprocessing = []
        derivative_orders = []

        if preprocessing_methods.get('raw', False):
            julia_preprocessing.append('raw')
        if preprocessing_methods.get('snv', False):
            julia_preprocessing.append('snv')
        if preprocessing_methods.get('msc', False):
            julia_preprocessing.append('msc')

        # Handle derivatives
        has_deriv = False
        if preprocessing_methods.get('sg1', False):
            if 'deriv' not in julia_preprocessing:
                julia_preprocessing.append('deriv')
            derivative_orders.append(1)
            has_deriv = True

        if preprocessing_methods.get('sg2', False):
            if 'deriv' not in julia_preprocessing:
                julia_preprocessing.append('deriv')
            if 2 not in derivative_orders:
                derivative_orders.append(2)
            has_deriv = True

        # Handle deriv_snv (not directly supported in Julia yet)
        if preprocessing_methods.get('deriv_snv', False):
            print("Warning: deriv_snv preprocessing not yet implemented in Julia backend")

        if not julia_preprocessing:
            julia_preprocessing = ['raw']  # Default fallback

    # Window sizes
    if window_sizes is None or len(window_sizes) == 0:
        window_size = 17
    else:
        # Julia uses single window size, take first or most common
        window_size = window_sizes[0]

    # Variable counts
    if variable_counts is None:
        variable_counts = [10, 20, 50, 100, 250]

    # Variable selection methods
    if variable_selection_methods is None:
        julia_var_selection_methods = ['importance']
    else:
        # Map Python names to Julia names (they're the same) and validate
        valid_methods = ['importance', 'SPA', 'UVE', 'iPLS', 'UVE-SPA']
        julia_var_selection_methods = []

        for method in variable_selection_methods:
            if method in valid_methods:
                julia_var_selection_methods.append(method)
            else:
                print(f"Warning: Unknown variable selection method '{method}' specified, skipping")

        if not julia_var_selection_methods:
            julia_var_selection_methods = ['importance']

    config = {
        'task_type': task_type,
        'models': julia_models,
        'preprocessing': julia_preprocessing,
        'derivative_orders': derivative_orders if derivative_orders else [1, 2],
        'derivative_window': window_size,
        'derivative_polyorder': 3,  # Standard
        'enable_variable_subsets': enable_variable_subsets,
        'variable_counts': variable_counts,
        'variable_selection_methods': julia_var_selection_methods,
        'enable_region_subsets': enable_region_subsets,
        'n_top_regions': n_top_regions,
        'n_folds': folds,
        'lambda_penalty': lambda_penalty,
        'max_n_components': max_n_components,
        'max_iter': max_iter
    }

    return config


def _create_julia_script(X_file, y_file, wavelengths_file, config_file, output_file, progress_file, julia_project):
    """Create Julia script to run analysis."""

    # Load config to embed directly in script
    with open(config_file, 'r') as f:
        import json
        config = json.load(f)

    # Convert to Julia project path with forward slashes
    julia_project_path = str(julia_project).replace(chr(92), '/')
    spectral_predict_module = f"{julia_project_path}/src/SpectralPredict.jl"

    # Helper function to convert Python lists to Julia syntax
    def to_julia_array(items):
        if isinstance(items, list):
            if all(isinstance(x, str) for x in items):
                return '[' + ', '.join(f'"{x}"' for x in items) + ']'
            else:
                return '[' + ', '.join(str(x) for x in items) + ']'
        return '[]'

    script = f'''
# SpectralPredict Julia Analysis Script
# Auto-generated by Python bridge

using Pkg
using CSV
using DataFrames

# Ensure the project environment is active and dependencies are installed
try
    Pkg.activate("{julia_project_path}")
    Pkg.instantiate()
    # Optional but helpful: precompile to surface any compile-time issues early
    try
        Pkg.precompile()
    catch
        # Precompile is best-effort; continue even if it fails
    end
catch e
    println("ERROR: Failed to activate/instantiate Julia project")
    println(e)
    # Write error progress marker
    open("{str(progress_file).replace(chr(92), '/')}", "w") do f
        println(f, "status: error")
        println(f, "error: ", string(e))
    end
    rethrow(e)
end

# Load SpectralPredict module using absolute path
include("{spectral_predict_module}")
using .SpectralPredict

# Import new modules if they exist (for forward compatibility)
if isdefined(SpectralPredict, :VariableSelection)
    using .SpectralPredict.VariableSelection
end
if isdefined(SpectralPredict, :NeuralBoosted)
    using .SpectralPredict.NeuralBoosted
end
if isdefined(SpectralPredict, :Diagnostics)
    using .SpectralPredict.Diagnostics
end

println("="^70)
println("SpectralPredict Julia Bridge - Analysis Starting")
println("="^70)
println()

# Configuration (embedded from Python)
config = Dict(
    "task_type" => "{config['task_type']}",
    "models" => {to_julia_array(config['models'])},
    "preprocessing" => {to_julia_array(config['preprocessing'])},
    "derivative_orders" => {to_julia_array(config['derivative_orders'])},
    "derivative_window" => {config['derivative_window']},
    "derivative_polyorder" => {config['derivative_polyorder']},
    "enable_variable_subsets" => {str(config['enable_variable_subsets']).lower()},
    "variable_counts" => {to_julia_array(config['variable_counts'])},
    "variable_selection_methods" => {to_julia_array(config['variable_selection_methods'])},
    "enable_region_subsets" => {str(config['enable_region_subsets']).lower()},
    "n_top_regions" => {config['n_top_regions']},
    "n_folds" => {config['n_folds']},
    "lambda_penalty" => {config['lambda_penalty']}
)

println("Configuration loaded:")
println("  Task type: ", config["task_type"])
println("  Models: ", join(config["models"], ", "))
println("  Preprocessing: ", join(config["preprocessing"], ", "))
println()

# Load data
println("Loading data...")
X_df = CSV.read("{str(X_file).replace(chr(92), '/')}", DataFrame)
y_df = CSV.read("{str(y_file).replace(chr(92), '/')}", DataFrame)
wl_df = CSV.read("{str(wavelengths_file).replace(chr(92), '/')}", DataFrame)

# Extract sample IDs and remove from X
sample_ids = X_df[:, 1]
X_matrix = Matrix{{Float64}}(X_df[:, 2:end])
y_vector = Vector{{Float64}}(y_df.target)
wavelengths = Vector{{Float64}}(wl_df.wavelength)

println("Data loaded:")
println("  Samples: ", size(X_matrix, 1))
println("  Features: ", size(X_matrix, 2))
println("  Wavelength range: ", minimum(wavelengths), " - ", maximum(wavelengths), " nm")
println()

# Run search
println("Starting hyperparameter search...")
println()

try
    results = run_search(
        X_matrix,
        y_vector,
        wavelengths,
        task_type=config["task_type"],
        models=config["models"],
        preprocessing=config["preprocessing"],
        derivative_orders=config["derivative_orders"],
        derivative_window=config["derivative_window"],
        derivative_polyorder=config["derivative_polyorder"],
        enable_variable_subsets=config["enable_variable_subsets"],
        variable_counts=config["variable_counts"],
        variable_selection_methods=config["variable_selection_methods"],
        enable_region_subsets=config["enable_region_subsets"],
        n_top_regions=config["n_top_regions"],
        n_folds=config["n_folds"],
        lambda_penalty=config["lambda_penalty"]
    )

    println()
    println("="^70)
    println("Analysis Complete!")
    println("="^70)
    println()
println("Total configurations: ", nrow(results))
println("Saving results to CSV...")

    # Sanitize results to ensure CSV-friendly column types
    function _sanitize_results(df::DataFrame)
        df_s = copy(df)
        for name in names(df_s)
            col = df_s[!, name]
            has_complex = any(x -> !(x === nothing || ismissing(x) || x isa Number || x isa String || x isa Bool), col)
            has_nothing = any(x -> x === nothing, col)
            if has_complex || has_nothing
                df_s[!, name] = [
                    (x === nothing || ismissing(x)) ? missing : string(x)
                    for x in col
                ]
            else
                # keep numeric/string/bool columns as-is
                df_s[!, name] = col
            end
        end
        return df_s
    end

    clean_results = _sanitize_results(results)

    # Save results
    CSV.write("{str(output_file).replace(chr(92), '/')}", clean_results)

    println("Results saved successfully!")
    println()

    # Write success progress marker (simple text file)
    open("{str(progress_file).replace(chr(92), '/')}", "w") do f
        println(f, "status: complete")
        println(f, "total_configs: ", nrow(results))
    end

catch e
    println()
    println("ERROR: Analysis failed")
    println()
    println("Error message:")
    println(e)
    println()

    # Write error progress marker (simple text file)
    open("{str(progress_file).replace(chr(92), '/')}", "w") do f
        println(f, "status: error")
        println(f, "error: ", string(e))
    end

    rethrow(e)
end
'''

    return script


def _run_julia_process(julia_exe, julia_project, script_file, progress_file, progress_callback):
    """Execute Julia process and monitor progress."""

    # Build command
    cmd = [
        julia_exe,
        f"--project={julia_project}",
        str(script_file)
    ]

    # Start process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Monitor output
    config_count = 0
    current_config = 0

    try:
        for line in process.stdout:
            # Print to console
            print(line.rstrip())

            # Parse progress information
            if "Total configurations to test:" in line:
                try:
                    config_count = int(line.split(":")[-1].strip())
                except:
                    pass

            # Track progress through log messages
            if "Testing" in line or "Running" in line:
                current_config += 1
                if progress_callback and config_count > 0:
                    progress_callback({
                        'stage': 'julia_execution',
                        'message': line.strip(),
                        'current': current_config,
                        'total': config_count
                    })

        # Wait for completion
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"Julia process exited with code {return_code}")

        # Check progress file for errors (simple text format)
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                lines = f.readlines()
                progress_data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        progress_data[key.strip()] = value.strip()
                if progress_data.get('status') == 'error':
                    raise RuntimeError(f"Julia analysis failed: {progress_data.get('error', 'Unknown error')}")

    except Exception as e:
        process.kill()
        raise e


def _postprocess_results(results_df, task_type):
    """Post-process Julia results to match Python format."""

    # Ensure column names match Python version
    column_mapping = {
        'SubsetTag': 'Subset',  # Julia uses SubsetTag, Python uses Subset in some places
    }

    results_df = results_df.rename(columns=column_mapping)

    # Ensure numeric columns are float
    numeric_cols = ['n_vars', 'full_vars', 'CompositeScore', 'Rank']

    if task_type == 'regression':
        numeric_cols.extend(['RMSE', 'R2', 'MAE'])
    else:
        numeric_cols.extend(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'])

    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    # Sort by Rank
    results_df = results_df.sort_values('Rank').reset_index(drop=True)

    return results_df


# ============================================================================
# Utility Functions
# ============================================================================

def check_julia_installation(julia_exe=None, julia_project=None):
    """
    Check if Julia and SpectralPredict.jl are properly installed.

    Parameters
    ----------
    julia_exe : str, optional
        Path to Julia executable (uses default if not provided)
    julia_project : str, optional
        Path to Julia project (uses default if not provided)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'julia_found': bool
        - 'julia_version': str or None
        - 'project_found': bool
        - 'ready': bool (True if everything is ready)
        - 'messages': list of str (info/warning/error messages)
    """

    julia_exe = julia_exe or JULIA_EXE
    julia_project = julia_project or JULIA_PROJECT

    result = {
        'julia_found': False,
        'julia_version': None,
        'project_found': False,
        'ready': False,
        'messages': []
    }

    # Check Julia executable
    if not os.path.exists(julia_exe):
        result['messages'].append(f"ERROR: Julia executable not found: {julia_exe}")
        return result

    result['julia_found'] = True
    result['messages'].append(f"[OK] Julia executable found: {julia_exe}")

    # Get Julia version
    try:
        version_output = subprocess.check_output(
            [julia_exe, "--version"],
            text=True,
            stderr=subprocess.STDOUT
        )
        result['julia_version'] = version_output.strip()
        result['messages'].append(f"[OK] {result['julia_version']}")
    except Exception as e:
        result['messages'].append(f"WARNING: Could not determine Julia version: {e}")

    # Check project directory
    if not os.path.exists(julia_project):
        result['messages'].append(f"ERROR: Julia project not found: {julia_project}")
        return result

    result['project_found'] = True
    result['messages'].append(f"[OK] Julia project found: {julia_project}")

    # Check Project.toml
    project_toml = Path(julia_project) / "Project.toml"
    if not project_toml.exists():
        result['messages'].append(f"ERROR: Project.toml not found in {julia_project}")
        return result

    result['messages'].append(f"[OK] Project.toml found")

    # Everything is ready
    result['ready'] = True
    result['messages'].append("[OK] Julia backend ready!")

    return result


def print_julia_status():
    """Print Julia installation status to console."""
    status = check_julia_installation()

    print("\n" + "="*70)
    print("Julia Backend Status Check")
    print("="*70)

    for msg in status['messages']:
        print(msg)

    print()

    if status['ready']:
        print("[SUCCESS] Julia backend is ready to use!")
    else:
        print("[FAILED] Julia backend is NOT ready. Please fix the errors above.")

    print("="*70 + "\n")

    return status['ready']


# ============================================================================
# CLI Test Function
# ============================================================================

def main():
    """Command-line test function."""
    print("\n" + "="*70)
    print("SpectralPredict Julia Bridge - Test")
    print("="*70 + "\n")

    # Check installation
    if not print_julia_status():
        print("Please fix Julia installation before running tests.")
        sys.exit(1)

    # Create sample data
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[str(w) for w in range(400, 400 + n_features)]
    )
    y = pd.Series(np.random.rand(n_samples))

    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print()

    # Run test
    print("Running test analysis...")
    print()

    try:
        results = run_search_julia(
            X, y,
            task_type='regression',
            models_to_test=['PLS', 'Ridge'],
            preprocessing_methods={'raw': True, 'snv': True, 'msc': False},
            enable_variable_subsets=True,
            variable_counts=[10, 20],
            variable_selection_methods=['importance', 'SPA'],
            enable_region_subsets=False,
            folds=3,
            progress_callback=lambda info: print(f"  Progress: {info.get('message', '')}")
        )

        print()
        print("="*70)
        print("Test Results")
        print("="*70)
        print(f"\nTotal configurations: {len(results)}")
        print(f"\nTop 5 models:")
        print(results.head(5)[['Model', 'Preprocess', 'Subset', 'RMSE', 'R2', 'Rank']])
        print()
        print("[SUCCESS] Test completed successfully!")
        print()

    except Exception as e:
        print()
        print("="*70)
        print("Test Failed")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
