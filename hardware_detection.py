#!/usr/bin/env python3
"""
Hardware Detection and Adaptive Performance Module

Auto-detects CPU cores, GPU, and memory to automatically select
optimal performance settings. Gracefully degrades for lower-end hardware.

Usage:
    from hardware_detection import detect_hardware, get_model_params

    # Auto-detect
    hw_config = detect_hardware()

    # Get optimal parameters for model
    model_params = get_model_params('XGBoost', hw_config)
"""

import multiprocessing
import warnings

# psutil is optional - used for RAM detection
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: psutil not available. RAM detection will use estimates.")


def detect_hardware(verbose=True):
    """
    Detect hardware capabilities and return optimal configuration.

    Returns
    -------
    config : dict
        Hardware configuration with keys:
        - tier: int (1=Standard, 2=Parallel, 3=Power)
        - gpu_available: bool
        - gpu_type: str or None
        - n_workers: int (optimal worker count)
        - use_parallel: bool
        - use_gpu: bool
        - n_cores: int
        - memory_gb: float
    """
    config = {
        'tier': 1,
        'gpu_available': False,
        'gpu_type': None,
        'n_cores': 1,
        'n_workers': 1,
        'use_parallel': False,
        'use_gpu': False,
        'memory_gb': 0
    }

    if verbose:
        print("=" * 70)
        print("HARDWARE DETECTION")
        print("=" * 70)

    # Detect CPU cores
    config['n_cores'] = multiprocessing.cpu_count()
    # Use n-1 cores (leave 1 for OS)
    config['n_workers'] = max(1, config['n_cores'] - 1)

    if verbose:
        print(f"CPU: {config['n_cores']} cores detected")

    # Detect RAM
    if HAS_PSUTIL:
        config['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    else:
        # Estimate based on core count (rough heuristic)
        config['memory_gb'] = max(4, config['n_cores'] * 2)

    if verbose:
        print(f"RAM: {config['memory_gb']:.1f} GB")

    # Detect GPU
    gpu_info = _detect_gpu(verbose=verbose)
    config['gpu_available'] = gpu_info['available']
    config['gpu_type'] = gpu_info['type']

    # Determine performance tier
    if config['gpu_available'] and config['n_cores'] >= 8 and config['memory_gb'] >= 16:
        # High-end workstation
        config['tier'] = 3
        config['use_parallel'] = True
        config['use_gpu'] = True
        tier_name = "POWER MODE"
        tier_desc = f"GPU + {config['n_workers']} CPU cores"

    elif config['n_cores'] >= 4 and config['memory_gb'] >= 8:
        # Mid-range desktop (no GPU or not enough resources)
        config['tier'] = 2
        config['use_parallel'] = True
        config['use_gpu'] = False
        tier_name = "PARALLEL MODE"
        tier_desc = f"{config['n_workers']} CPU cores (no GPU)"

    else:
        # Laptop or older PC
        config['tier'] = 1
        config['use_parallel'] = False
        config['use_gpu'] = False
        tier_name = "STANDARD MODE"
        tier_desc = f"{config['n_cores']} cores (limited resources)"

    if verbose:
        print("-" * 70)
        print(f"PERFORMANCE TIER: {config['tier']} - {tier_name}")
        print(f"Configuration: {tier_desc}")
        print("=" * 70)
        print()

    return config


def _detect_gpu(verbose=True):
    """
    Detect GPU availability for XGBoost/LightGBM.

    Returns
    -------
    gpu_info : dict
        - available: bool
        - type: str or None (e.g., 'NVIDIA', 'AMD')
    """
    gpu_info = {'available': False, 'type': None}

    # Try XGBoost GPU
    try:
        import xgboost as xgb
        import numpy as np
        import sys
        from io import StringIO

        # Suppress ALL output during detection
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Suppress XGBoost warnings during detection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Try to create a GPU model and run a tiny fit
                test_model = xgb.XGBRegressor(
                    tree_method='gpu_hist',
                    gpu_id=0,
                    n_estimators=1,
                    verbosity=0
                )
                X_test = np.random.randn(10, 10).astype(np.float32)
                y_test = np.random.randn(10).astype(np.float32)
                test_model.fit(X_test, y_test, verbose=False)

            # If we got here, GPU works!
            gpu_info['available'] = True
            gpu_info['type'] = 'NVIDIA'  # XGBoost GPU requires CUDA (NVIDIA)

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if verbose:
            print("GPU: ✓ NVIDIA CUDA detected (XGBoost GPU enabled)")

    except Exception as e:
        # GPU not available or not working
        gpu_info['available'] = False
        gpu_info['type'] = None

        if verbose:
            error_msg = str(e)
            print("GPU: ✗ Not detected or not working")
            if len(error_msg) > 80:
                print(f"     Reason: {error_msg[:80]}...")
            else:
                print(f"     Reason: {error_msg}")
            print("     Note: Ensure NVIDIA drivers and CUDA are installed, and XGBoost is built with GPU support")

    return gpu_info


def get_model_params(model_name, hw_config, base_params=None):
    """
    Get optimal model parameters based on hardware.

    Parameters
    ----------
    model_name : str
        Model type: 'XGBoost', 'LightGBM', 'Ridge', 'PLS', etc.
    hw_config : dict
        Hardware configuration from detect_hardware()
    base_params : dict, optional
        Base parameters to merge with hardware-specific params

    Returns
    -------
    params : dict
        Optimized parameters for this hardware
    """
    if base_params is None:
        base_params = {}

    params = base_params.copy()

    # XGBoost
    if model_name == 'XGBoost':
        if hw_config['use_gpu']:
            # Use GPU
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            params['predictor'] = 'gpu_predictor'
        else:
            # Use optimized CPU histogram method
            params['tree_method'] = 'hist'
            params['predictor'] = 'cpu_predictor'

        # Use available threads
        if 'n_jobs' not in params:
            params['n_jobs'] = hw_config['n_workers']

    # LightGBM
    elif model_name == 'LightGBM':
        if hw_config['use_gpu']:
            # Use GPU
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        else:
            # Use CPU
            params['device'] = 'cpu'

        # Use available threads
        if 'n_jobs' not in params:
            params['n_jobs'] = hw_config['n_workers']

    # CatBoost
    elif model_name == 'CatBoost':
        if hw_config['use_gpu']:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        else:
            params['task_type'] = 'CPU'

        if 'thread_count' not in params:
            params['thread_count'] = hw_config['n_workers']

    # Other models (Ridge, PLS, Lasso, ElasticNet)
    else:
        # These typically have n_jobs parameter for sklearn
        if 'n_jobs' not in params and model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            params['n_jobs'] = hw_config['n_workers']

    return params


def estimate_memory_usage(X, y, hw_config):
    """
    Estimate memory usage and check if safe for current hardware.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target values
    hw_config : dict
        Hardware configuration

    Returns
    -------
    safe : bool
        True if estimated usage is safe
    estimated_gb : float
        Estimated memory usage in GB
    """
    import sys

    # Calculate data size
    data_size_gb = (X.nbytes + y.nbytes) / (1024**3)

    # Estimate total usage
    if hw_config['use_parallel']:
        # Multiprocessing duplicates data across workers
        # Add safety margin
        estimated_gb = data_size_gb * (hw_config['n_workers'] + 2)
    else:
        # Sequential mode - just need 2x for working copies
        estimated_gb = data_size_gb * 2

    # Available memory (leave 30% for OS)
    available_gb = hw_config['memory_gb'] * 0.7

    # Check if safe
    safe = estimated_gb <= available_gb

    return safe, estimated_gb


def adjust_for_memory(hw_config, X, y, verbose=True):
    """
    Adjust hardware configuration if memory is insufficient.

    Modifies hw_config in-place to reduce memory usage if needed.

    Parameters
    ----------
    hw_config : dict
        Hardware configuration (modified in-place)
    X : ndarray
        Feature matrix
    y : ndarray
        Target values
    verbose : bool
        Print warnings

    Returns
    -------
    hw_config : dict
        Potentially modified configuration
    """
    safe, estimated_gb = estimate_memory_usage(X, y, hw_config)

    if not safe:
        if verbose:
            print("⚠ WARNING: Dataset may exceed available memory")
            print(f"   Available: {hw_config['memory_gb']*0.7:.1f} GB")
            print(f"   Estimated: {estimated_gb:.1f} GB")

        # Fall back to sequential mode to reduce memory
        if hw_config['use_parallel']:
            if verbose:
                print("   → Disabling parallel mode to reduce memory usage")

            hw_config['use_parallel'] = False
            hw_config['n_workers'] = 1
            hw_config['tier'] = min(hw_config['tier'], 1)

            # Re-check
            safe, estimated_gb = estimate_memory_usage(X, y, hw_config)

            if safe:
                if verbose:
                    print("   ✓ Sequential mode should work")
            else:
                if verbose:
                    print("   ⚠ Warning: May still be tight on memory")

    return hw_config


def print_performance_tips(hw_config):
    """
    Print tips for improving performance based on current hardware.

    Parameters
    ----------
    hw_config : dict
        Hardware configuration
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE TIPS")
    print("=" * 70)

    if hw_config['tier'] == 1:
        print("Your system is running in STANDARD MODE.")
        print("\nTo improve performance, consider:")
        print("  • Upgrading RAM (currently {:.1f}GB, recommend 16GB+)".format(
            hw_config['memory_gb']))
        if hw_config['n_cores'] < 4:
            print("  • Upgrading CPU (currently {} cores, recommend 4+)".format(
                hw_config['n_cores']))
        if not hw_config['gpu_available']:
            print("  • Adding a GPU (NVIDIA with CUDA support)")

    elif hw_config['tier'] == 2:
        print("Your system is running in PARALLEL MODE (good performance).")
        print("\nTo further improve performance:")
        if not hw_config['gpu_available']:
            print("  • Add a GPU for 5-10x additional speedup")
            print("    (NVIDIA with CUDA support recommended)")

    elif hw_config['tier'] == 3:
        print("Your system is running in POWER MODE (maximum performance).")
        print("\n✓ You have optimal hardware configuration!")
        print("  • GPU: Available")
        print("  • CPU: {} cores".format(hw_config['n_cores']))
        print("  • RAM: {:.1f} GB".format(hw_config['memory_gb']))

    print("=" * 70 + "\n")


# Example usage
if __name__ == '__main__':
    # Detect hardware
    hw_config = detect_hardware(verbose=True)

    # Show tips
    print_performance_tips(hw_config)

    # Example: Get optimal XGBoost parameters
    print("\nExample: Optimal XGBoost parameters for your hardware:")
    xgb_params = get_model_params('XGBoost', hw_config, base_params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    })
    print(xgb_params)

    # Example: Memory check
    print("\nExample: Memory safety check:")
    import numpy as np
    X_test = np.random.randn(1000, 2000)
    y_test = np.random.randn(1000)

    safe, estimated = estimate_memory_usage(X_test, y_test, hw_config)
    print(f"Data size: {(X_test.nbytes + y_test.nbytes)/(1024**3):.2f} GB")
    print(f"Estimated usage: {estimated:.2f} GB")
    print(f"Safe: {'✓' if safe else '✗'}")
