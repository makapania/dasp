"""
Reproducibility utilities for spectral prediction.

This module provides functions to control sources of non-determinism
in numerical computing, ensuring bit-identical results across runs
for scientific research applications.
"""

import os
import warnings
from contextlib import contextmanager


def set_blas_threads(n_threads=1):
    """
    Set the number of threads used by BLAS/LAPACK libraries.

    This function controls threading in various numerical libraries to ensure
    reproducible results. Multi-threaded BLAS operations can produce slightly
    different results due to non-deterministic thread scheduling and parallel
    reduction order, even with the same random seed.

    Parameters
    ----------
    n_threads : int, default=1
        Number of threads to use. Set to 1 for full reproducibility.
        Higher values trade reproducibility for performance.

    Notes
    -----
    This function sets environment variables that must be configured BEFORE
    importing numpy/scipy. If called after these libraries are imported,
    it may not take full effect.

    Controlled libraries:
    - OpenBLAS (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS)
    - Intel MKL (MKL_NUM_THREADS)
    - Apple Accelerate (VECLIB_MAXIMUM_THREADS)
    - NumExpr (NUMEXPR_NUM_THREADS)

    Examples
    --------
    >>> # Call before any numerical imports
    >>> set_blas_threads(1)
    >>> import numpy as np
    >>> import scipy

    >>> # For production (faster, less reproducible)
    >>> set_blas_threads(4)
    """
    n_threads_str = str(n_threads)

    # OpenMP (used by many BLAS implementations)
    os.environ['OMP_NUM_THREADS'] = n_threads_str

    # OpenBLAS
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads_str

    # Intel MKL
    os.environ['MKL_NUM_THREADS'] = n_threads_str

    # Apple Accelerate
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_threads_str

    # NumExpr (used by pandas)
    os.environ['NUMEXPR_NUM_THREADS'] = n_threads_str

    # Also set threadpool size in numpy if already imported
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads)
    except ImportError:
        # numpy not yet imported - environment variables will take effect
        pass


def ensure_reproducibility(random_state=42, n_threads=1):
    """
    Configure all reproducibility settings for scientific computing.

    This is a convenience function that sets up all necessary controls
    for reproducible scientific computing:
    - Sets BLAS/LAPACK threads to 1
    - Configures random seeds for numpy
    - Warns about potential remaining sources of non-determinism

    Parameters
    ----------
    random_state : int, default=42
        Random seed for numpy and other libraries.

    n_threads : int, default=1
        Number of BLAS threads (1 for full reproducibility).

    Returns
    -------
    dict
        Configuration summary with keys:
        - 'blas_threads': Number of BLAS threads set
        - 'random_state': Random seed used
        - 'warnings': List of potential reproducibility issues

    Examples
    --------
    >>> config = ensure_reproducibility(random_state=42)
    >>> print(f"BLAS threads: {config['blas_threads']}")
    >>> print(f"Random state: {config['random_state']}")
    """
    warnings_list = []

    # Set BLAS threads
    set_blas_threads(n_threads)

    # Set numpy random seed
    try:
        import numpy as np
        np.random.seed(random_state)
    except ImportError:
        warnings_list.append("numpy not available - could not set random seed")

    # Check for potential GPU operations
    try:
        import torch
        if torch.cuda.is_available():
            warnings_list.append(
                "PyTorch GPU detected - GPU operations may not be reproducible. "
                "Use torch.use_deterministic_algorithms(True) for GPU reproducibility."
            )
    except ImportError:
        pass  # No PyTorch, no GPU issues

    # Check BLAS library
    try:
        import numpy as np
        config_info = np.__config__.show()
        # Note: This is informational, not a warning
    except:
        pass

    config = {
        'blas_threads': n_threads,
        'random_state': random_state,
        'warnings': warnings_list
    }

    return config


def check_reproducibility_status():
    """
    Check current reproducibility configuration and report status.

    Returns
    -------
    dict
        Status report with keys:
        - 'blas_library': Name of BLAS library in use
        - 'blas_threads_env': Current environment variable settings
        - 'numpy_version': NumPy version
        - 'scipy_version': SciPy version (if available)
        - 'sklearn_version': scikit-learn version (if available)

    Examples
    --------
    >>> status = check_reproducibility_status()
    >>> print(status['blas_library'])
    >>> print(f"OMP_NUM_THREADS: {status['blas_threads_env']['OMP_NUM_THREADS']}")
    """
    status = {
        'blas_threads_env': {},
        'numpy_version': None,
        'scipy_version': None,
        'sklearn_version': None,
        'blas_library': 'unknown'
    }

    # Check environment variables
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        status['blas_threads_env'][var] = os.environ.get(var, 'not set')

    # Check library versions
    try:
        import numpy as np
        status['numpy_version'] = np.__version__

        # Try to detect BLAS library
        try:
            config = np.__config__
            if hasattr(config, 'blas_opt_info'):
                blas_info = config.blas_opt_info
                if 'libraries' in blas_info:
                    libs = blas_info['libraries']
                    if any('mkl' in lib.lower() for lib in libs):
                        status['blas_library'] = 'Intel MKL'
                    elif any('openblas' in lib.lower() for lib in libs):
                        status['blas_library'] = 'OpenBLAS'
                    elif any('accelerate' in lib.lower() for lib in libs):
                        status['blas_library'] = 'Apple Accelerate'
        except:
            pass
    except ImportError:
        pass

    try:
        import scipy
        status['scipy_version'] = scipy.__version__
    except ImportError:
        pass

    try:
        import sklearn
        status['sklearn_version'] = sklearn.__version__
    except ImportError:
        pass

    return status


@contextmanager
def reproducible_context(n_threads=1, random_state=42):
    """
    Context manager for reproducible execution.

    This context manager temporarily sets BLAS threads and random seeds,
    then restores the original settings when exiting the context.

    Parameters
    ----------
    n_threads : int, default=1
        Number of BLAS threads (1 for full reproducibility)
    random_state : int, default=42
        Random seed for numpy

    Yields
    ------
    dict
        Configuration that was applied

    Examples
    --------
    >>> with reproducible_context():
    ...     # Code here runs with BLAS=1 and seeded RNG
    ...     results = run_analysis(data)
    >>> # BLAS threads and RNG restored to original state here

    Notes
    -----
    This is the recommended way to use reproducible mode, as it ensures
    settings are properly restored even if an exception occurs.
    """
    # Save current environment variables
    env_vars = [
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS'
    ]

    saved_env = {}
    for var in env_vars:
        saved_env[var] = os.environ.get(var, None)

    # Save numpy random state if numpy is imported
    saved_numpy_state = None
    try:
        import numpy as np
        saved_numpy_state = np.random.get_state()
    except ImportError:
        pass

    try:
        # Apply reproducible settings
        set_blas_threads(n_threads)

        if saved_numpy_state is not None:
            import numpy as np
            np.random.seed(random_state)

        # Try to use threadpoolctl for runtime control (more reliable)
        threadpool_controller = None
        try:
            from threadpoolctl import threadpool_limits
            # Create controller but don't activate yet
            # User code will run inside this limit
            threadpool_controller = threadpool_limits(limits=n_threads)
            threadpool_controller.__enter__()
        except Exception:
            pass  # threadpoolctl failed, use environment variables only

        config = {
            'blas_threads': n_threads,
            'random_state': random_state,
            'method': 'threadpoolctl' if threadpool_controller else 'environment'
        }

        yield config

    finally:
        # Restore environment variables
        for var, value in saved_env.items():
            if value is None:
                # Variable wasn't set before, remove it
                os.environ.pop(var, None)
            else:
                # Restore original value
                os.environ[var] = value

        # Restore numpy random state
        if saved_numpy_state is not None:
            import numpy as np
            np.random.set_state(saved_numpy_state)

        # Exit threadpoolctl context if it was used
        if threadpool_controller is not None:
            try:
                threadpool_controller.__exit__(None, None, None)
            except:
                pass


def restore_default_threads():
    """
    Restore BLAS threading to system defaults (unrestricted).

    This function removes thread restrictions that may have been set
    by reproducible mode, allowing BLAS to use all available cores.

    Examples
    --------
    >>> set_blas_threads(1)  # Restrict to 1 thread
    >>> # ... do reproducible work ...
    >>> restore_default_threads()  # Back to full speed

    Notes
    -----
    After calling this function, BLAS libraries will use their default
    behavior (typically all available cores). This may make results
    non-reproducible.
    """
    # Remove thread limits by unsetting environment variables
    env_vars = [
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS'
    ]

    for var in env_vars:
        os.environ.pop(var, None)

    # Also reset threadpoolctl
    from threadpoolctl import threadpool_limits
    # Setting limits to None removes restrictions
    threadpool_limits(limits=None)

    print("BLAS thread restrictions removed. Using system defaults (all cores).")
