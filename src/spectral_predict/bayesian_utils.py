"""
Utility functions for Bayesian hyperparameter optimization with Optuna.

This module provides helper functions for:
    - Creating reproducible Optuna studies
    - Converting parameters between formats
    - Handling pruning and early stopping
    - Error handling and validation
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging


# Configure logging for Optuna (suppress verbose output)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_optuna_study(
    direction: str = 'minimize',
    sampler: str = 'TPE',
    pruner: Optional[str] = None,
    random_state: int = 42,
    study_name: Optional[str] = None
) -> optuna.Study:
    """
    Create an Optuna study with specified configuration.

    Parameters
    ----------
    direction : str, default='minimize'
        Optimization direction ('minimize' for RMSE, 'maximize' for R²)
    sampler : str, default='TPE'
        Sampling algorithm:
            - 'TPE': Tree-structured Parzen Estimator (recommended, smart sampling)
            - 'Random': Random sampling (baseline)
    pruner : str or None, default=None
        Pruning strategy for early stopping:
            - None: No pruning
            - 'Median': Stop if trial is worse than median
            - 'Halving': Successive halving (aggressive)
    random_state : int, default=42
        Random seed for reproducibility
    study_name : str, optional
        Name for the study (for logging)

    Returns
    -------
    study : optuna.Study
        Configured Optuna study object

    Notes
    -----
    TPE sampler:
        - Uses Bayesian optimization to suggest promising parameters
        - Learns from previous trials
        - Typically finds good solutions in 20-50 trials

    Median pruner:
        - Stops trials that perform worse than median after K steps
        - Saves computation time on unpromising configurations
        - Safe default for most cases
    """
    # Configure sampler
    if sampler == 'TPE':
        sampler_obj = TPESampler(
            seed=random_state,
            n_startup_trials=10,  # Random exploration first
            n_ei_candidates=24,   # Number of candidates for expected improvement
            multivariate=True     # Consider parameter interactions
        )
    elif sampler == 'Random':
        sampler_obj = RandomSampler(seed=random_state)
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Use 'TPE' or 'Random'")

    # Configure pruner
    if pruner == 'Median':
        pruner_obj = MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=2,    # Wait 2 CV folds before pruning
            interval_steps=1     # Check after each fold
        )
    elif pruner == 'Halving':
        pruner_obj = SuccessiveHalvingPruner(
            min_resource=1,      # Start with 1 fold
            reduction_factor=3   # Keep top 1/3 trials
        )
    elif pruner is None:
        pruner_obj = None
    else:
        raise ValueError(f"Unknown pruner: {pruner}. Use 'Median', 'Halving', or None")

    # Create study (in-memory, no database)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler_obj,
        pruner=pruner_obj,
        study_name=study_name,
        storage=None  # In-memory (no SQLite locking issues)
    )

    return study


def create_objective_function(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    preprocess_cfg: Dict,
    cv_splitter,
    task_type: str,
    is_binary_classification: bool,
    run_single_config_fn: Callable,
    tier: str = 'standard',
    n_features: int = None,
    max_n_components: int = 8,
    **kwargs
) -> Callable:
    """
    Create objective function for Optuna optimization.

    Parameters
    ----------
    model_name : str
        Name of model to optimize
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    wavelengths : np.ndarray
        Wavelength values
    preprocess_cfg : dict
        Preprocessing configuration
    cv_splitter : sklearn splitter
        Cross-validation splitter
    task_type : str
        'regression' or 'classification'
    is_binary_classification : bool
        Whether classification is binary
    run_single_config_fn : callable
        Function to run single model config (from search.py)
    tier : str
        Model tier level
    n_features : int, optional
        Number of features
    max_n_components : int
        Maximum PLS components
    **kwargs : dict
        Additional parameters for run_single_config

    Returns
    -------
    objective : callable
        Objective function for Optuna

    Notes
    -----
    The objective function:
    1. Suggests hyperparameters using Optuna trial
    2. Trains model with CV using existing infrastructure
    3. Returns metric to minimize (RMSE) or maximize (R²)
    """
    from .bayesian_config import get_bayesian_search_space
    from .models import build_model

    # Calculate n_classes for classification tasks
    n_classes = len(np.unique(y)) if task_type == 'classification' else 2

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object

        Returns
        -------
        metric : float
            Metric to optimize (RMSE for regression, negative accuracy for classification)
        """
        # Get hyperparameters from Optuna
        params = get_bayesian_search_space(
            model_name=model_name,
            trial=trial,
            tier=tier,
            n_features=n_features,
            max_n_components=max_n_components,
            task_type=task_type,
            n_classes=n_classes
        )

        # Build model with suggested parameters
        model = build_model(model_name, params, task_type=task_type)

        # Run cross-validation using existing infrastructure
        try:
            result = run_single_config_fn(
                X, y, wavelengths,
                model, model_name, params,
                preprocess_cfg, cv_splitter, task_type,
                is_binary_classification,
                skip_preprocessing=True,  # Already preprocessed
                **kwargs
            )

            # Extract metric to optimize
            if task_type == 'regression':
                # Minimize RMSE
                metric = result['RMSE']
                # Store R² as user attribute for reporting
                if 'R2' in result:
                    trial.set_user_attr('R2', result['R2'])
            else:
                # Maximize accuracy (minimize negative accuracy)
                metric = -result['Accuracy']
                # Store ROC_AUC as user attribute for reporting
                if 'ROC_AUC' in result:
                    trial.set_user_attr('ROC_AUC', result['ROC_AUC'])

            return metric

        except Exception as e:
            # If model training fails, return large penalty value
            # This marks the trial as completed but with worst score
            logging.warning(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
            # Return very large penalty (for minimization) or very negative (for maximization)
            if task_type == 'regression':
                return 1e10  # Large RMSE penalty
            else:
                return 1e10  # Large penalty (negative accuracy is being minimized)

    return objective


def convert_optuna_result_to_dasp_format(
    study: optuna.Study,
    model_name: str,
    preprocess_cfg: Dict,
    task_type: str
) -> Dict:
    """
    Convert Optuna study result to DASP result format.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    model_name : str
        Model name
    preprocess_cfg : dict
        Preprocessing configuration
    task_type : str
        'regression' or 'classification'

    Returns
    -------
    result : dict
        Result in DASP format (compatible with results DataFrame)
    """
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    result = {
        'Model': model_name,
        'Params': str(best_params),
        'Preprocess': preprocess_cfg.get('name', 'unknown'),
        'Deriv': preprocess_cfg.get('deriv', 0),
        'Window': preprocess_cfg.get('window', 0),
        'Poly': preprocess_cfg.get('polyorder', 0),
        'n_trials': len(study.trials),
        'best_trial_number': best_trial.number,
        'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration)
    }

    if task_type == 'regression':
        result['RMSE'] = best_value
        result['R2'] = best_trial.user_attrs.get('R2', np.nan)
    else:
        result['Accuracy'] = -best_value  # Un-negate
        result['ROC_AUC'] = best_trial.user_attrs.get('ROC_AUC', np.nan)

    return result


def print_optimization_summary(study: optuna.Study, model_name: str):
    """
    Print summary of Bayesian optimization results.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    model_name : str
        Model name
    """
    print(f"\n{'='*70}")
    print(f"Bayesian Optimization Summary: {model_name}")
    print(f"{'='*70}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Print trial statistics
    values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if values:
        print(f"\nTrial statistics:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")

    print(f"{'='*70}\n")


def get_param_importance(study: optuna.Study, top_n: int = 5) -> Dict[str, float]:
    """
    Calculate hyperparameter importance using fANOVA.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    top_n : int
        Number of top parameters to return

    Returns
    -------
    importance : dict
        Dictionary mapping parameter names to importance scores
    """
    try:
        from optuna.importance import get_param_importances

        importance = get_param_importances(study)

        # Get top N
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:top_n])

    except Exception:
        # If importance calculation fails, return empty dict
        return {}


def save_optimization_plots(study: optuna.Study, output_dir: str, model_name: str):
    """
    Save Optuna visualization plots.

    Parameters
    ----------
    study : optuna.Study
        Completed study
    output_dir : str
        Directory to save plots
    model_name : str
        Model name (for filename)

    Notes
    -----
    Requires plotly to be installed. Silently skips if not available.
    """
    try:
        import optuna.visualization as vis
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_history.html'))

        # Parameter importances
        if len(study.trials) > 10:  # Need enough trials
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(output_dir, f'{model_name}_importance.html'))

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_parallel.html'))

    except ImportError:
        # Plotly not installed - silently skip
        pass
    except Exception as e:
        logging.warning(f"Could not save plots: {e}")


class ProgressCallback:
    """
    Callback for progress reporting during Bayesian optimization.

    Parameters
    ----------
    progress_callback : callable, optional
        Function to call with progress updates (for GUI)
    model_name : str
        Model name for logging
    n_trials : int
        Total number of trials
    """

    def __init__(self, progress_callback: Optional[Callable] = None, model_name: str = '', n_trials: int = 30):
        self.progress_callback = progress_callback
        self.model_name = model_name
        self.n_trials = n_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Called after each trial completes."""
        if self.progress_callback:
            self.progress_callback({
                'stage': 'bayesian_optimization',
                'message': f'{self.model_name}: Trial {trial.number + 1}/{self.n_trials} - Score: {trial.value:.4f}',
                'current': trial.number + 1,
                'total': self.n_trials,
                'best_model': {'RMSE': study.best_value if study.best_value else np.nan}
            })


def handle_failed_trial(trial: optuna.Trial, exception: Exception) -> float:
    """
    Handle failed trials gracefully.

    Parameters
    ----------
    trial : optuna.Trial
        Failed trial
    exception : Exception
        Exception that caused failure

    Returns
    -------
    penalty : float
        Large penalty value to mark trial as failed

    Notes
    -----
    Common failure causes:
    - Model doesn't converge
    - Invalid hyperparameter combination
    - Numerical instability
    """
    logging.warning(f"Trial {trial.number} failed: {type(exception).__name__}: {exception}")

    # Return very large value (worst possible score)
    return 1e10  # Will be marked as worst trial


if __name__ == '__main__':
    # Example usage
    print("Bayesian Optimization Utilities - Example")
    print("=" * 70)

    # Create study
    study = create_optuna_study(
        direction='minimize',
        sampler='TPE',
        pruner='Median',
        random_state=42
    )

    print(f"✓ Created study: {study.study_name}")
    print(f"  Sampler: {type(study.sampler).__name__}")
    print(f"  Pruner: {type(study.pruner).__name__}")
    print(f"  Direction: {study.direction}")

    # Example objective function
    def simple_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return x**2 + y**2

    # Optimize
    study.optimize(simple_objective, n_trials=20, show_progress_bar=False)

    # Print summary
    print_optimization_summary(study, "Simple Quadratic")

    # Parameter importance
    importance = get_param_importance(study)
    if importance:
        print("Parameter Importance:")
        for param, score in importance.items():
            print(f"  {param}: {score:.3f}")
