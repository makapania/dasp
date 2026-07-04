"""
GA-PLS (Genetic Algorithm - Partial Least Squares) wavelength selection.

This module implements GA-based variable selection for spectroscopy,
following established chemometrics literature (Leardi 1998, Stefansson 2020).

The algorithm uses a binary chromosome where each gene represents whether
a wavelength is selected (1) or excluded (0). Multiple GA runs are aggregated
to produce stable importance scores based on selection frequency.

References
----------
Leardi, R. & Lupiáñez González, A. (1998). Genetic algorithms applied to
feature selection in PLS regression. Chemometrics and Intelligent
Laboratory Systems, 41(2), 195-207.

Stefansson, P., et al. (2020). Fast method for GA-PLS with simultaneous
feature selection and identification of optimal preprocessing technique.
Journal of Chemometrics, 34(3), e3195.
"""

import numpy as np
import threading
from typing import Optional, Callable, Dict, Any, Tuple, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed


# =============================================================================
# Constants and Defaults
# =============================================================================

DEFAULT_POPULATION_SIZE = 64
DEFAULT_GENERATIONS = 100
DEFAULT_CROSSOVER_RATE = 0.6
DEFAULT_MUTATION_RATE = 0.01
DEFAULT_N_RUNS = 5
DEFAULT_SELECTION_THRESHOLD = 0.6
DEFAULT_MIN_WAVELENGTHS = 5
DEFAULT_EARLY_STOPPING = 20


# =============================================================================
# Fitness Cache for Performance
# =============================================================================

class FitnessCache:
    """
    Thread-safe cache for fitness evaluations to avoid redundant CV runs.

    Key: tuple(chromosome as bytes)
    Value: fitness score
    """
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, chromosome: np.ndarray) -> Optional[float]:
        key = chromosome.tobytes()
        with self._lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def set(self, chromosome: np.ndarray, fitness: float):
        key = chromosome.tobytes()
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Simple LRU: clear oldest half
                keys = list(self.cache.keys())[:len(self.cache)//2]
                for k in keys:
                    del self.cache[k]
            self.cache[key] = fitness

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


# =============================================================================
# Genetic Algorithm Components
# =============================================================================

def _initialize_population(
    n_wavelengths: int,
    pop_size: int,
    init_ratio: float = 0.3,
    min_wavelengths: int = 5,
    seed_importances: Optional[np.ndarray] = None,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Initialize GA population with random binary chromosomes.

    Parameters
    ----------
    n_wavelengths : int
        Number of wavelengths (chromosome length)
    pop_size : int
        Population size
    init_ratio : float
        Expected selection ratio for random initialization
    min_wavelengths : int
        Minimum number of wavelengths to select
    seed_importances : np.ndarray, optional
        Prior importance scores to bias initialization
    rng : np.random.RandomState
        Random number generator

    Returns
    -------
    population : np.ndarray
        Binary population matrix (pop_size, n_wavelengths)
    """
    if rng is None:
        rng = np.random.RandomState()

    population = np.zeros((pop_size, n_wavelengths), dtype=np.uint8)

    for i in range(pop_size):
        if seed_importances is not None and i < pop_size // 4:
            # Seed 25% of population using prior importances
            probs = seed_importances / (seed_importances.sum() + 1e-10)
            n_select = max(min_wavelengths, int(n_wavelengths * init_ratio))
            selected = rng.choice(n_wavelengths, size=n_select, replace=False, p=probs)
            population[i, selected] = 1
        else:
            # Random initialization
            mask = rng.random(n_wavelengths) < init_ratio
            # Ensure minimum wavelengths
            if np.sum(mask) < min_wavelengths:
                additional = rng.choice(
                    np.where(~mask)[0],
                    size=min_wavelengths - np.sum(mask),
                    replace=False
                )
                mask[additional] = True
            population[i] = mask.astype(np.uint8)

    return population


def _fitness_function(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    task_type: str = 'regression',
    n_components: int = 10,
    min_wavelengths: int = 5
) -> float:
    """
    Evaluate chromosome fitness using cross-validated PLS.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary selection mask
    X : np.ndarray
        Full spectral data (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    cv : cross-validator
        KFold or StratifiedKFold
    task_type : str
        'regression' or 'classification'
    n_components : int
        Maximum PLS components
    min_wavelengths : int
        Minimum wavelengths required

    Returns
    -------
    fitness : float
        Negative RMSECV for regression (higher is better for GA)
        Accuracy for classification (higher is better)
    """
    selected_indices = np.where(chromosome == 1)[0]
    n_selected = len(selected_indices)

    # Invalid chromosome: too few wavelengths
    if n_selected < min_wavelengths:
        return -np.inf

    X_selected = X[:, selected_indices]

    # Auto-select n_components based on selected variables
    max_comp = min(n_components, n_selected // 2, X.shape[0] // 5)
    actual_n_comp = max(1, max_comp)

    try:
        if task_type == 'regression':
            pls = PLSRegression(n_components=actual_n_comp, scale=False)
            y_arr = np.asarray(y)
            if y_arr.ndim > 1 and y_arr.shape[1] > 1:
                # Multi-target (T-17): JOINT PLS fitness = pooled normalized
                # RMSECV sqrt(mean_target(1 - Q2_s)) on the multi_y foundation.
                # Scale-free (divergent target units) and monotone in joint Q2,
                # so higher fitness == higher joint Q2, matching single-Y sign.
                from .multi_y import multi_y_cv_pool, multi_y_metrics

                yt, yp = multi_y_cv_pool(pls, X_selected, y_arr, cv, scale_y=True)
                m = multi_y_metrics(yt, yp)
                joint_q2 = float(m["joint_q2"])
                mean_nmse = float(np.mean(1.0 - np.asarray(m["q2"])))
                # A non-finite per-target q2 (PLS divergence on a degenerate
                # chromosome) poisons mean_nmse to NaN; `NaN > 0.0` is False so
                # rmsecv would be 0.0 -> fitness -0.0, the MAXIMUM achievable
                # value, promoting a broken subset to best (GA maximizes). Fall
                # to the worst-case sentinel the except branch uses. Mirrors the
                # guard in variable_selection._evaluate_interval_pls_multi.
                if not (np.isfinite(joint_q2) and np.isfinite(mean_nmse)):
                    return -np.inf
                rmsecv = float(np.sqrt(mean_nmse)) if mean_nmse > 0.0 else 0.0
                return -rmsecv
            # Single-target path -- byte-identical to pre-T-17.
            y_pred = cross_val_predict(pls, X_selected, y, cv=cv)
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            return -rmsecv  # Negative because GA maximizes
        else:
            # Classification: PLS-DA (PLS + threshold)
            pls = PLSRegression(n_components=actual_n_comp, scale=False)
            X_scores = cross_val_predict(pls, X_selected, y, cv=cv, method='predict')
            # Use accuracy as fitness
            if hasattr(y, 'astype'):
                y_class = y.astype(int)
            else:
                y_class = np.array(y, dtype=int)
            # Threshold for binary classification
            y_pred = (X_scores > np.median(X_scores)).astype(int).ravel()
            return accuracy_score(y_class, y_pred)

    except Exception:
        return -np.inf


def _tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    tournament_size: int = 3,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Tournament selection with replacement.

    Parameters
    ----------
    population : np.ndarray
        Current population
    fitness_scores : np.ndarray
        Fitness for each individual
    tournament_size : int
        Number of individuals in each tournament
    rng : np.random.RandomState
        Random number generator

    Returns
    -------
    selected : np.ndarray
        Selected individuals (same size as population)
    """
    if rng is None:
        rng = np.random.RandomState()

    pop_size = len(population)
    selected = np.zeros_like(population)

    for i in range(pop_size):
        competitors = rng.choice(pop_size, size=tournament_size, replace=False)
        winner = competitors[np.argmax(fitness_scores[competitors])]
        selected[i] = population[winner].copy()

    return selected


def _two_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float = 0.6,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover - preserves spectral regions.

    Better for wavelength selection as it maintains contiguous regions.
    """
    if rng is None:
        rng = np.random.RandomState()

    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    points = sorted(rng.choice(n, size=2, replace=False))

    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[points[0]:points[1]] = parent2[points[0]:points[1]]
    child2[points[0]:points[1]] = parent1[points[0]:points[1]]

    return child1, child2


def _bit_flip_mutation(
    chromosome: np.ndarray,
    mutation_rate: float = 0.01,
    min_wavelengths: int = 5,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Standard bit-flip mutation with minimum wavelength constraint.
    """
    if rng is None:
        rng = np.random.RandomState()

    mutated = chromosome.copy()
    mutation_mask = rng.random(len(chromosome)) < mutation_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]

    # Ensure minimum wavelengths
    if np.sum(mutated) < min_wavelengths:
        # Add random wavelengths
        zeros = np.where(mutated == 0)[0]
        if len(zeros) > 0:
            n_add = min(min_wavelengths - np.sum(mutated), len(zeros))
            to_add = rng.choice(zeros, size=int(n_add), replace=False)
            mutated[to_add] = 1

    return mutated


def _run_single_ga(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    population_size: int,
    n_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    cv_folds: int,
    min_wavelengths: int,
    n_components: int,
    early_stopping: int,
    random_state: int,
    progress_callback: Optional[Callable] = None,
    run_idx: int = 0,
    n_runs: int = 1,
    n_jobs: int = -1
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run a single GA optimization.

    Returns
    -------
    best_chromosome : np.ndarray
        Best solution found
    best_fitness : float
        Best fitness achieved
    history : list
        Fitness history over generations
    """
    rng = np.random.RandomState(random_state)
    n_wavelengths = X.shape[1]

    # Set up cross-validation
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Initialize population
    population = _initialize_population(
        n_wavelengths, population_size,
        init_ratio=0.3, min_wavelengths=min_wavelengths, rng=rng
    )

    # Fitness cache
    cache = FitnessCache()

    # Track best solution
    best_chromosome = None
    best_fitness = -np.inf
    history = []
    generations_without_improvement = 0

    for gen in range(n_generations):
        # Evaluate fitness for all individuals (with parallel support)
        fitness_scores = np.zeros(population_size)

        # First, check cache for all individuals
        to_compute = []
        for i in range(population_size):
            cached = cache.get(population[i])
            if cached is not None:
                fitness_scores[i] = cached
            else:
                to_compute.append(i)

        # Parallel evaluation of non-cached individuals
        if to_compute:
            if n_jobs == 1 or len(to_compute) == 1:
                # Sequential fallback
                for i in to_compute:
                    fitness = _fitness_function(
                        population[i], X, y, cv, task_type, n_components, min_wavelengths
                    )
                    cache.set(population[i], fitness)
                    fitness_scores[i] = fitness
            else:
                # Parallel evaluation - use threading in frozen apps
                from spectral_predict.search import _frozen_needs_threading_fallback
                backend = 'threading' if _frozen_needs_threading_fallback() else 'loky'
                results = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_fitness_function)(
                        population[i], X, y, cv, task_type, n_components, min_wavelengths
                    )
                    for i in to_compute
                )
                # Update cache and fitness scores
                for i, fitness in zip(to_compute, results):
                    cache.set(population[i], fitness)
                    fitness_scores[i] = fitness

        # Track best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        history.append(best_fitness)

        # Progress callback - send in format GUI expects
        if progress_callback is not None:
            n_sel = np.sum(best_chromosome) if best_chromosome is not None else 0
            # Calculate overall progress across all runs and generations
            overall_current = run_idx * n_generations + (gen + 1)
            overall_total = n_runs * n_generations
            fitness_str = f"{-best_fitness:.4f}" if task_type == 'regression' else f"{best_fitness:.4f}"
            progress_callback({
                'algorithm': 'ga_pls',
                'generation': gen + 1,
                'total_generations': n_generations,
                'best_fitness': best_fitness,
                'n_selected': n_sel,
                # GUI-compatible keys
                'current': overall_current,
                'total': overall_total,
                'message': f"GA-PLS Run {run_idx + 1}/{n_runs}, Gen {gen + 1}/{n_generations}: RMSECV={fitness_str}, {n_sel} vars"
            })

        # Early stopping
        if generations_without_improvement >= early_stopping:
            break

        # Selection
        selected = _tournament_selection(population, fitness_scores, tournament_size=3, rng=rng)

        # Crossover and mutation
        new_population = []

        # Elitism: keep best 2 individuals
        elite_indices = np.argsort(fitness_scores)[-2:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate rest through crossover + mutation
        while len(new_population) < population_size:
            i, j = rng.choice(population_size, size=2, replace=False)
            child1, child2 = _two_point_crossover(
                selected[i], selected[j], crossover_rate, rng
            )
            child1 = _bit_flip_mutation(child1, mutation_rate, min_wavelengths, rng)
            child2 = _bit_flip_mutation(child2, mutation_rate, min_wavelengths, rng)
            new_population.extend([child1, child2])

        population = np.array(new_population[:population_size])

    return best_chromosome, best_fitness, history


# =============================================================================
# Public API
# =============================================================================

def ga_pls_selection(
    X: np.ndarray,
    y: np.ndarray,
    # GA parameters
    population_size: int = DEFAULT_POPULATION_SIZE,
    n_generations: int = DEFAULT_GENERATIONS,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    # Multi-run parameters
    n_runs: int = DEFAULT_N_RUNS,
    selection_threshold: float = DEFAULT_SELECTION_THRESHOLD,
    # Validation parameters
    enable_permutation_test: bool = False,
    n_permutations: int = 100,
    # Model parameters
    n_components: int = 10,
    cv: int = 5,
    min_wavelengths: int = DEFAULT_MIN_WAVELENGTHS,
    task_type: str = 'regression',
    # Control parameters
    early_stopping: int = DEFAULT_EARLY_STOPPING,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None
) -> np.ndarray:
    """
    GA-PLS wavelength selection using Genetic Algorithm optimization.

    Returns importance scores like other variable selection methods.
    Higher score = more important (more frequently selected across GA runs).

    Parameters
    ----------
    X : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    y : np.ndarray
        Target values
    population_size : int, default=64
        GA population size. Recommended: 50-100 for spectral data
    n_generations : int, default=100
        Maximum GA generations. Typically converges in 50-100
    crossover_rate : float, default=0.6
        Probability of crossover (40-60% typical for GA-PLS)
    mutation_rate : float, default=0.01
        Per-gene mutation probability (0.8-1.5% typical)
    n_runs : int, default=5
        Number of independent GA runs for stability
    selection_threshold : float, default=0.6
        Minimum selection frequency for stable variables
    enable_permutation_test : bool, default=False
        Run permutation test after GA (adds computational cost)
    n_permutations : int, default=100
        Number of permutations for statistical test
    n_components : int, default=10
        Maximum PLS components for fitness evaluation
    cv : int, default=5
        Cross-validation folds for fitness evaluation
    min_wavelengths : int, default=5
        Minimum number of wavelengths to select
    task_type : str, default='regression'
        'regression' or 'classification'
    early_stopping : int, default=20
        Stop if no improvement for this many generations
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=1
        Parallel jobs for multi-run. Use 1 for reproducibility
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Function called with progress updates

    Returns
    -------
    importances : np.ndarray
        Selection frequency scores for each wavelength (0.0 to 1.0)
        Higher = more important, more consistently selected
        Shape: (n_wavelengths,)

    Notes
    -----
    GA-PLS is computationally expensive. For 2000 wavelengths:
    - Single run: ~2-5 minutes (depending on population/generations)
    - Multi-run (5 runs): ~10-25 minutes

    Consider using 'quick' settings for initial exploration:
    - population_size=32, n_generations=50, n_runs=3

    References
    ----------
    Leardi, R. & Lupiáñez González, A. (1998). Genetic algorithms applied to
    feature selection in PLS regression. Chemometrics and Intelligent
    Laboratory Systems, 41(2), 195-207.

    Examples
    --------
    >>> # Basic usage
    >>> importances = ga_pls_selection(X, y)
    >>> top_wavelengths = np.argsort(importances)[-100:]  # Top 100

    >>> # With validation
    >>> importances = ga_pls_selection(
    ...     X, y,
    ...     enable_permutation_test=True,
    ...     n_runs=10,
    ...     selection_threshold=0.7
    ... )
    """
    # Convert inputs. Shape-aware (T-17): a 1-D or single-column target ravels
    # to (n,) -- byte-identical to the pre-T-17 path -- while a genuine
    # multi-target block (n, n_targets>=2) is kept 2-D so the JOINT-PLS
    # multi-Y fitness branch is taken. GA structure/RNG is y-independent, so
    # only the fitness criterion changes for multi-target.
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        y = y.ravel()

    n_samples, n_wavelengths = X.shape

    cv_folds = cv

    # Validate inputs
    if X.shape[0] != len(y):
        raise ValueError("X and y must have same number of samples")

    # Adjust parameters for small datasets
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        if verbose > 0:
            print(f"Warning: Reduced cv_folds to {cv_folds} due to small sample size")

    if verbose > 0:
        print(f"GA-PLS: {n_wavelengths} wavelengths, {n_runs} runs, "
              f"pop={population_size}, gen={n_generations}")

    # Run multiple GA optimizations
    all_chromosomes = []
    all_fitness = []

    # Determine parallelization strategy for runs
    import os
    from spectral_predict.search import _frozen_needs_threading_fallback
    run_backend = 'threading' if _frozen_needs_threading_fallback() else 'loky'
    n_cores = os.cpu_count() or 1

    # Parallelize runs when we have multiple runs and cores available.
    # When runs are parallel, each run uses n_jobs=1 internally to avoid
    # CPU over-subscription. When sequential, each run gets full n_jobs.
    parallel_runs = n_runs > 1 and n_cores >= 2 and n_jobs != 1 and progress_callback is None
    inner_n_jobs = 1 if parallel_runs else n_jobs

    if parallel_runs and verbose > 0:
        print(f"  Running {n_runs} GA runs in parallel ({min(n_runs, n_cores)} workers)")

    def _execute_single_run(run_idx: int) -> tuple:
        """Execute a single GA run and return (best_chrom, best_fit, history)."""
        run_seed = random_state + run_idx * 1000
        if verbose > 0 and not parallel_runs:
            print(f"\n  Run {run_idx + 1}/{n_runs}...")

        return _run_single_ga(
            X, y, task_type,
            population_size, n_generations,
            crossover_rate, mutation_rate,
            cv_folds, min_wavelengths, n_components,
            early_stopping, run_seed,
            progress_callback,
            run_idx, n_runs,
            inner_n_jobs
        )

    if parallel_runs:
        # Parallel execution of independent GA runs
        run_results = Parallel(n_jobs=min(n_runs, n_cores), backend=run_backend)(
            delayed(_execute_single_run)(run_idx) for run_idx in range(n_runs)
        )
    else:
        # Sequential execution (single run or user requested n_jobs=1)
        run_results = [_execute_single_run(run_idx) for run_idx in range(n_runs)]

    for run_idx, (best_chrom, best_fit, history) in enumerate(run_results):
        if best_chrom is not None:
            all_chromosomes.append(best_chrom)
            all_fitness.append(best_fit)

            if verbose > 0:
                n_selected = np.sum(best_chrom)
                if task_type == 'regression':
                    print(f"    Run {run_idx + 1}: Best RMSECV: {-best_fit:.4f}, {n_selected} wavelengths")
                else:
                    print(f"    Run {run_idx + 1}: Best Accuracy: {best_fit:.4f}, {n_selected} wavelengths")

    if len(all_chromosomes) == 0:
        if verbose > 0:
            print("Warning: All GA runs failed. Returning uniform importances.")
        return np.ones(n_wavelengths)

    # Compute selection frequency across runs
    selection_counts = np.zeros(n_wavelengths)
    for chrom in all_chromosomes:
        selection_counts += chrom

    selection_frequency = selection_counts / len(all_chromosomes)

    # Report summary
    if verbose > 0:
        n_stable = np.sum(selection_frequency >= selection_threshold)
        print(f"\nGA-PLS Summary:")
        print(f"  Successful runs: {len(all_chromosomes)}/{n_runs}")
        print(f"  Stable wavelengths (freq >= {selection_threshold}): {n_stable}")
        print(f"  Best fitness across runs: {max(all_fitness):.4f}")

    # Optional: Permutation test (single-target only; multi-Y permutation
    # significance is deferred with the multi-Y varsel reporting work).
    if enable_permutation_test and verbose > 0 and np.asarray(y).ndim == 1:
        print(f"\nRunning permutation test ({n_permutations} permutations)...")
        p_value = _permutation_test(
            X, y, selection_frequency, n_permutations,
            cv_folds, n_components, random_state
        )
        print(f"  Permutation test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Selection is statistically significant (p < 0.05)")
        else:
            print("  Warning: Selection may not be significant (p >= 0.05)")

    return selection_frequency


def ga_pls_selection_detailed(
    X: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    GA-PLS with detailed results for analysis.

    Parameters
    ----------
    X : np.ndarray
        Spectral data
    y : np.ndarray
        Target values
    **kwargs
        All parameters from ga_pls_selection

    Returns
    -------
    results : dict
        - 'importances': np.ndarray - Selection frequency scores
        - 'selected_indices': np.ndarray - Indices of stable selections
        - 'best_fitness': float - Best fitness achieved
        - 'n_runs_successful': int - Number of successful runs
        - 'selection_threshold': float - Threshold used
    """
    selection_threshold = kwargs.get('selection_threshold', DEFAULT_SELECTION_THRESHOLD)

    importances = ga_pls_selection(X, y, **kwargs)

    selected_indices = np.where(importances >= selection_threshold)[0]

    return {
        'importances': importances,
        'selected_indices': selected_indices,
        'selection_threshold': selection_threshold,
        'n_selected': len(selected_indices)
    }


def _permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    selection_frequency: np.ndarray,
    n_permutations: int,
    cv_folds: int,
    n_components: int,
    random_state: int
) -> float:
    """
    Run permutation test to validate GA-PLS selection.

    Returns p-value: probability that selection is due to chance.
    """
    rng = np.random.RandomState(random_state)

    # Get indices of top selected wavelengths
    top_k = max(10, int(np.sum(selection_frequency >= 0.5)))
    selected_indices = np.argsort(selection_frequency)[-top_k:]

    # Real score
    X_selected = X[:, selected_indices]
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    try:
        pls = PLSRegression(n_components=min(n_components, top_k // 2), scale=False)
        y_pred = cross_val_predict(pls, X_selected, y, cv=cv)
        real_rmsecv = np.sqrt(mean_squared_error(y, y_pred))
    except Exception:
        return 1.0  # Cannot compute

    # Null distribution
    null_scores = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            y_pred_perm = cross_val_predict(pls, X_selected, y_perm, cv=cv)
            null_rmsecv = np.sqrt(mean_squared_error(y_perm, y_pred_perm))
            null_scores.append(null_rmsecv)
        except Exception:
            continue

    if len(null_scores) == 0:
        return 1.0

    # p-value: proportion of null scores better than real
    p_value = np.mean(np.array(null_scores) <= real_rmsecv)

    return p_value
