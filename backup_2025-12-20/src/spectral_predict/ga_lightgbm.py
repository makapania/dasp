"""
GA-LightGBM (Genetic Algorithm - LightGBM) wavelength selection for tree-based models.

This module implements GA-based variable selection using LightGBM for fitness
evaluation, designed for tree-based models (LightGBM, XGBoost, RandomForest, CatBoost).

Complements GA-PLS which is optimized for linear models (PLS, Ridge, Lasso, etc.).

The algorithm uses a binary chromosome where each gene represents whether
a wavelength is selected (1) or excluded (0). Multiple GA runs are aggregated
to produce stable importance scores based on selection frequency.

Key differences from GA-PLS:
- Uses LightGBM instead of PLS for fitness evaluation
- Faster for high-dimensional data (no matrix decomposition)
- Native classification support (no PLS-DA workaround needed)
- Tree-based models may select different wavelengths than linear models

References
----------
Leardi, R. & Lupiáñez González, A. (1998). Genetic algorithms applied to
feature selection in PLS regression. Chemometrics and Intelligent
Laboratory Systems, 41(2), 195-207.

Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting
Decision Tree. Advances in Neural Information Processing Systems.
"""

import numpy as np
import threading
from typing import Optional, Callable, Dict, Any, Tuple, List
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed

# Optional LightGBM import
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LGBMRegressor = None
    LGBMClassifier = None


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

# LightGBM defaults
DEFAULT_N_ESTIMATORS = 50
DEFAULT_NUM_LEAVES_REGRESSION = 31
DEFAULT_NUM_LEAVES_CLASSIFICATION = 15
DEFAULT_REG_LAMBDA = 0.1
DEFAULT_LEARNING_RATE = 0.1


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


def _fitness_function_lgbm(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    task_type: str = 'regression',
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    num_leaves: int = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    min_wavelengths: int = DEFAULT_MIN_WAVELENGTHS
) -> float:
    """
    Evaluate chromosome fitness using cross-validated LightGBM.

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
    n_estimators : int
        Number of boosting iterations
    num_leaves : int
        Maximum tree leaves (default: 31 for regression, 15 for classification)
    learning_rate : float
        Boosting learning rate
    reg_lambda : float
        L2 regularization parameter
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

    # Set default num_leaves based on task type
    if num_leaves is None:
        num_leaves = DEFAULT_NUM_LEAVES_CLASSIFICATION if task_type == 'classification' else DEFAULT_NUM_LEAVES_REGRESSION

    try:
        if task_type == 'regression':
            model = LGBMRegressor(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                reg_lambda=reg_lambda,
                verbose=-1,
                n_jobs=1,
                force_col_wise=True  # Better for high-dim data
            )
            scores = cross_val_score(
                model, X_selected, y, cv=cv,
                scoring='neg_mean_squared_error'
            )
            rmsecv = np.sqrt(-np.mean(scores))
            return -rmsecv  # Negative because GA maximizes
        else:
            model = LGBMClassifier(
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                reg_lambda=reg_lambda,
                verbose=-1,
                n_jobs=1,
                force_col_wise=True
            )
            scores = cross_val_score(
                model, X_selected, y, cv=cv,
                scoring='accuracy'
            )
            return np.mean(scores)

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


def _run_single_ga_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    population_size: int,
    n_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    cv_folds: int,
    min_wavelengths: int,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    reg_lambda: float,
    early_stopping: int,
    random_state: int,
    progress_callback: Optional[Callable] = None,
    run_idx: int = 0,
    n_runs: int = 1,
    n_jobs: int = -1
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run a single GA optimization with LightGBM fitness.

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
                    fitness = _fitness_function_lgbm(
                        population[i], X, y, cv, task_type,
                        n_estimators, num_leaves, learning_rate, reg_lambda,
                        min_wavelengths
                    )
                    cache.set(population[i], fitness)
                    fitness_scores[i] = fitness
            else:
                # Parallel evaluation
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fitness_function_lgbm)(
                        population[i], X, y, cv, task_type,
                        n_estimators, num_leaves, learning_rate, reg_lambda,
                        min_wavelengths
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
                'algorithm': 'ga_lightgbm',
                'generation': gen + 1,
                'total_generations': n_generations,
                'best_fitness': best_fitness,
                'n_selected': n_sel,
                # GUI-compatible keys
                'current': overall_current,
                'total': overall_total,
                'message': f"GA-LightGBM Run {run_idx + 1}/{n_runs}, Gen {gen + 1}/{n_generations}: RMSECV={fitness_str}, {n_sel} vars"
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

def ga_lightgbm_selection(
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
    # LightGBM parameters
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    num_leaves: int = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    # Validation parameters
    cv_folds: int = 5,
    min_wavelengths: int = DEFAULT_MIN_WAVELENGTHS,
    task_type: str = 'regression',
    # Control parameters
    early_stopping: int = DEFAULT_EARLY_STOPPING,
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    n_jobs: int = -1
) -> np.ndarray:
    """
    GA-LightGBM wavelength selection using Genetic Algorithm optimization.

    Returns importance scores like other variable selection methods.
    Higher score = more important (more frequently selected across GA runs).

    This method is designed for tree-based models (LightGBM, XGBoost, RandomForest,
    CatBoost). For linear models (PLS, Ridge, Lasso), use ga_pls_selection instead.

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
        Probability of crossover (40-60% typical)
    mutation_rate : float, default=0.01
        Per-gene mutation probability (0.8-1.5% typical)
    n_runs : int, default=5
        Number of independent GA runs for stability
    selection_threshold : float, default=0.6
        Minimum selection frequency for stable variables
    n_estimators : int, default=50
        Number of LightGBM boosting iterations
    num_leaves : int, default=None
        Maximum tree leaves. If None, uses 31 for regression, 15 for classification
    learning_rate : float, default=0.1
        LightGBM learning rate
    reg_lambda : float, default=0.1
        L2 regularization parameter for LightGBM
    cv_folds : int, default=5
        Cross-validation folds for fitness evaluation
    min_wavelengths : int, default=5
        Minimum number of wavelengths to select
    task_type : str, default='regression'
        'regression' or 'classification'
    early_stopping : int, default=20
        Stop if no improvement for this many generations
    random_state : int, default=42
        Random seed for reproducibility
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

    Raises
    ------
    ImportError
        If LightGBM is not installed

    Notes
    -----
    GA-LightGBM is computationally efficient due to LightGBM's speed.
    For 2000 wavelengths:
    - Single run: ~1-3 minutes
    - Multi-run (5 runs): ~5-15 minutes

    Consider using 'quick' settings for initial exploration:
    - population_size=32, n_generations=50, n_runs=3

    Examples
    --------
    >>> # Basic usage for tree-based models
    >>> importances = ga_lightgbm_selection(X, y)
    >>> top_wavelengths = np.argsort(importances)[-100:]  # Top 100

    >>> # Quick search
    >>> importances = ga_lightgbm_selection(
    ...     X, y,
    ...     population_size=32,
    ...     n_generations=50,
    ...     n_runs=3
    ... )
    """
    if not HAS_LIGHTGBM:
        raise ImportError(
            "LightGBM is required for GA-LightGBM selection. "
            "Install with: pip install lightgbm"
        )

    # Convert inputs
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples, n_wavelengths = X.shape

    # Validate inputs
    if X.shape[0] != len(y):
        raise ValueError("X and y must have same number of samples")

    # Adjust parameters for small datasets
    if n_samples < cv_folds:
        cv_folds = max(2, n_samples // 2)
        if verbose > 0:
            print(f"Warning: Reduced cv_folds to {cv_folds} due to small sample size")

    # Set default num_leaves based on task type
    if num_leaves is None:
        num_leaves = DEFAULT_NUM_LEAVES_CLASSIFICATION if task_type == 'classification' else DEFAULT_NUM_LEAVES_REGRESSION

    if verbose > 0:
        print(f"GA-LightGBM: {n_wavelengths} wavelengths, {n_runs} runs, "
              f"pop={population_size}, gen={n_generations}")
        print(f"  LightGBM: n_estimators={n_estimators}, num_leaves={num_leaves}, "
              f"lr={learning_rate}, reg_lambda={reg_lambda}")

    # Run multiple GA optimizations
    all_chromosomes = []
    all_fitness = []

    for run_idx in range(n_runs):
        run_seed = random_state + run_idx * 1000

        if verbose > 0:
            print(f"\n  Run {run_idx + 1}/{n_runs}...")

        best_chrom, best_fit, history = _run_single_ga_lgbm(
            X, y, task_type,
            population_size, n_generations,
            crossover_rate, mutation_rate,
            cv_folds, min_wavelengths,
            n_estimators, num_leaves, learning_rate, reg_lambda,
            early_stopping, run_seed,
            progress_callback,  # Pass to all runs for continuous progress
            run_idx, n_runs,  # Pass run context for overall progress calculation
            n_jobs  # Parallelization
        )

        if best_chrom is not None:
            all_chromosomes.append(best_chrom)
            all_fitness.append(best_fit)

            if verbose > 0:
                n_selected = np.sum(best_chrom)
                if task_type == 'regression':
                    print(f"    Best RMSECV: {-best_fit:.4f}, {n_selected} wavelengths")
                else:
                    print(f"    Best Accuracy: {best_fit:.4f}, {n_selected} wavelengths")

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
        print(f"\nGA-LightGBM Summary:")
        print(f"  Successful runs: {len(all_chromosomes)}/{n_runs}")
        print(f"  Stable wavelengths (freq >= {selection_threshold}): {n_stable}")
        print(f"  Best fitness across runs: {max(all_fitness):.4f}")

    return selection_frequency


def ga_lightgbm_selection_detailed(
    X: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    GA-LightGBM with detailed results for analysis.

    Parameters
    ----------
    X : np.ndarray
        Spectral data
    y : np.ndarray
        Target values
    **kwargs
        All parameters from ga_lightgbm_selection

    Returns
    -------
    results : dict
        - 'importances': np.ndarray - Selection frequency scores
        - 'selected_indices': np.ndarray - Indices of stable selections
        - 'n_selected': int - Number of selected wavelengths
        - 'selection_threshold': float - Threshold used
    """
    selection_threshold = kwargs.get('selection_threshold', DEFAULT_SELECTION_THRESHOLD)

    importances = ga_lightgbm_selection(X, y, **kwargs)

    selected_indices = np.where(importances >= selection_threshold)[0]

    return {
        'importances': importances,
        'selected_indices': selected_indices,
        'selection_threshold': selection_threshold,
        'n_selected': len(selected_indices)
    }
