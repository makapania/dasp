"""
Genetic Algorithm for Preprocessing Optimization (v3 standalone).

This module implements a genetic algorithm to optimize spectral preprocessing
parameters including:
- Preprocessing type (raw, SNV, derivatives, combinations)
- Savitzky-Golay window size
- Baseline correction method and parameters
- Smoothing settings

The GA evaluates preprocessing configurations using cross-validated RMSECV
with a quick PLS model, then returns the optimal preprocessing transform.

References
----------
- Stefansson, A., et al. (2020). "Fast method for GA-PLS."
  Journal of Chemometrics.
- Studies show 30-110% improvement in RMSECV over manual preprocessing selection.
"""

import numpy as np
from typing import Tuple, Callable, Optional, List, Dict, Any
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score

# Import preprocessing transformers
from .preprocess import SNV, SavgolDerivative
from .baseline import BaselinePolynomial, BaselineAsLS, BaselineAirPLS, SavgolSmooth


# =============================================================================
# CHROMOSOME ENCODING
# =============================================================================

# Gene 0: Preprocessing type
PREPROC_TYPES = [
    'raw',           # 0
    'snv',           # 1
    'deriv1',        # 2
    'deriv2',        # 3
    'snv_deriv1',    # 4
    'snv_deriv2',    # 5
    'deriv1_snv',    # 6
    'deriv2_snv',    # 7
    'deriv3',        # 8
    'deriv4',        # 9
]

# Gene 1: S-G window sizes (odd values only)
WINDOW_SIZES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 51]

# Gene 2: Baseline methods
BASELINE_METHODS = [
    'none',          # 0
    'polynomial',    # 1
    'asls',          # 2
    'airpls',        # 3
]

# Gene 3: Baseline lambda (log scale)
BASELINE_LAMBDAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

# Gene 4: Smoothing enabled
SMOOTHING_OPTIONS = [False, True]

# Gene 5: Smoothing window
SMOOTHING_WINDOWS = [5, 7, 9, 11, 13, 15, 17, 19, 21]

# Chromosome structure: [preproc_type, window, baseline, baseline_lambda, smoothing, smooth_window]
N_GENES = 6


def random_chromosome(rng: np.random.RandomState) -> np.ndarray:
    """Generate a random chromosome."""
    return np.array([
        rng.randint(0, len(PREPROC_TYPES)),
        rng.randint(0, len(WINDOW_SIZES)),
        rng.randint(0, len(BASELINE_METHODS)),
        rng.randint(0, len(BASELINE_LAMBDAS)),
        rng.randint(0, len(SMOOTHING_OPTIONS)),
        rng.randint(0, len(SMOOTHING_WINDOWS)),
    ], dtype=np.int32)


def chromosome_to_transform(genes: np.ndarray) -> Tuple[str, Optional[Callable]]:
    """
    Convert chromosome to (name, transform_func) tuple.

    The returned tuple is compatible with _build_preprocessing_configs() output
    in search.py, making integration seamless.

    Parameters
    ----------
    genes : np.ndarray
        Integer-encoded chromosome [preproc_type, window, baseline, lambda, smoothing, smooth_window]

    Returns
    -------
    name : str
        Human-readable name for the preprocessing configuration
    transform_func : callable or None
        Function that takes X and returns preprocessed X, or None for 'raw'
    """
    preproc_idx = genes[0]
    window_idx = genes[1]
    baseline_idx = genes[2]
    lambda_idx = genes[3]
    smoothing_enabled = SMOOTHING_OPTIONS[genes[4]]
    smooth_window_idx = genes[5]

    preproc_type = PREPROC_TYPES[preproc_idx]
    window = WINDOW_SIZES[window_idx]
    baseline_method = BASELINE_METHODS[baseline_idx]
    baseline_lambda = BASELINE_LAMBDAS[lambda_idx]
    smooth_window = SMOOTHING_WINDOWS[smooth_window_idx]

    # Build name
    name_parts = [preproc_type]
    if preproc_type not in ['raw', 'snv']:
        name_parts.append(f'w{window}')
    if baseline_method != 'none':
        name_parts.append(f'{baseline_method}')
    if smoothing_enabled:
        name_parts.append(f'smooth{smooth_window}')

    name = '_'.join(name_parts)

    # Build transform function
    def transform(X, pt=preproc_type, w=window, bl=baseline_method,
                  bl_lam=baseline_lambda, sm=smoothing_enabled, sm_w=smooth_window):
        X_out = np.asarray(X, dtype=np.float64)

        # Apply smoothing first (if enabled)
        if sm:
            smoother = SavgolSmooth(window_length=sm_w, polyorder=2)
            X_out = smoother.fit_transform(X_out)

        # Apply baseline correction (before main preprocessing)
        if bl == 'polynomial':
            baseline = BaselinePolynomial(degree=2, n_segments=20)
            X_out = baseline.fit_transform(X_out)
        elif bl == 'asls':
            baseline = BaselineAsLS(lam=bl_lam, p=0.01, max_iter=10)
            X_out = baseline.fit_transform(X_out)
        elif bl == 'airpls':
            baseline = BaselineAirPLS(lam=bl_lam, max_iter=15)
            X_out = baseline.fit_transform(X_out)

        # Apply main preprocessing
        if pt == 'raw':
            pass
        elif pt == 'snv':
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv1':
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == 'deriv2':
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == 'deriv3':
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == 'deriv4':
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == 'snv_deriv1':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
        elif pt == 'snv_deriv2':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
        elif pt == 'deriv1_snv':
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv2_snv':
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)

        return X_out

    if preproc_type == 'raw' and baseline_method == 'none' and not smoothing_enabled:
        return (name, None)

    return (name, transform)


def get_config_description(genes: np.ndarray) -> str:
    """Get human-readable description of chromosome configuration."""
    preproc_type = PREPROC_TYPES[genes[0]]
    window = WINDOW_SIZES[genes[1]]
    baseline_method = BASELINE_METHODS[genes[2]]
    baseline_lambda = BASELINE_LAMBDAS[genes[3]]
    smoothing_enabled = SMOOTHING_OPTIONS[genes[4]]
    smooth_window = SMOOTHING_WINDOWS[genes[5]]

    parts = [f"Preproc: {preproc_type}"]
    if preproc_type not in ['raw', 'snv']:
        parts.append(f"Window: {window}")
    if baseline_method != 'none':
        parts.append(f"Baseline: {baseline_method} (lam={baseline_lambda:.0e})")
    if smoothing_enabled:
        parts.append(f"Smooth: w{smooth_window}")

    return ", ".join(parts)


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

def evaluate_fitness(
    genes: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    random_state: int = 42
) -> float:
    """
    Evaluate fitness of a preprocessing configuration.

    Uses cross-validated RMSECV (regression) or accuracy (classification) as fitness.

    Parameters
    ----------
    genes : np.ndarray
        Chromosome encoding preprocessing configuration
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds
    n_components : int
        Max PLS components to test
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random state for CV splitting

    Returns
    -------
    fitness : float
        For regression: negative RMSECV (higher = better)
        For classification: accuracy (higher = better)
    """
    try:
        # Get transform function
        name, transform_func = chromosome_to_transform(genes)

        # Apply preprocessing
        if transform_func is not None:
            X_preproc = transform_func(X)
        else:
            X_preproc = X

        # Check for invalid values
        if not np.isfinite(X_preproc).all():
            return -np.inf

        # Check for zero variance columns
        if np.any(np.std(X_preproc, axis=0) < 1e-10):
            return -np.inf

        # Determine n_components
        n_samples, n_features = X_preproc.shape
        n_comp = min(n_components, n_features // 2, n_samples // 2)
        n_comp = max(1, n_comp)

        # Set up CV based on task type
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Fit PLS with cross-validation
        pls = PLSRegression(n_components=n_comp, scale=False)
        y_pred = cross_val_predict(pls, X_preproc, y, cv=cv)

        if task_type == 'classification':
            # For classification, threshold predictions and compute accuracy
            y_class = np.asarray(y)
            if y_class.dtype == object or not np.issubdtype(y_class.dtype, np.number):
                # Encode string labels
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_class = le.fit_transform(y_class)
            y_pred_class = (y_pred > np.median(y_pred)).astype(int).ravel()
            return accuracy_score(y_class, y_pred_class)
        else:
            # Calculate RMSECV
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            # Return negative RMSECV (we maximize fitness)
            return -rmsecv

    except Exception as e:
        # Any error = very poor fitness
        return -np.inf


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Select parent using tournament selection."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = fitness[indices]
    winner_idx = indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    child1 = parent1.copy()
    child2 = parent2.copy()

    # Uniform crossover - swap each gene independently
    for i in range(len(parent1)):
        if rng.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]

    return child1, child2


def mutate(
    chromosome: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Mutate chromosome with given mutation rate per gene."""
    mutated = chromosome.copy()

    gene_ranges = [
        len(PREPROC_TYPES),
        len(WINDOW_SIZES),
        len(BASELINE_METHODS),
        len(BASELINE_LAMBDAS),
        len(SMOOTHING_OPTIONS),
        len(SMOOTHING_WINDOWS),
    ]

    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] = rng.randint(0, gene_ranges[i])

    return mutated


# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def optimize_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 32,
    n_generations: int = 50,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    cv_folds: int = 5,
    n_components: int = 10,
    elitism: int = 2,
    task_type: str = 'regression',
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Optimize spectral preprocessing using genetic algorithm.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    population_size : int
        Number of individuals in population
    n_generations : int
        Number of generations to evolve
    crossover_rate : float
        Probability of crossover (0-1)
    mutation_rate : float
        Probability of mutation per gene (0-1)
    tournament_size : int
        Number of individuals in tournament selection
    cv_folds : int
        Number of CV folds for fitness evaluation
    n_components : int
        Max PLS components for evaluation
    elitism : int
        Number of best individuals to preserve each generation
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Function called with dict containing:
        - 'algorithm': 'ga_preprocessing'
        - 'generation': current generation
        - 'total_generations': total generations
        - 'best_fitness': best fitness
        - 'message': human-readable status

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'best_genes': np.ndarray - Best chromosome
        - 'best_name': str - Name of best preprocessing
        - 'best_transform': callable - Transform function
        - 'best_rmsecv': float - Best RMSECV (regression) or 1-accuracy (classification)
        - 'best_config': str - Human-readable configuration
        - 'history': list - Fitness history per generation
    """
    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if verbose >= 1:
        print(f"GA Preprocessing Optimization")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Task: {task_type}")
        print(f"  Population: {population_size}, Generations: {n_generations}")
        print(f"  CV folds: {cv_folds}, PLS components: {n_components}")

    # Initialize population
    population = np.array([random_chromosome(rng) for _ in range(population_size)])

    # Evaluate initial fitness
    fitness = np.array([
        evaluate_fitness(ind, X, y, cv_folds, n_components, task_type, random_state)
        for ind in population
    ])

    # Track best
    best_idx = np.argmax(fitness)
    best_genes = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    history = [{'generation': 0, 'best_fitness': best_fitness, 'mean_fitness': np.mean(fitness[fitness > -np.inf])}]

    # Score label depends on task type
    score_label = "Accuracy" if task_type == 'classification' else "RMSECV"

    if verbose >= 1:
        if task_type == 'classification':
            print(f"  Gen 0: Best Accuracy = {best_fitness:.4f}")
        else:
            print(f"  Gen 0: Best RMSECV = {-best_fitness:.4f}")

    # Evolution loop
    for gen in range(1, n_generations + 1):
        # Create new population
        new_population = []

        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness)[-elitism:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Fill rest with offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness, tournament_size, rng)
            parent2 = tournament_selection(population, fitness, tournament_size, rng)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate, rng)

            # Mutation
            child1 = mutate(child1, mutation_rate, rng)
            child2 = mutate(child2, mutation_rate, rng)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population[:population_size])

        # Evaluate fitness
        fitness = np.array([
            evaluate_fitness(ind, X, y, cv_folds, n_components, task_type, random_state)
            for ind in population
        ])

        # Update best
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_genes = population[gen_best_idx].copy()
            best_fitness = fitness[gen_best_idx]

        valid_fitness = fitness[fitness > -np.inf]
        mean_fit = np.mean(valid_fitness) if len(valid_fitness) > 0 else -np.inf
        history.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fit
        })

        if verbose >= 1 and gen % 10 == 0:
            if task_type == 'classification':
                print(f"  Gen {gen}: Best Accuracy = {best_fitness:.4f}")
            else:
                print(f"  Gen {gen}: Best RMSECV = {-best_fitness:.4f}")

        if progress_callback:
            if task_type == 'classification':
                score_str = f"Accuracy={best_fitness:.4f}"
            else:
                score_str = f"RMSECV={-best_fitness:.4f}"
            progress_callback({
                'algorithm': 'ga_preprocessing',
                'generation': gen,
                'total_generations': n_generations,
                'best_fitness': best_fitness,
                'message': f"Gen {gen}/{n_generations}: {score_str} ({get_config_description(best_genes)})"
            })

    # Get final result
    best_name, best_transform = chromosome_to_transform(best_genes)
    best_config = get_config_description(best_genes)

    if verbose >= 1:
        print(f"\nOptimization complete!")
        if task_type == 'classification':
            print(f"  Best Accuracy: {best_fitness:.4f}")
        else:
            print(f"  Best RMSECV: {-best_fitness:.4f}")
        print(f"  Best config: {best_config}")

    # Return score in format expected by callers
    # For regression: positive RMSECV, for classification: 1 - accuracy (to match error metric)
    if task_type == 'classification':
        best_score = 1.0 - best_fitness  # Convert accuracy to error
    else:
        best_score = -best_fitness  # Convert negative RMSECV to positive

    return {
        'best_genes': best_genes,
        'best_name': best_name,
        'best_transform': best_transform,
        'best_rmsecv': best_score,  # Kept as 'best_rmsecv' for backward compatibility
        'best_config': best_config,
        'history': history,
        'task_type': task_type
    }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_optimized_preproc_config(
    X: np.ndarray,
    y: np.ndarray,
    quick: bool = True,
    random_state: int = 42,
    verbose: int = 1
) -> Tuple[str, Optional[Callable]]:
    """
    Get optimized preprocessing as (name, transform_func) tuple.

    This is a convenience function that returns the result in the same
    format as _build_preprocessing_configs() for easy integration.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data
    y : np.ndarray
        Target values
    quick : bool
        If True, use quick settings (fewer generations)
    random_state : int
        Random seed
    verbose : int
        Verbosity level

    Returns
    -------
    name : str
        Preprocessing configuration name
    transform_func : callable or None
        Transform function, or None for 'raw'
    """
    if quick:
        result = optimize_preprocessing(
            X, y,
            population_size=16,
            n_generations=25,
            cv_folds=3,
            random_state=random_state,
            verbose=verbose
        )
    else:
        result = optimize_preprocessing(
            X, y,
            population_size=32,
            n_generations=50,
            cv_folds=5,
            random_state=random_state,
            verbose=verbose
        )

    return (result['best_name'], result['best_transform'])
