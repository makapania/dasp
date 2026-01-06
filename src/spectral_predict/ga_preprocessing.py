"""
Genetic Algorithm for Preprocessing Optimization (V1).

This module implements a genetic algorithm to optimize spectral preprocessing
parameters. The search space is simplified to just 2 genes:
- Preprocessing type (raw, SNV, derivatives, combinations)
- Savitzky-Golay window size

Baseline correction and smoothing are removed as they are redundant when using
derivatives (SG derivatives already smooth, and derivatives remove baselines).

Total search space: 14 preprocessing types × 17 window sizes = 238 combinations

The GA evaluates preprocessing configurations using cross-validated RMSECV
with either PLS or LightGBM models for fitness evaluation.

Fitness Models
--------------
- 'pls': Uses PLS regression (default, always available)
- 'mlp': Uses Multi-Layer Perceptron neural network (for Neural/SVM models)
- 'lightgbm': Uses LightGBM (better for tree-based models, requires LightGBM)
- 'neuralboosted': Uses NeuralBoosted hybrid (requires NeuralBoosted module, falls back to PLS)

References
----------
- Stefansson, A., et al. (2020). "Fast method for GA-PLS."
  Journal of Chemometrics.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Callable, Optional, Dict, Any, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score

# Import V1 preprocessing transformers
from .preprocess import SNV, SavgolDerivative

# Import LightGBM for fitness evaluation (required dependency)
from lightgbm import LGBMRegressor, LGBMClassifier


# =============================================================================
# CHROMOSOME ENCODING (SIMPLIFIED: 2 GENES ONLY)
# =============================================================================

# Gene 0: Preprocessing type (14 options)
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
    'snv_deriv3',    # 10
    'snv_deriv4',    # 11
    'deriv3_snv',    # 12
    'deriv4_snv',    # 13
]

# Gene 1: S-G window sizes (odd values only, 17 options)
WINDOW_SIZES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 51]

# Total search space: 14 × 17 = 238 combinations
N_GENES = 2
TOTAL_COMBINATIONS = len(PREPROC_TYPES) * len(WINDOW_SIZES)  # 238


def random_chromosome(rng: np.random.RandomState) -> np.ndarray:
    """Generate a random chromosome with 2 genes."""
    return np.array([
        rng.randint(0, len(PREPROC_TYPES)),
        rng.randint(0, len(WINDOW_SIZES)),
    ], dtype=np.int32)


def get_seed_chromosomes() -> List[np.ndarray]:
    """
    Return proven good preprocessing configurations as seed chromosomes.

    These seeds ensure the GA population includes common, effective
    preprocessing methods used in spectroscopy.
    """
    seeds = []

    def make_seed(preproc: str, window: int = 17) -> np.ndarray:
        """Helper to create a seed chromosome from named parameters."""
        return np.array([
            PREPROC_TYPES.index(preproc),
            WINDOW_SIZES.index(window) if window in WINDOW_SIZES else 6,  # default idx 6 = 17
        ], dtype=np.int32)

    # Proven good configurations from literature and practice
    seeds.append(make_seed('raw'))                          # Baseline
    seeds.append(make_seed('snv'))                          # Standard scatter correction
    seeds.append(make_seed('deriv1', window=17))            # 1st derivative, common window
    seeds.append(make_seed('deriv2', window=17))            # 2nd derivative, common window
    seeds.append(make_seed('deriv3', window=21))            # 3rd derivative (needs larger window)
    seeds.append(make_seed('deriv4', window=25))            # 4th derivative (needs even larger window)
    seeds.append(make_seed('snv_deriv1', window=17))        # SNV then 1st deriv
    seeds.append(make_seed('snv_deriv2', window=17))        # SNV then 2nd deriv
    seeds.append(make_seed('snv_deriv3', window=21))        # SNV then 3rd deriv
    seeds.append(make_seed('snv_deriv4', window=25))        # SNV then 4th deriv
    seeds.append(make_seed('deriv1_snv', window=17))        # 1st deriv then SNV
    seeds.append(make_seed('deriv2_snv', window=17))        # 2nd deriv then SNV
    seeds.append(make_seed('deriv3_snv', window=21))        # 3rd deriv then SNV
    seeds.append(make_seed('deriv4_snv', window=25))        # 4th deriv then SNV

    return seeds


def chromosome_to_transform(genes: np.ndarray) -> Tuple[str, Optional[Callable]]:
    """
    Convert chromosome to (name, transform_func) tuple.

    The returned tuple is compatible with _build_preprocessing_configs() output
    in search.py, making integration seamless.

    Parameters
    ----------
    genes : np.ndarray
        Integer-encoded chromosome [preproc_type, window]

    Returns
    -------
    name : str
        Human-readable name for the preprocessing configuration
    transform_func : callable or None
        Function that takes X and returns preprocessed X, or None for 'raw'
    """
    preproc_idx = genes[0]
    window_idx = genes[1]

    preproc_type = PREPROC_TYPES[preproc_idx]
    window = WINDOW_SIZES[window_idx]

    # Build name
    if preproc_type in ['raw', 'snv']:
        name = preproc_type
    else:
        name = f"{preproc_type}_w{window}"

    # Return early for raw (no transform needed)
    if preproc_type == 'raw':
        return (name, None)

    # Build transform function
    def transform(X, pt=preproc_type, w=window):
        X_out = np.asarray(X, dtype=np.float64)

        if pt == 'snv':
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
        elif pt == 'snv_deriv3':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
        elif pt == 'snv_deriv4':
            X_out = SNV().fit_transform(X_out)
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
        elif pt == 'deriv1_snv':
            X_out = SavgolDerivative(deriv=1, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv2_snv':
            X_out = SavgolDerivative(deriv=2, window=w).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv3_snv':
            X_out = SavgolDerivative(deriv=3, window=w, polyorder=4).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)
        elif pt == 'deriv4_snv':
            X_out = SavgolDerivative(deriv=4, window=w, polyorder=5).fit_transform(X_out)
            X_out = SNV().fit_transform(X_out)

        return X_out

    return (name, transform)


def get_config_description(genes: np.ndarray) -> str:
    """Get human-readable description of chromosome configuration."""
    preproc_type = PREPROC_TYPES[genes[0]]
    window = WINDOW_SIZES[genes[1]]

    if preproc_type in ['raw', 'snv']:
        return f"Preproc: {preproc_type}"
    else:
        return f"Preproc: {preproc_type}, Window: {window}"


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
    random_state: int = 42,
    fitness_model: str = 'pls'
) -> float:
    """
    Evaluate fitness of a preprocessing configuration.

    Uses cross-validated RMSECV (regression) or accuracy (classification) as fitness.

    Parameters
    ----------
    genes : np.ndarray
        Chromosome encoding preprocessing configuration [preproc_type, window]
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds
    n_components : int
        Max PLS components to test (used for PLS fitness model)
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random state for CV splitting
    fitness_model : str
        Model to use for fitness evaluation: 'pls', 'lightgbm', 'mlp', or 'neuralboosted'

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

        # Choose fitness model
        if fitness_model == 'lightgbm':
            return _evaluate_lightgbm(X_preproc, y, cv, task_type, random_state)
        elif fitness_model == 'mlp':
            return _evaluate_mlp(X_preproc, y, cv, task_type, random_state)
        elif fitness_model == 'neuralboosted':
            return _evaluate_neuralboosted(X_preproc, y, cv, n_comp, task_type, random_state)
        else:
            # Default: PLS
            return _evaluate_pls(X_preproc, y, cv, n_comp, task_type)

    except Exception:
        # Any error = very poor fitness
        return -np.inf


def _evaluate_pls(X, y, cv, n_comp, task_type):
    """Evaluate fitness using PLS."""
    pls = PLSRegression(n_components=n_comp, scale=False)
    y_pred = cross_val_predict(pls, X, y, cv=cv)

    if task_type == 'classification':
        y_class = np.asarray(y)
        if y_class.dtype == object or not np.issubdtype(y_class.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_class = le.fit_transform(y_class)
        y_pred_class = (y_pred > np.median(y_pred)).astype(int).ravel()
        return accuracy_score(y_class, y_pred_class)
    else:
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_lightgbm(X, y, cv, task_type, random_state):
    """Evaluate fitness using LightGBM."""
    if task_type == 'classification':
        y_class = np.asarray(y)
        if y_class.dtype == object or not np.issubdtype(y_class.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_class = le.fit_transform(y_class)
        else:
            y_class = y_class.astype(int)

        model = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=random_state, verbosity=-1, force_col_wise=True
        )
        y_pred = cross_val_predict(model, X, y_class, cv=cv)
        return accuracy_score(y_class, y_pred)
    else:
        model = LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=random_state, verbosity=-1, force_col_wise=True
        )
        y_pred = cross_val_predict(model, X, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_mlp(X, y, cv, task_type, random_state):
    """Evaluate fitness using MLP."""
    from sklearn.neural_network import MLPRegressor, MLPClassifier

    if task_type == 'classification':
        y_class = np.asarray(y)
        if y_class.dtype == object or not np.issubdtype(y_class.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_class = le.fit_transform(y_class)
        else:
            y_class = y_class.astype(int)

        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=random_state, early_stopping=True, validation_fraction=0.1
        )
        y_pred = cross_val_predict(model, X, y_class, cv=cv)
        return accuracy_score(y_class, y_pred)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=random_state, early_stopping=True, validation_fraction=0.1
        )
        y_pred = cross_val_predict(model, X, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_pred))
        return -rmsecv


def _evaluate_neuralboosted(X, y, cv, n_comp, task_type, random_state):
    """Evaluate fitness using NeuralBoosted (falls back to PLS if unavailable)."""
    try:
        from .neural_boosted import NeuralBoostedRegressor, NeuralBoostedClassifier

        if task_type == 'classification':
            y_class = np.asarray(y)
            if y_class.dtype == object or not np.issubdtype(y_class.dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_class = le.fit_transform(y_class)
            else:
                y_class = y_class.astype(int)

            model = NeuralBoostedClassifier(random_state=random_state)
            y_pred = cross_val_predict(model, X, y_class, cv=cv)
            return accuracy_score(y_class, y_pred)
        else:
            model = NeuralBoostedRegressor(random_state=random_state)
            y_pred = cross_val_predict(model, X, y, cv=cv)
            rmsecv = np.sqrt(mean_squared_error(y, y_pred))
            return -rmsecv
    except ImportError:
        # Fall back to PLS
        return _evaluate_pls(X, y, cv, n_comp, task_type)


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
    ]

    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] = rng.randint(0, gene_ranges[i])

    return mutated


# =============================================================================
# EXHAUSTIVE SEARCH (FOR SMALL SEARCH SPACE)
# =============================================================================

def exhaustive_search(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_components: int = 10,
    task_type: str = 'regression',
    random_state: int = 42,
    fitness_model: str = 'pls',
    n_jobs: int = 1,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Exhaustively search all 238 preprocessing combinations.

    With only 238 combinations (14 preprocessing types × 17 window sizes),
    exhaustive search is feasible and guarantees finding the optimal solution.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    cv_folds : int
        Number of CV folds for fitness evaluation
    n_components : int
        Max PLS components for evaluation
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    fitness_model : str
        Model to use for fitness evaluation
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Progress callback function

    Returns
    -------
    result : dict
        Same format as optimize_preprocessing()
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if verbose >= 1:
        print(f"Exhaustive Preprocessing Search")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Task: {task_type}")
        print(f"  Fitness model: {fitness_model.upper()}")
        print(f"  Total combinations: {TOTAL_COMBINATIONS}")

    # Generate all possible chromosomes
    all_genes = [
        np.array([p, w], dtype=np.int32)
        for p in range(len(PREPROC_TYPES))
        for w in range(len(WINDOW_SIZES))
    ]

    # Evaluate all combinations
    if n_jobs != 1:
        # Parallel evaluation
        try:
            from joblib import Parallel, delayed

            if verbose >= 1:
                print(f"  Parallel evaluation with n_jobs={n_jobs}")

            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fitness)(
                    genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model
                )
                for genes in all_genes
            )
        except ImportError:
            # Fall back to sequential
            if verbose >= 1:
                print("  joblib not available, using sequential evaluation")
            results = []
            for i, genes in enumerate(all_genes):
                fitness = evaluate_fitness(
                    genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model
                )
                results.append(fitness)
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback({
                        'algorithm': 'exhaustive_preprocessing',
                        'current': i + 1,
                        'total': len(all_genes),
                        'best_fitness': max(results),
                        'message': f"Tested {i+1}/{len(all_genes)} combinations"
                    })
    else:
        # Sequential evaluation
        results = []
        for i, genes in enumerate(all_genes):
            fitness = evaluate_fitness(
                genes, X, y, cv_folds, n_components, task_type, random_state, fitness_model
            )
            results.append(fitness)

            if verbose >= 2 and (i + 1) % 50 == 0:
                print(f"  Tested {i+1}/{len(all_genes)} combinations")

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback({
                    'algorithm': 'exhaustive_preprocessing',
                    'current': i + 1,
                    'total': len(all_genes),
                    'best_fitness': max(results),
                    'message': f"Tested {i+1}/{len(all_genes)} combinations"
                })

    # Find best
    best_idx = np.argmax(results)
    best_genes = all_genes[best_idx]
    best_fitness = results[best_idx]

    best_name, best_transform = chromosome_to_transform(best_genes)
    best_config = get_config_description(best_genes)

    if verbose >= 1:
        print(f"\nExhaustive search complete!")
        if task_type == 'classification':
            print(f"  Best Accuracy: {best_fitness:.4f}")
        else:
            print(f"  Best RMSECV: {-best_fitness:.4f}")
        print(f"  Best config: {best_config}")

    # Return score in format expected by callers
    if task_type == 'classification':
        best_score = 1.0 - best_fitness
    else:
        best_score = -best_fitness

    return {
        'best_genes': best_genes,
        'best_name': best_name,
        'best_transform': best_transform,
        'best_rmsecv': best_score,
        'best_config': best_config,
        'history': [{'combination': i, 'fitness': f} for i, f in enumerate(results)],
        'task_type': task_type,
        'method': 'exhaustive'
    }


# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def optimize_preprocessing(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'ga',
    population_size: int = 48,
    n_generations: int = 30,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.15,
    tournament_size: int = 3,
    cv_folds: int = 5,
    n_components: int = 10,
    elitism: int = 2,
    task_type: str = 'regression',
    random_state: int = 42,
    verbose: int = 1,
    progress_callback: Optional[Callable] = None,
    fitness_model: str = 'pls',
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Optimize spectral preprocessing using genetic algorithm or exhaustive search.

    Parameters
    ----------
    X : np.ndarray
        Raw spectral data (n_samples, n_features)
    y : np.ndarray
        Target values
    method : str
        Optimization method: 'ga' (genetic algorithm) or 'exhaustive'
        Default is 'ga'. Exhaustive search tests all 238 combinations.
    population_size : int
        Number of individuals in population (GA only)
    n_generations : int
        Number of generations to evolve (GA only)
    crossover_rate : float
        Probability of crossover (0-1, GA only)
    mutation_rate : float
        Probability of mutation per gene (0-1, GA only)
    tournament_size : int
        Number of individuals in tournament selection (GA only)
    cv_folds : int
        Number of CV folds for fitness evaluation
    n_components : int
        Max PLS components for evaluation
    elitism : int
        Number of best individuals to preserve each generation (GA only)
    task_type : str
        'regression' or 'classification'
    random_state : int
        Random seed for reproducibility
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
    progress_callback : callable, optional
        Function called with progress dict
    fitness_model : str
        Model to use for fitness evaluation: 'pls', 'lightgbm', 'mlp', 'neuralboosted'
    n_jobs : int
        Number of parallel jobs for exhaustive search (-1 for all cores)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'best_genes': np.ndarray - Best chromosome
        - 'best_name': str - Name of best preprocessing
        - 'best_transform': callable - Transform function
        - 'best_rmsecv': float - Best RMSECV (regression) or 1-accuracy (classification)
        - 'best_config': str - Human-readable configuration
        - 'history': list - Fitness history
        - 'method': str - Method used ('ga' or 'exhaustive')
    """
    if method == 'exhaustive':
        return exhaustive_search(
            X, y, cv_folds, n_components, task_type, random_state,
            fitness_model, n_jobs, verbose, progress_callback
        )

    # Genetic Algorithm
    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    n_samples, n_features = X.shape

    if verbose >= 1:
        print(f"GA Preprocessing Optimization")
        print(f"  Data: {n_samples} samples, {n_features} features")
        print(f"  Task: {task_type}")
        print(f"  Fitness model: {fitness_model.upper()}")
        print(f"  Search space: {TOTAL_COMBINATIONS} combinations")
        print(f"  Population: {population_size}, Generations: {n_generations}")
        print(f"  CV folds: {cv_folds}, PLS components: {n_components}")

    # Initialize population with seeds + random
    seed_chromosomes = get_seed_chromosomes()
    n_seeds = min(len(seed_chromosomes), population_size // 3)  # Max 33% seeds

    population_list = []
    for i in range(n_seeds):
        population_list.append(seed_chromosomes[i])

    # Fill rest with random chromosomes
    for _ in range(population_size - n_seeds):
        population_list.append(random_chromosome(rng))

    population = np.array(population_list)

    if verbose >= 1:
        print(f"  Seeds: {n_seeds} (max 33% of population)")

    # Evaluate initial fitness
    fitness = np.array([
        evaluate_fitness(ind, X, y, cv_folds, n_components, task_type, random_state, fitness_model)
        for ind in population
    ])

    # Track best
    best_idx = np.argmax(fitness)
    best_genes = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    valid_fitness = fitness[fitness > -np.inf]
    mean_fit = np.mean(valid_fitness) if len(valid_fitness) > 0 else -np.inf
    history = [{'generation': 0, 'best_fitness': best_fitness, 'mean_fitness': mean_fit}]

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
            evaluate_fitness(ind, X, y, cv_folds, n_components, task_type, random_state, fitness_model)
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

        if verbose >= 1 and gen % 5 == 0:
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
    if task_type == 'classification':
        best_score = 1.0 - best_fitness
    else:
        best_score = -best_fitness

    return {
        'best_genes': best_genes,
        'best_name': best_name,
        'best_transform': best_transform,
        'best_rmsecv': best_score,
        'best_config': best_config,
        'history': history,
        'task_type': task_type,
        'method': 'ga'
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
            method='ga',
            population_size=32,
            n_generations=20,
            cv_folds=3,
            random_state=random_state,
            verbose=verbose
        )
    else:
        result = optimize_preprocessing(
            X, y,
            method='ga',
            population_size=48,
            n_generations=30,
            cv_folds=5,
            random_state=random_state,
            verbose=verbose
        )

    return (result['best_name'], result['best_transform'])
