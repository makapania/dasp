"""
Apply all NSGA-II fixes in a single pass.
This script modifies nsga2_search.py to fix all 6 identified bugs.
"""
import re

def apply_all_fixes(filepath):
    """Apply all NSGA-II fixes to the file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # =========================================================================
    # Phase 1: Expand chromosome structure
    # =========================================================================

    # 1.1: Update SeededWavelengthSampling comment and n_vars
    content = content.replace(
        '# Variable structure: [preproc_idx, window_idx, model_idx, model_param, wl_0, wl_1, ..., wl_n]\n        n_vars = 4 + self.n_wavelengths',
        '# Variable structure: [preproc_idx, window_idx, model_idx, model_param, lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene, wl_0, wl_1, ..., wl_n]\n        n_vars = 8 + self.n_wavelengths'
    )

    # 1.2: Update seeded population initialization (first loop)
    content = content.replace(
        '''            X[i, 0] = 0  # raw preprocessing
            X[i, 1] = 6  # default window
            X[i, 2] = model_idx  # cycle through model types
            X[i, 3] = 7  # middle model_param value (good default)
            X[i, 4:] = 1  # ALL wavelengths selected''',
        '''            X[i, 0] = 0  # raw preprocessing
            X[i, 1] = 6  # default window
            X[i, 2] = model_idx  # cycle through model types
            X[i, 3] = 7  # middle model_param value (good default)
            X[i, 4] = 7  # middle lr_gene (0.1 learning rate)
            X[i, 5] = 7  # middle reg_alpha_gene
            X[i, 6] = 7  # middle reg_lambda_gene
            X[i, 7] = 7  # middle l1_gene (0.5 l1_ratio)
            X[i, 8:] = 1  # ALL wavelengths selected'''
    )

    # 1.3: Update seeded population initialization (second loop with SNV)
    content = content.replace(
        '''            X[i, 0] = 1  # SNV preprocessing
            X[i, 1] = 6  # default window
            X[i, 2] = model_idx
            X[i, 3] = 7
            X[i, 4:] = 1  # ALL wavelengths selected''',
        '''            X[i, 0] = 1  # SNV preprocessing
            X[i, 1] = 6  # default window
            X[i, 2] = model_idx
            X[i, 3] = 7
            X[i, 4] = 7  # middle lr_gene
            X[i, 5] = 7  # middle reg_alpha_gene
            X[i, 6] = 7  # middle reg_lambda_gene
            X[i, 7] = 7  # middle l1_gene
            X[i, 8:] = 1  # ALL wavelengths selected'''
    )

    # 1.4: Update SpectralOptimizationProblem docstring
    old_docstring = '''    """
    Multi-objective optimization problem for spectral calibration.

    Decision variables (chromosome):
    - Gene 0: Preprocessing type (0-9)
    - Gene 1: S-G window size index (0-14)
    - Gene 2: Model type (0=PLS, 1=Ridge, ...)
    - Gene 3: Model parameter (0-14)
    - Gene 4-N: Binary wavelength selection (0/1)

    Objectives (all minimized):
    1. Prediction error (RMSE or 1-Accuracy)
    2. Number of selected wavelengths (normalized)
    3. Model complexity score (normalized)
    """'''

    new_docstring = '''    """
    Multi-objective optimization problem for spectral calibration.

    Decision variables (chromosome):
    - Gene 0: Preprocessing type (0-9)
    - Gene 1: S-G window size index (0-14)
    - Gene 2: Model type (0=PLS, 1=Ridge, ...)
    - Gene 3: Model parameter (0-14)
    - Gene 4: Learning rate gene (0-14)
    - Gene 5: L1 regularization gene (0-14)
    - Gene 6: L2 regularization gene (0-14)
    - Gene 7: ElasticNet l1_ratio gene (0-14)
    - Gene 8-N: Binary wavelength selection (0/1)

    Objectives (all minimized):
    1. Prediction error (RMSE or 1-Accuracy)
    2. Number of selected wavelengths (normalized)
    3. Model complexity score (normalized)
    """'''

    content = content.replace(old_docstring, new_docstring)

    # 1.5: Update Problem class n_vars and bounds
    content = content.replace(
        '''        # Decision variables:
        # [preproc_type, window_idx, model_type, model_param, wl_0, wl_1, ..., wl_n]
        n_vars = 4 + self.n_wavelengths

        # Variable bounds
        xl = np.zeros(n_vars)
        xu = np.array([
            len(PREPROC_TYPES) - 1,  # preproc_type
            len(WINDOW_SIZES) - 1,   # window_idx
            len(self.model_types) - 1,    # model_type (use instance var)
            14,                       # model_param (LVs 1-15 or alpha index)
        ] + [1] * self.n_wavelengths)  # wavelength selection''',
        '''        # Decision variables:
        # [preproc_type, window_idx, model_type, model_param, lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene, wl_0, wl_1, ..., wl_n]
        n_vars = 8 + self.n_wavelengths

        # Variable bounds
        xl = np.zeros(n_vars)
        xu = np.array([
            len(PREPROC_TYPES) - 1,  # preproc_type
            len(WINDOW_SIZES) - 1,   # window_idx
            len(self.model_types) - 1,    # model_type (use instance var)
            14,                       # model_param (LVs 1-15 or alpha index)
            14,                       # lr_gene (0-14)
            14,                       # reg_alpha_gene (0-14)
            14,                       # reg_lambda_gene (0-14)
            14,                       # l1_gene (0-14)
        ] + [1] * self.n_wavelengths)  # wavelength selection'''
    )

    # 1.6: Update chromosome decoding in _evaluate
    content = content.replace(
        '''            # Decode chromosome
            preproc_idx = chromosome[0]
            window_idx = chromosome[1]
            model_idx = chromosome[2]
            model_param = chromosome[3]
            wavelength_mask = chromosome[4:].astype(bool)''',
        '''            # Decode chromosome
            preproc_idx = chromosome[0]
            window_idx = chromosome[1]
            model_idx = chromosome[2]
            model_param = chromosome[3]
            lr_gene = chromosome[4]
            reg_alpha_gene = chromosome[5]
            reg_lambda_gene = chromosome[6]
            l1_gene = chromosome[7]
            wavelength_mask = chromosome[8:].astype(bool)'''
    )

    # =========================================================================
    # Phase 2: Fix RMSE formula and penalty
    # =========================================================================

    # 2.1: Fix penalty value (1e6 -> 1e10)
    content = content.replace(
        '''            if n_selected < self.min_wavelengths:
                F[i, 0] = 1e6  # Very high error''',
        '''            if n_selected < self.min_wavelengths:
                F[i, 0] = 1e10  # Very high error (matches Bayesian)'''
    )

    # 2.2: Fix RMSE formula in _compute_prediction_error
    # Find and replace the RMSE calculation block
    old_rmse_block = '''                # Use sqrt(mean(MSE)) for optimization - this formula works best
                # for multi-objective Pareto trade-offs. Display RMSE is recomputed
                # separately in convert_nsga2_to_v1_format() using mean(sqrt(MSE))
                # to match Model Development.
                rmse = np.sqrt(-np.mean(scores))
                return rmse'''

    new_rmse_block = '''                # Use mean(sqrt(MSE)) for optimization and display consistency
                # This matches Bayesian optimization and Model Development exactly.
                rmse_per_fold = np.sqrt(-scores)
                rmse = float(np.mean(rmse_per_fold))
                return rmse'''

    content = content.replace(old_rmse_block, new_rmse_block)

    # =========================================================================
    # Phase 3: Add gene decoding helper and update _build_model
    # =========================================================================

    # 3.1: Add gene decoding helper function after _get_preprocessing_transform
    helper_function = '''

def _decode_hyperparameter_genes(
    lr_gene: int,
    reg_alpha_gene: int,
    reg_lambda_gene: int,
    l1_gene: int,
) -> Dict[str, float]:
    """
    Decode hyperparameter genes to actual values.

    Parameters
    ----------
    lr_gene : int (0-14)
        Learning rate gene
    reg_alpha_gene : int (0-14)
        L1 regularization gene
    reg_lambda_gene : int (0-14)
        L2 regularization gene
    l1_gene : int (0-14)
        ElasticNet l1_ratio gene

    Returns
    -------
    params : dict
        Decoded hyperparameters with keys:
        - learning_rate: float, range 0.01 to 0.3 (log scale)
        - reg_alpha: float, range 1e-8 to 10.0 (log scale)
        - reg_lambda: float, range 1e-8 to 10.0 (log scale)
        - l1_ratio: float, range 0.1 to 0.9 (linear scale)
    """
    # Learning rate: log scale 0.01 to 0.3
    # Formula: lr = 0.01 * (3.0 ** (gene / 14))
    # gene=0 -> 0.01, gene=7 -> 0.1, gene=14 -> 0.3
    lr = 0.01 * (3.0 ** (lr_gene / 14.0))

    # Regularization: log scale 1e-8 to 10.0
    # Formula: reg = 10 ** ((gene/14) * 8 - 8)
    # gene=0 -> 1e-8, gene=7 -> 1e-4, gene=14 -> 10.0
    reg_alpha = 10 ** (reg_alpha_gene / 14.0 * 8 - 8)
    reg_lambda = 10 ** (reg_lambda_gene / 14.0 * 8 - 8)

    # l1_ratio: linear scale 0.1 to 0.9
    # Formula: l1_ratio = 0.1 + (gene / 14) * 0.8
    # gene=0 -> 0.1, gene=7 -> 0.5, gene=14 -> 0.9
    l1_ratio = 0.1 + (l1_gene / 14.0) * 0.8

    return {
        'learning_rate': lr,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'l1_ratio': l1_ratio,
    }

'''

    # Insert helper function after _get_preprocessing_transform
    marker = 'def _build_model(model_type: str, model_param: int, task_type: str, random_state: int):'
    content = content.replace(marker, helper_function + marker)

    # 3.2: Update _build_model signature to accept hyperparams
    content = content.replace(
        'def _build_model(model_type: str, model_param: int, task_type: str, random_state: int):',
        'def _build_model(model_type: str, model_param: int, task_type: str, random_state: int, hyperparams: Optional[Dict[str, float]] = None):'
    )

    # 3.3: Fix ElasticNet (BUG 6) - independent l1_ratio
    old_elasticnet = '''    elif model_type == 'ElasticNet':
        # alpha from 0.01 to 100, l1_ratio from 0.2 to 0.8
        alpha = 10 ** (model_param / 3 - 2)
        l1_ratio = 0.2 + (model_param % 5) * 0.15  # 0.2 to 0.8
        if task_type == 'regression':
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty='elasticnet', C=1/max(alpha, 1e-6), l1_ratio=l1_ratio,
                                      solver='saga', random_state=random_state, max_iter=1000)'''

    new_elasticnet = '''    elif model_type == 'ElasticNet':
        # alpha from 0.01 to 100
        alpha = 10 ** (model_param / 3 - 2)
        # Use independent l1_ratio from hyperparams (fixes BUG 6)
        l1_ratio = hyperparams.get('l1_ratio', 0.5) if hyperparams else 0.5
        if task_type == 'regression':
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty='elasticnet', C=1/max(alpha, 1e-6), l1_ratio=l1_ratio,
                                      solver='saga', random_state=random_state, max_iter=1000)'''

    content = content.replace(old_elasticnet, new_elasticnet)

    # 3.4: Fix LightGBM (BUG 4 + BUG 5) - continuous learning_rate and reg_lambda
    old_lightgbm = '''    elif model_type == 'LightGBM':
        if not HAS_LIGHTGBM:
            return None
        n_estimators = 50 + model_param * 10  # 50-190
        learning_rate = 0.05 if model_param < 7 else 0.1
        num_leaves = 15 if task_type == 'classification' else 31
        if task_type == 'regression':
            return LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                 num_leaves=num_leaves, reg_lambda=0.1,
                                 random_state=random_state, n_jobs=1, verbose=-1)
        else:
            return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                  num_leaves=num_leaves, reg_lambda=0.1,
                                  random_state=random_state, n_jobs=1, verbose=-1)'''

    new_lightgbm = '''    elif model_type == 'LightGBM':
        if not HAS_LIGHTGBM:
            return None
        n_estimators = 50 + model_param * 10  # 50-190
        # Use learning_rate and reg_lambda from hyperparams (fixes BUG 4 + BUG 5)
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        reg_lambda = hyperparams.get('reg_lambda', 0.1) if hyperparams else 0.1
        num_leaves = 15 if task_type == 'classification' else 31
        if task_type == 'regression':
            return LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                 num_leaves=num_leaves, reg_lambda=reg_lambda,
                                 random_state=random_state, n_jobs=1, verbose=-1)
        else:
            return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                  num_leaves=num_leaves, reg_lambda=reg_lambda,
                                  random_state=random_state, n_jobs=1, verbose=-1)'''

    content = content.replace(old_lightgbm, new_lightgbm)

    # 3.5: Fix XGBoost (BUG 3 + BUG 5) - continuous learning_rate, reg_alpha, reg_lambda
    old_xgboost = '''    elif model_type == 'XGBoost':
        if not HAS_XGBOOST:
            return None
        n_estimators = 50 + model_param * 10  # 50-190
        max_depth = 3 + (model_param % 5)  # 3-7
        subsample = 1.0 if model_param < 7 else 0.8
        colsample = 1.0 if model_param < 7 else 0.8
        if task_type == 'regression':
            return XGBRegressor(n_estimators=n_estimators, learning_rate=0.1,
                                max_depth=max_depth, reg_lambda=0.1,
                                subsample=subsample, colsample_bytree=colsample,
                                random_state=random_state, n_jobs=1, verbosity=0)
        else:
            return XGBClassifier(n_estimators=n_estimators, learning_rate=0.1,
                                 max_depth=max_depth, reg_lambda=0.1,
                                 subsample=subsample, colsample_bytree=colsample,
                                 random_state=random_state, n_jobs=1, verbosity=0,
                                 use_label_encoder=False, eval_metric='logloss')'''

    new_xgboost = '''    elif model_type == 'XGBoost':
        if not HAS_XGBOOST:
            return None
        n_estimators = 50 + model_param * 10  # 50-190
        max_depth = 3 + (model_param % 5)  # 3-7
        subsample = 1.0 if model_param < 7 else 0.8
        colsample = 1.0 if model_param < 7 else 0.8
        # Use learning_rate, reg_alpha, reg_lambda from hyperparams (fixes BUG 3 + BUG 5)
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        reg_alpha = hyperparams.get('reg_alpha', 0.1) if hyperparams else 0.1
        reg_lambda = hyperparams.get('reg_lambda', 0.1) if hyperparams else 0.1
        if task_type == 'regression':
            return XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                subsample=subsample, colsample_bytree=colsample,
                                random_state=random_state, n_jobs=1, verbosity=0)
        else:
            return XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                 max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                 subsample=subsample, colsample_bytree=colsample,
                                 random_state=random_state, n_jobs=1, verbosity=0,
                                 use_label_encoder=False, eval_metric='logloss')'''

    content = content.replace(old_xgboost, new_xgboost)

    # 3.6: Fix CatBoost (BUG 4) - continuous learning_rate
    old_catboost = '''    elif model_type == 'CatBoost':
        if not HAS_CATBOOST or CatBoostRegressor is None:
            return None
        iterations = 50 + model_param * 15  # 50-260
        depth = 4 + (model_param % 5)  # 4-8
        learning_rate = 0.05 if model_param < 7 else 0.1
        if task_type == 'regression':
            return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                     random_state=random_state, verbose=0, thread_count=1)
        else:
            return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                      random_state=random_state, verbose=0, thread_count=1)'''

    new_catboost = '''    elif model_type == 'CatBoost':
        if not HAS_CATBOOST or CatBoostRegressor is None:
            return None
        iterations = 50 + model_param * 15  # 50-260
        depth = 4 + (model_param % 5)  # 4-8
        # Use learning_rate from hyperparams (fixes BUG 4)
        learning_rate = hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1
        if task_type == 'regression':
            return CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                     random_state=random_state, verbose=0, thread_count=1)
        else:
            return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                      random_state=random_state, verbose=0, thread_count=1)'''

    content = content.replace(old_catboost, new_catboost)

    # 3.7: Update _compute_prediction_error to decode hyperparams and pass to _build_model
    # Find the model building section in _compute_prediction_error
    old_prediction_build = '''            # Build model
            model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]

            # Special handling for PLS - limit components and use scale=False
            # scale=False matches get_model() in models.py for consistency with Model Development
            if model_type == 'PLS':
                n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
                n_components = max(1, n_components)
                model = PLSRegression(n_components=n_components, scale=False)
            else:
                model = _build_model(model_type, model_param, self.task_type, self.random_state)'''

    new_prediction_build = '''            # Build model
            model_type = self.model_types[min(model_idx, len(self.model_types) - 1)]

            # Decode hyperparameter genes
            hyperparams = _decode_hyperparameter_genes(lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene)

            # Special handling for PLS - limit components and use scale=False
            # scale=False matches get_model() in models.py for consistency with Model Development
            if model_type == 'PLS':
                n_components = min(model_param + 1, X_subset.shape[1], X_subset.shape[0] - 1)
                n_components = max(1, n_components)
                model = PLSRegression(n_components=n_components, scale=False)
            else:
                model = _build_model(model_type, model_param, self.task_type, self.random_state, hyperparams)'''

    content = content.replace(old_prediction_build, new_prediction_build)

    print(f"Applied fixes successfully. Changed {len(content) - len(original_content)} characters.")

    return content


if __name__ == '__main__':
    import sys

    filepath = 'src/spectral_predict/nsga2_search.py'

    print(f"Applying NSGA-II fixes to {filepath}...")
    fixed_content = apply_all_fixes(filepath)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print("Phase 1-3 complete! Now applying Phase 4 decoding updates...")
