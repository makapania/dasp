"""
Apply Phase 4 NSGA-II fixes: Update all decoding functions.
"""
import re

def apply_phase4_fixes(filepath):
    """Apply Phase 4 fixes to decode_solution and related functions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # =========================================================================
    # Phase 4: Update all decoding functions
    # =========================================================================

    # 4.1: Update decode_solution() - extract 8 genes
    old_decode_extract = '''    preproc_idx = int(chromosome[0])
    window_idx = int(chromosome[1])
    model_idx = int(chromosome[2])
    model_param = int(chromosome[3])
    wavelength_mask = chromosome[4:].astype(bool)'''

    new_decode_extract = '''    preproc_idx = int(chromosome[0])
    window_idx = int(chromosome[1])
    model_idx = int(chromosome[2])
    model_param = int(chromosome[3])
    lr_gene = int(chromosome[4])
    reg_alpha_gene = int(chromosome[5])
    reg_lambda_gene = int(chromosome[6])
    l1_gene = int(chromosome[7])
    wavelength_mask = chromosome[8:].astype(bool)

    # Decode hyperparameter genes
    hyperparams = _decode_hyperparameter_genes(lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene)'''

    content = content.replace(old_decode_extract, new_decode_extract, 1)  # Only first occurrence (decode_solution)

    # 4.2: Update ElasticNet in decode_solution()
    old_decode_elasticnet = '''    elif model_type == 'ElasticNet':
        alpha = 10 ** (model_param / 3 - 2)
        l1_ratio = 0.2 + (model_param % 5) * 0.15
        nsga_overrides = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': 10000}'''

    new_decode_elasticnet = '''    elif model_type == 'ElasticNet':
        alpha = 10 ** (model_param / 3 - 2)
        l1_ratio = hyperparams['l1_ratio']  # Use decoded l1_ratio
        nsga_overrides = {'alpha': alpha, 'l1_ratio': l1_ratio, 'max_iter': 10000}'''

    content = content.replace(old_decode_elasticnet, new_decode_elasticnet)

    # 4.3: Update LightGBM in decode_solution()
    old_decode_lightgbm = '''    elif model_type == 'LightGBM':
        n_estimators = 50 + model_param * 10
        learning_rate = 0.05 if model_param < 7 else 0.1
        nsga_overrides = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'n_jobs': 1,
        }'''

    new_decode_lightgbm = '''    elif model_type == 'LightGBM':
        n_estimators = 50 + model_param * 10
        learning_rate = hyperparams['learning_rate']
        reg_lambda = hyperparams['reg_lambda']
        nsga_overrides = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'reg_lambda': reg_lambda,
            'n_jobs': 1,
        }'''

    content = content.replace(old_decode_lightgbm, new_decode_lightgbm)

    # 4.4: Update XGBoost in decode_solution()
    old_decode_xgboost = '''    elif model_type == 'XGBoost':
        n_estimators = 50 + model_param * 10
        max_depth = 3 + (model_param % 5)
        subsample = 1.0 if model_param < 7 else 0.8
        colsample_bytree = 1.0 if model_param < 7 else 0.8
        nsga_overrides = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'n_jobs': 1,
        }'''

    new_decode_xgboost = '''    elif model_type == 'XGBoost':
        n_estimators = 50 + model_param * 10
        max_depth = 3 + (model_param % 5)
        subsample = 1.0 if model_param < 7 else 0.8
        colsample_bytree = 1.0 if model_param < 7 else 0.8
        learning_rate = hyperparams['learning_rate']
        reg_alpha = hyperparams['reg_alpha']
        reg_lambda = hyperparams['reg_lambda']
        nsga_overrides = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': 1,
        }'''

    content = content.replace(old_decode_xgboost, new_decode_xgboost)

    # 4.5: Update CatBoost in decode_solution()
    old_decode_catboost = '''    elif model_type == 'CatBoost':
        iterations = 50 + model_param * 15
        depth = 4 + (model_param % 5)
        learning_rate = 0.05 if model_param < 7 else 0.1
        nsga_overrides = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'thread_count': 1
        }'''

    new_decode_catboost = '''    elif model_type == 'CatBoost':
        iterations = 50 + model_param * 15
        depth = 4 + (model_param % 5)
        learning_rate = hyperparams['learning_rate']
        nsga_overrides = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'thread_count': 1
        }'''

    content = content.replace(old_decode_catboost, new_decode_catboost)

    # 4.6: Update _compute_solution_r2() - extract 8 genes and decode
    old_r2_extract = '''        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        wavelength_mask = solution[4:].astype(bool)'''

    new_r2_extract = '''        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        lr_gene = int(solution[4])
        reg_alpha_gene = int(solution[5])
        reg_lambda_gene = int(solution[6])
        l1_gene = int(solution[7])
        wavelength_mask = solution[8:].astype(bool)

        # Decode hyperparameter genes
        hyperparams = _decode_hyperparameter_genes(lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene)'''

    content = content.replace(old_r2_extract, new_r2_extract)

    # 4.7: Update _build_model call in _compute_solution_r2()
    # Find and replace the _build_model call in _compute_solution_r2
    old_r2_build = '''        else:
            # Use _build_model for all other models (maintains original NSGA behavior)
            model = _build_model(model_type, model_param, task_type, random_state)'''

    new_r2_build = '''        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)'''

    content = content.replace(old_r2_build, new_r2_build)

    # 4.8: Update _compute_display_rmse() - extract 8 genes and decode
    old_rmse_extract = '''        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        wavelength_mask = solution[4:].astype(bool)'''

    new_rmse_extract = '''        # Decode solution
        preproc_idx = int(solution[0])
        window_idx = int(solution[1])
        model_idx = int(solution[2])
        model_param = int(solution[3])
        lr_gene = int(solution[4])
        reg_alpha_gene = int(solution[5])
        reg_lambda_gene = int(solution[6])
        l1_gene = int(solution[7])
        wavelength_mask = solution[8:].astype(bool)

        # Decode hyperparameter genes
        hyperparams = _decode_hyperparameter_genes(lr_gene, reg_alpha_gene, reg_lambda_gene, l1_gene)'''

    # Find the second occurrence (in _compute_display_rmse)
    parts = content.split(old_rmse_extract)
    if len(parts) >= 3:  # Found at least 2 occurrences (decode_solution and _compute_display_rmse)
        # Reconstruct: first part + first replacement + second part + second replacement + rest
        content = parts[0] + new_rmse_extract + parts[1] + old_rmse_extract + parts[2]
        # Now replace the second occurrence
        content = content.replace(old_rmse_extract, new_rmse_extract, 1)

    # 4.9: Update _build_model call in _compute_display_rmse()
    # This will be the second occurrence after _compute_solution_r2
    old_rmse_build = '''        else:
            # Use _build_model for all other models
            model = _build_model(model_type, model_param, task_type, random_state)'''

    new_rmse_build = '''        else:
            # Use _build_model for all other models with hyperparams
            model = _build_model(model_type, model_param, task_type, random_state, hyperparams)'''

    # Replace the second occurrence (in _compute_display_rmse)
    parts = content.split(old_rmse_build)
    if len(parts) == 3:  # Found 2 occurrences
        content = parts[0] + new_rmse_build + parts[1] + new_rmse_build + parts[2]
    elif len(parts) == 2:  # Found 1 occurrence
        content = parts[0] + new_rmse_build + parts[1]

    print(f"Phase 4 applied successfully. Changed {len(content) - len(original_content)} characters.")

    return content


if __name__ == '__main__':
    filepath = 'src/spectral_predict/nsga2_search.py'

    print(f"Applying Phase 4 fixes to {filepath}...")
    fixed_content = apply_phase4_fixes(filepath)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print("All phases complete! Running validation...")
