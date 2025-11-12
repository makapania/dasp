"""
R Validation Test Suite

This test suite validates that Python spectral analysis models produce results
equivalent to their R package counterparts.

Test workflow:
1. Load sample spectral data
2. Train Python models with fixed hyperparameters and random seeds
3. Export predictions, feature importances, and model parameters
4. Run R scripts (manually) to train equivalent models
5. Compare results using compare_results.py

To run:
    pytest tests/test_r_validation.py -v -s
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spectral_predict.models import get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor, XGBClassifier


# Fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Output directory for results
OUTPUT_DIR = Path(__file__).parent.parent / 'r_validation_scripts' / 'results'
DATA_DIR = Path(__file__).parent.parent / 'data'


def setup_output_dir():
    """Create output directory for results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'python').mkdir(exist_ok=True)
    (OUTPUT_DIR / 'r').mkdir(exist_ok=True)
    return OUTPUT_DIR


def load_regression_data(size='medium'):
    """
    Load regression test data.

    Parameters
    ----------
    size : str
        'small', 'medium', or 'large'

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Train/test split data
    """
    data_file = DATA_DIR / f'sample_nir_regression_{size}.csv'

    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}. Run generate_sample_data.py first.")

    df = pd.read_csv(data_file)

    # Separate features and target
    target_col = 'target'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    # Fixed train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )

    return X_train, X_test, y_train, y_test


def load_classification_data():
    """
    Load classification test data.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Train/test split data
    """
    data_file = DATA_DIR / 'sample_nir_classification.csv'

    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}. Run generate_sample_data.py first.")

    df = pd.read_csv(data_file)

    # Separate features and target
    target_col = 'target'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    # Fixed train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_data_for_r(X_train, X_test, y_train, y_test, prefix):
    """
    Save train/test data in R-compatible format.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : arrays
        Train/test split data
    prefix : str
        File prefix (e.g., 'pls_regression')
    """
    output_dir = setup_output_dir()

    # Save training data
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    train_df.to_csv(output_dir / f'{prefix}_train.csv', index=False)

    # Save test data
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test
    test_df.to_csv(output_dir / f'{prefix}_test.csv', index=False)

    print(f"Saved train/test data: {prefix}_train.csv, {prefix}_test.csv")


def save_python_results(predictions, model_info, output_file):
    """
    Save Python model results.

    Parameters
    ----------
    predictions : dict
        Dictionary with 'train' and 'test' predictions
    model_info : dict
        Model metadata and parameters
    output_file : str
        Output filename (in results/python/)
    """
    output_dir = setup_output_dir()
    output_path = output_dir / 'python' / output_file

    results = {
        'predictions': {
            'train': predictions['train'].tolist() if isinstance(predictions['train'], np.ndarray) else predictions['train'],
            'test': predictions['test'].tolist() if isinstance(predictions['test'], np.ndarray) else predictions['test']
        },
        'model_info': model_info,
        'random_seed': RANDOM_SEED
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved Python results: {output_path}")


# =============================================================================
# PLS REGRESSION VALIDATION
# =============================================================================

def test_pls_regression():
    """Test PLS regression against R's pls package."""
    print("\n" + "="*80)
    print("TEST: PLS Regression vs R pls package")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'pls_regression')

    # Fixed hyperparameters (MUST match R script)
    n_components = 10

    # Train Python PLS model
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython PLS Results:")
    print(f"  Components: {n_components}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")

    # Save results
    model_info = {
        'model_type': 'PLS',
        'n_components': n_components,
        'scale': False,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'x_loadings_shape': model.x_loadings_.shape,
        'y_loadings_shape': model.y_loadings_.shape
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'pls_regression.json')

    # Save loadings for comparison
    loadings_df = pd.DataFrame(model.x_loadings_, columns=[f'comp_{i+1}' for i in range(n_components)])
    loadings_df.to_csv(OUTPUT_DIR / 'python' / 'pls_regression_loadings.csv', index=False)

    print("\nNext step: Run r_validation_scripts/pls_comparison.R")


# =============================================================================
# RANDOM FOREST REGRESSION VALIDATION
# =============================================================================

def test_randomforest_regression():
    """Test Random Forest regression against R's randomForest package."""
    print("\n" + "="*80)
    print("TEST: Random Forest Regression vs R randomForest package")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'rf_regression')

    # Fixed hyperparameters (MUST match R script)
    n_estimators = 500
    max_features = 'sqrt'  # In R: mtry = sqrt(n_features)
    min_samples_split = 2  # In R: nodesize (similar concept)
    random_state = RANDOM_SEED

    # Train Python Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython RandomForest Results:")
    print(f"  Trees: {n_estimators}")
    print(f"  Max features: {max_features} ({int(np.sqrt(X_train.shape[1]))} features)")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")

    # Save results
    model_info = {
        'model_type': 'RandomForest',
        'n_estimators': n_estimators,
        'max_features': max_features,
        'mtry_actual': int(np.sqrt(X_train.shape[1])),
        'min_samples_split': min_samples_split,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'rf_regression.json')

    # Save feature importances
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
        'importance': model.feature_importances_
    })
    importance_df.to_csv(OUTPUT_DIR / 'python' / 'rf_regression_importance.csv', index=False)

    print("\nNext step: Run r_validation_scripts/random_forest_comparison.R")


# =============================================================================
# XGBOOST REGRESSION VALIDATION
# =============================================================================

def test_xgboost_regression():
    """Test XGBoost regression against R's xgboost package."""
    print("\n" + "="*80)
    print("TEST: XGBoost Regression vs R xgboost package")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'xgb_regression')

    # Fixed hyperparameters (MUST match R script)
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 0
    }

    # Train Python XGBoost model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython XGBoost Results:")
    print(f"  N estimators: {params['n_estimators']}")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Max depth: {params['max_depth']}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")

    # Save results
    model_info = {
        'model_type': 'XGBoost',
        **params,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'xgb_regression.json')

    # Save feature importances
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
        'importance': model.feature_importances_
    })
    importance_df.to_csv(OUTPUT_DIR / 'python' / 'xgb_regression_importance.csv', index=False)

    print("\nNext step: Run r_validation_scripts/xgboost_comparison.R")


# =============================================================================
# RIDGE/LASSO/ELASTICNET REGRESSION VALIDATION (glmnet)
# =============================================================================

def test_ridge_regression():
    """Test Ridge regression against R's glmnet package."""
    print("\n" + "="*80)
    print("TEST: Ridge Regression vs R glmnet package (alpha=0)")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'ridge_regression')

    # Fixed hyperparameters (MUST match R script)
    alpha = 1.0  # In glmnet, this is lambda

    # Train Python Ridge model
    model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython Ridge Results:")
    print(f"  Alpha (lambda): {alpha}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")

    # Save results
    model_info = {
        'model_type': 'Ridge',
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coefs': np.sum(np.abs(model.coef_) > 1e-10)
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'ridge_regression.json')

    # Save coefficients
    coef_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.coef_))],
        'coefficient': model.coef_
    })
    coef_df.to_csv(OUTPUT_DIR / 'python' / 'ridge_regression_coefs.csv', index=False)

    print("\nNext step: Run r_validation_scripts/glmnet_comparison.R")


def test_lasso_regression():
    """Test Lasso regression against R's glmnet package."""
    print("\n" + "="*80)
    print("TEST: Lasso Regression vs R glmnet package (alpha=1)")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'lasso_regression')

    # Fixed hyperparameters (MUST match R script)
    alpha = 0.1  # In glmnet, this is lambda

    # Train Python Lasso model
    model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=10000)
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython Lasso Results:")
    print(f"  Alpha (lambda): {alpha}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-10)}/{len(model.coef_)}")

    # Save results
    model_info = {
        'model_type': 'Lasso',
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coefs': np.sum(np.abs(model.coef_) > 1e-10)
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'lasso_regression.json')

    # Save coefficients
    coef_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.coef_))],
        'coefficient': model.coef_
    })
    coef_df.to_csv(OUTPUT_DIR / 'python' / 'lasso_regression_coefs.csv', index=False)

    print("\nNext step: Run r_validation_scripts/glmnet_comparison.R")


def test_elasticnet_regression():
    """Test ElasticNet regression against R's glmnet package."""
    print("\n" + "="*80)
    print("TEST: ElasticNet Regression vs R glmnet package")
    print("="*80)

    # Load data
    X_train, X_test, y_train, y_test = load_regression_data('medium')
    save_data_for_r(X_train, X_test, y_train, y_test, 'elasticnet_regression')

    # Fixed hyperparameters (MUST match R script)
    alpha = 0.1  # In glmnet, this is lambda
    l1_ratio = 0.5  # In glmnet, this is alpha (confusing naming!)

    # Train Python ElasticNet model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_SEED, max_iter=10000)
    model.fit(X_train, y_train)

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nPython ElasticNet Results:")
    print(f"  Alpha (lambda): {alpha}")
    print(f"  L1 ratio (alpha in R): {l1_ratio}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Train R²:   {train_r2:.6f}")
    print(f"  Test R²:    {test_r2:.6f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-10)}/{len(model.coef_)}")

    # Save results
    model_info = {
        'model_type': 'ElasticNet',
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coefs': np.sum(np.abs(model.coef_) > 1e-10)
    }

    predictions = {
        'train': y_train_pred,
        'test': y_test_pred
    }

    save_python_results(predictions, model_info, 'elasticnet_regression.json')

    # Save coefficients
    coef_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.coef_))],
        'coefficient': model.coef_
    })
    coef_df.to_csv(OUTPUT_DIR / 'python' / 'elasticnet_regression_coefs.csv', index=False)

    print("\nNext step: Run r_validation_scripts/glmnet_comparison.R")


# =============================================================================
# SUMMARY TEST
# =============================================================================

def test_generate_all_python_results():
    """Run all Python models and generate results for R comparison."""
    print("\n" + "="*80)
    print("GENERATING ALL PYTHON RESULTS FOR R VALIDATION")
    print("="*80)

    test_pls_regression()
    test_randomforest_regression()
    test_xgboost_regression()
    test_ridge_regression()
    test_lasso_regression()
    test_elasticnet_regression()

    print("\n" + "="*80)
    print("PYTHON RESULTS GENERATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/python/")
    print("\nNext steps:")
    print("1. Install R packages (see README_R_VALIDATION.md)")
    print("2. Run R comparison scripts:")
    print("   - Rscript r_validation_scripts/pls_comparison.R")
    print("   - Rscript r_validation_scripts/random_forest_comparison.R")
    print("   - Rscript r_validation_scripts/xgboost_comparison.R")
    print("   - Rscript r_validation_scripts/glmnet_comparison.R")
    print("3. Compare results:")
    print("   - python r_validation_scripts/compare_results.py")


if __name__ == '__main__':
    # Run all tests
    setup_output_dir()
    test_generate_all_python_results()
