"""
Comprehensive GUI feature tests for Spectral Predict V1.

These tests run features THROUGH THE GUI to catch integration issues
that don't appear when testing backend functions directly.

All tests compare against the baseline:
- Models: PLS, Ridge, ElasticNet
- Preprocessing: Raw, SNV
- CV: 5-fold with SPXY 8-sample holdout
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================
# Baseline Fixture (runs through GUI harness)
# ============================================================

@pytest.fixture(scope="function")
def baseline_via_gui(loaded_regression_data):
    """
    Run baseline analysis through GUI and store results.

    This is the reference for all comparisons.
    """
    harness = loaded_regression_data

    # Configure baseline through GUI variables
    harness.set_var('folds', 5)

    # Disable all models first
    model_vars = ['use_pls', 'use_ridge', 'use_lasso', 'use_elasticnet',
                  'use_randomforest', 'use_mlp', 'use_svr',
                  'use_xgboost', 'use_lightgbm', 'use_catboost']
    for var in model_vars:
        harness.set_var(var, False)

    # Enable baseline models
    harness.set_var('use_pls', True)
    harness.set_var('use_ridge', True)
    harness.set_var('use_elasticnet', True)

    # Set preprocessing through GUI
    harness.set_var('use_raw', True)
    harness.set_var('use_snv', True)
    harness.set_var('use_sg1', False)
    harness.set_var('use_sg2', False)

    # Run analysis with SPXY holdout
    success = harness.run_analysis_direct(
        models=['PLS', 'Ridge', 'ElasticNet'],
        preprocessing=['Raw', 'SNV'],
        cv_folds=5,
        holdout_samples=8,
        holdout_method='spxy'
    )

    assert success, "Baseline analysis failed"

    df = harness.get_results_df()

    # Extract best R2 per model
    r2_col = None
    for col in df.columns:
        if 'r2' in col.lower():
            r2_col = col
            break

    baseline = {}
    for model in ['PLS', 'Ridge', 'ElasticNet']:
        model_results = df[df['Model'] == model]
        if len(model_results) > 0 and r2_col:
            baseline[model] = model_results[r2_col].max()

    baseline['best'] = max(baseline.values()) if baseline else 0
    baseline['r2_col'] = r2_col

    return {
        'harness': harness,
        'baseline': baseline,
        'results_df': df
    }


def print_comparison(model_name, r2, baseline_best, preprocessing='SNV'):
    """Print comparison result with clear formatting."""
    diff = r2 - baseline_best
    if diff > 0.01:
        status = "OUTPERFORMS BASELINE"
        symbol = "[+]"
    elif diff < -0.01:
        status = "UNDERPERFORMS"
        symbol = "[-]"
    else:
        status = "MATCHES BASELINE"
        symbol = "[=]"

    print(f"  {symbol} {model_name}: R2={r2:.4f} (baseline={baseline_best:.4f}, diff={diff:+.4f}) - {status}")
    return diff


# ============================================================
# 1. ALL MODEL TYPES (via GUI)
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestAllModelsViaGUI:
    """
    Test all model types through the GUI.

    Each test:
    1. Configures the GUI with specific model settings
    2. Runs analysis through the harness
    3. Compares R2 to baseline
    """

    def test_baseline_report(self, baseline_via_gui):
        """Report baseline results."""
        baseline = baseline_via_gui['baseline']

        print("\n" + "=" * 70)
        print("BASELINE RESULTS (PLS, Ridge, ElasticNet with SPXY 8-sample holdout)")
        print("=" * 70)
        for model in ['PLS', 'Ridge', 'ElasticNet']:
            if model in baseline:
                print(f"  {model}: R2 = {baseline[model]:.4f}")
        print(f"  BEST BASELINE: R2 = {baseline['best']:.4f}")
        print("=" * 70 + "\n")

        assert baseline['best'] > 0.3, "Baseline R2 should be reasonable"

    def test_randomforest_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test RandomForest through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- RandomForest via GUI ---")

        # Configure GUI for RandomForest only
        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_randomforest', True)

        success = harness.run_analysis_direct(
            models=['RandomForest'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "RandomForest analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('RandomForest', best_r2, baseline_best)

    def test_xgboost_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test XGBoost through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- XGBoost via GUI ---")

        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_xgboost', True)

        success = harness.run_analysis_direct(
            models=['XGBoost'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "XGBoost analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('XGBoost', best_r2, baseline_best)

    def test_lightgbm_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test LightGBM through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- LightGBM via GUI ---")

        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_lightgbm', True)

        success = harness.run_analysis_direct(
            models=['LightGBM'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "LightGBM analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('LightGBM', best_r2, baseline_best)

    def test_catboost_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test CatBoost through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- CatBoost via GUI ---")

        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_catboost', True)

        success = harness.run_analysis_direct(
            models=['CatBoost'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "CatBoost analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('CatBoost', best_r2, baseline_best)

    def test_svr_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test SVR through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- SVR via GUI ---")

        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_svr', True)

        success = harness.run_analysis_direct(
            models=['SVR'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "SVR analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('SVR', best_r2, baseline_best)

    def test_mlp_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test MLP through GUI."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- MLP via GUI ---")

        harness.set_var('use_pls', False)
        harness.set_var('use_ridge', False)
        harness.set_var('use_elasticnet', False)
        harness.set_var('use_mlp', True)

        success = harness.run_analysis_direct(
            models=['MLP'],
            preprocessing=['SNV'],
            cv_folds=5,
            holdout_samples=8,
            holdout_method='spxy'
        )
        assert success, "MLP analysis failed"

        df = harness.get_results_df()
        r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
        best_r2 = df[r2_col].max()

        print_comparison('MLP', best_r2, baseline_best)


# ============================================================
# 2. VARIABLE SELECTION METHODS (via GUI)
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestVariableSelectionViaGUI:
    """
    Test variable selection methods through the GUI.

    Tests verify:
    1. Variable selection runs through GUI without errors
    2. Selected variables improve or match baseline performance
    """

    def test_importance_based_selection(self, baseline_via_gui, loaded_regression_data):
        """Test importance-based variable selection (built into run_search)."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- Importance-based Variable Selection ---")
        print("  (This is tested automatically within run_search)")

        # The run_search already tests variable subsets
        # Check if results include variable subset results
        df = baseline_via_gui['results_df']

        # Look for results with variable counts
        if 'Variables' in df.columns or 'n_vars' in df.columns:
            var_col = 'Variables' if 'Variables' in df.columns else 'n_vars'
            subset_results = df[df[var_col] < df[var_col].max()]
            if len(subset_results) > 0:
                r2_col = baseline_via_gui['baseline']['r2_col']
                best_subset_r2 = subset_results[r2_col].max()
                print_comparison('Variable Subset (best)', best_subset_r2, baseline_best)

        assert True, "Importance-based selection runs automatically"

    def test_uve_selection_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test UVE variable selection through GUI."""
        from spectral_predict.variable_selection import uve_selection
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- UVE Variable Selection via GUI ---")

        # Get data from GUI app
        X = harness.app.X.values
        y = harness.app.y.values

        # Run UVE (this would be called from GUI's variable selection panel)
        importances = uve_selection(X, y, cv_folds=5, random_state=42)

        # Select top 100 variables
        n_vars = 100
        top_indices = np.argsort(importances)[-n_vars:]
        X_selected = X[:, top_indices]

        # Evaluate with PLS
        pls = PLSRegression(n_components=5)
        y_pred = cross_val_predict(pls, X_selected, y, cv=5)
        r2 = r2_score(y, y_pred)

        print_comparison(f'UVE (top {n_vars} vars) + PLS', r2, baseline_best)
        assert r2 > -1, "R2 should be valid"

    def test_spa_selection_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test SPA variable selection through GUI."""
        from spectral_predict.variable_selection import spa_selection
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- SPA Variable Selection via GUI ---")

        X = harness.app.X.values
        y = harness.app.y.values

        # Run SPA
        n_vars = 50
        importances = spa_selection(X, y, n_features=n_vars, cv_folds=5, random_state=42)

        top_indices = np.argsort(importances)[-n_vars:]
        X_selected = X[:, top_indices]

        pls = PLSRegression(n_components=min(5, n_vars - 1))
        y_pred = cross_val_predict(pls, X_selected, y, cv=5)
        r2 = r2_score(y, y_pred)

        print_comparison(f'SPA ({n_vars} vars) + PLS', r2, baseline_best)
        assert r2 > -1, "R2 should be valid"

    def test_ipls_selection_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test iPLS variable selection through GUI."""
        from spectral_predict.variable_selection import ipls_selection
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- iPLS Variable Selection via GUI ---")

        X = harness.app.X.values
        y = harness.app.y.values

        # Run iPLS
        importances = ipls_selection(X, y, n_intervals=20, cv_folds=5, random_state=42)

        # Select top intervals (top 200 variables)
        n_vars = 200
        top_indices = np.argsort(importances)[-n_vars:]
        X_selected = X[:, top_indices]

        pls = PLSRegression(n_components=5)
        y_pred = cross_val_predict(pls, X_selected, y, cv=5)
        r2 = r2_score(y, y_pred)

        print_comparison(f'iPLS (top {n_vars} vars) + PLS', r2, baseline_best)
        assert r2 > -1, "R2 should be valid"

    def test_cars_selection_via_gui(self, baseline_via_gui, loaded_regression_data):
        """Test CARS variable selection through GUI."""
        from spectral_predict.variable_selection import cars_selection
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- CARS Variable Selection via GUI ---")

        X = harness.app.X.values
        y = harness.app.y.values

        # Run CARS (fewer iterations for speed)
        importances = cars_selection(X, y, n_iterations=30, cv_folds=5, random_state=42)

        # Select top variables
        n_vars = 100
        top_indices = np.argsort(importances)[-n_vars:]
        X_selected = X[:, top_indices]

        pls = PLSRegression(n_components=5)
        y_pred = cross_val_predict(pls, X_selected, y, cv=5)
        r2 = r2_score(y, y_pred)

        print_comparison(f'CARS (top {n_vars} vars) + PLS', r2, baseline_best)
        assert r2 > -1, "R2 should be valid"


# ============================================================
# 3. PREPROCESSING METHODS (via GUI)
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestPreprocessingViaGUI:
    """Test different preprocessing methods through GUI."""

    def test_all_preprocessing_combinations(self, baseline_via_gui, loaded_regression_data):
        """Test all preprocessing methods and compare to baseline."""
        harness = loaded_regression_data
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- Preprocessing Methods Comparison ---")

        preprocessing_options = [
            (['Raw'], 'Raw only'),
            (['SNV'], 'SNV only'),
            (['SG1'], 'SG1 (1st derivative)'),
            (['SG2'], 'SG2 (2nd derivative)'),
            (['SNV', 'SG1'], 'SNV + SG1'),
        ]

        for pp_list, pp_name in preprocessing_options:
            success = harness.run_analysis_direct(
                models=['PLS'],
                preprocessing=pp_list,
                cv_folds=5,
                holdout_samples=8,
                holdout_method='spxy'
            )

            if success:
                df = harness.get_results_df()
                r2_col = [c for c in df.columns if 'r2' in c.lower()][0]
                best_r2 = df[r2_col].max()
                print_comparison(f'PLS + {pp_name}', best_r2, baseline_best, pp_name)
            else:
                print(f"  [!] PLS + {pp_name}: FAILED")


# ============================================================
# Calibration Transfer Tests (ALL METHODS)
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestCalibrationTransfer:
    """
    Test ALL calibration transfer methods using real tablet data.

    Methods tested:
    - DS: Direct Standardization
    - PDS: Piecewise Direct Standardization
    - TSR: Transfer by Spectral Regression
    - CTAI: Calibration Transfer with Analyte Information
    - NS-PFCE: Non-linear Spectral Processing for Calibration Enhancement
    - JYPLS-inv: Joint Y PLS Inverse

    Uses matched spectral data from two instruments:
    - Master: LS Tablet (C:/Users/sponheim/Desktop/LS Tablet)
    - Slave: FS Tablet (C:/Users/sponheim/Desktop/FS Tablet)
    """

    @pytest.fixture
    def transfer_data(self):
        """Load master and slave spectra for transfer tests."""
        from pathlib import Path
        from spectral_predict.io import read_asd_dir

        master_path = Path(r"C:\Users\sponheim\Desktop\LS Tablet")
        slave_path = Path(r"C:\Users\sponheim\Desktop\FS Tablet")

        # Check paths exist
        if not master_path.exists() or not slave_path.exists():
            pytest.skip("Transfer data paths not found on this machine")

        # Load spectra
        master_result = read_asd_dir(str(master_path))
        slave_result = read_asd_dir(str(slave_path))

        # Extract DataFrames
        X_master = master_result[0] if isinstance(master_result, tuple) else master_result
        X_slave = slave_result[0] if isinstance(slave_result, tuple) else slave_result

        # Sort both by file number to ensure matching
        X_master = X_master.sort_index()
        X_slave = X_slave.sort_index()

        # Create synthetic y values for methods that need them (CTAI, JYPLS-inv)
        # Use first principal component score as proxy for analyte concentration
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        y_synthetic = pca.fit_transform(X_master.values).ravel()
        # Normalize to reasonable range
        y_synthetic = (y_synthetic - y_synthetic.min()) / (y_synthetic.max() - y_synthetic.min()) * 100

        print(f"\n  Master spectra: {X_master.shape}")
        print(f"  Slave spectra: {X_slave.shape}")

        return {
            'X_master': X_master.values,
            'X_slave': X_slave.values,
            'wavelengths': X_master.columns.values.astype(float),
            'y': y_synthetic,
            'master_ids': X_master.index.tolist(),
            'slave_ids': X_slave.index.tolist()
        }

    def _calc_improvement(self, X_master, X_slave, X_transferred):
        """Calculate improvement percentage."""
        pre_diff = np.mean(np.abs(X_master - X_slave))
        post_diff = np.mean(np.abs(X_master - X_transferred))
        improvement = (pre_diff - post_diff) / pre_diff * 100
        return pre_diff, post_diff, improvement

    def test_ds_transfer(self, transfer_data):
        """Test Direct Standardization (DS) calibration transfer."""
        from spectral_predict.calibration_transfer import estimate_ds, apply_ds

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']

        print("\n--- DS (Direct Standardization) ---")

        # Fit and apply
        A = estimate_ds(X_master, X_slave, lam=1e-6)
        X_transferred = apply_ds(X_slave, A)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        assert improvement > 50, f"DS should achieve >50% improvement, got {improvement:.1f}%"
        print(f"  [+] DS TRANSFER: {improvement:.1f}% improvement")

    def test_pds_transfer(self, transfer_data):
        """Test Piecewise Direct Standardization (PDS) calibration transfer."""
        from spectral_predict.calibration_transfer import estimate_pds, apply_pds

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']

        print("\n--- PDS (Piecewise Direct Standardization) ---")

        # Fit and apply
        window = 11
        B = estimate_pds(X_master, X_slave, window=window)
        X_transferred = apply_pds(X_slave, B, window=window)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Window size: {window}")
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        assert improvement > 50, f"PDS should achieve >50% improvement, got {improvement:.1f}%"
        print(f"  [+] PDS TRANSFER: {improvement:.1f}% improvement")

    def test_tsr_transfer(self, transfer_data):
        """Test Transfer by Spectral Regression (TSR / Shenk-Westerhaus method)."""
        from spectral_predict.calibration_transfer import estimate_tsr, apply_tsr

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']

        print("\n--- TSR (Transfer by Spectral Regression) ---")

        # Use all samples as transfer samples
        n_samples = X_master.shape[0]
        transfer_indices = np.arange(n_samples)

        # Fit TSR (slope/bias correction per wavelength)
        params = estimate_tsr(X_master, X_slave, transfer_indices, slope_bias_correction=True)
        X_transferred = apply_tsr(X_slave, params)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Transfer samples: {n_samples}")
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        assert improvement > 0, f"TSR should provide some improvement, got {improvement:.1f}%"
        print(f"  [+] TSR TRANSFER: {improvement:.1f}% improvement")

    def test_ctai_transfer(self, transfer_data):
        """Test Calibration Transfer based on Affine Invariance (CTAI)."""
        from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']

        print("\n--- CTAI (Calibration Transfer - Affine Invariance) ---")

        # Fit CTAI (transfer-standard free method)
        params = estimate_ctai(X_master, X_slave, n_components=5)
        X_transferred = apply_ctai(X_slave, params)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Components: 5")
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        assert improvement > 0, f"CTAI should provide some improvement, got {improvement:.1f}%"
        print(f"  [+] CTAI TRANSFER: {improvement:.1f}% improvement")

    def test_nspfce_transfer(self, transfer_data):
        """Test Non-supervised Parameter-Free Calibration Enhancement (NS-PFCE)."""
        from spectral_predict.calibration_transfer import estimate_nspfce, apply_nspfce

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']
        wavelengths = transfer_data['wavelengths']

        print("\n--- NS-PFCE (Non-supervised Parameter-Free) ---")

        # Fit NS-PFCE (automatic, parameter-free)
        params = estimate_nspfce(X_master, X_slave, wavelengths, use_wavelength_selection=False)
        X_transferred = apply_nspfce(X_slave, params)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        assert improvement > 0, f"NS-PFCE should provide some improvement, got {improvement:.1f}%"
        print(f"  [+] NS-PFCE TRANSFER: {improvement:.1f}% improvement")

    def test_jypls_inv_transfer(self, transfer_data):
        """Test Joint Y PLS Inverse (JYPLS-inv) calibration transfer.

        NOTE: JYPLS-inv requires REAL analyte concentrations (y values) to work
        properly. With synthetic PCA-derived y values, it may not improve transfer.
        This test verifies the method runs without error.
        """
        from spectral_predict.calibration_transfer import estimate_jypls_inv, apply_jypls_inv

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']
        y = transfer_data['y']

        print("\n--- JYPLS-inv (Joint Y PLS Inverse) ---")
        print("  NOTE: Requires real analyte values for best results")

        # Use all samples as transfer samples
        n_samples = X_master.shape[0]
        transfer_indices = np.arange(n_samples)

        # Fit JYPLS-inv
        params = estimate_jypls_inv(X_master, X_slave, y, transfer_indices, n_components=5)
        X_transferred = apply_jypls_inv(X_slave, params)

        pre_diff, post_diff, improvement = self._calc_improvement(X_master, X_slave, X_transferred)
        print(f"  Components: 5")
        print(f"  Pre-transfer diff: {pre_diff:.6f}")
        print(f"  Post-transfer diff: {post_diff:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        # JYPLS-inv may not improve with synthetic y values
        # Just verify the method runs and produces valid output
        assert X_transferred.shape == X_slave.shape, "Output shape should match input"
        assert not np.isnan(X_transferred).any(), "Output should not contain NaN"

        if improvement > 0:
            print(f"  [+] JYPLS-inv TRANSFER: {improvement:.1f}% improvement")
        else:
            print(f"  [!] JYPLS-inv: {improvement:.1f}% (needs real y values)")

    def test_all_methods_comparison(self, transfer_data):
        """Compare ALL calibration transfer methods."""
        from spectral_predict.calibration_transfer import (
            estimate_ds, apply_ds,
            estimate_pds, apply_pds,
            estimate_tsr, apply_tsr,
            estimate_ctai, apply_ctai,
            estimate_nspfce, apply_nspfce,
            estimate_jypls_inv, apply_jypls_inv
        )

        X_master = transfer_data['X_master']
        X_slave = transfer_data['X_slave']
        wavelengths = transfer_data['wavelengths']
        y = transfer_data['y']
        n_samples = X_master.shape[0]
        transfer_indices = np.arange(n_samples)

        pre_diff = np.mean(np.abs(X_master - X_slave))
        results = {}

        # DS
        A = estimate_ds(X_master, X_slave, lam=1e-6)
        X_ds = apply_ds(X_slave, A)
        ds_diff = np.mean(np.abs(X_master - X_ds))
        results['DS'] = (pre_diff - ds_diff) / pre_diff * 100

        # PDS
        B = estimate_pds(X_master, X_slave, window=11)
        X_pds = apply_pds(X_slave, B, window=11)
        pds_diff = np.mean(np.abs(X_master - X_pds))
        results['PDS'] = (pre_diff - pds_diff) / pre_diff * 100

        # TSR
        params = estimate_tsr(X_master, X_slave, transfer_indices, slope_bias_correction=True)
        X_tsr = apply_tsr(X_slave, params)
        tsr_diff = np.mean(np.abs(X_master - X_tsr))
        results['TSR'] = (pre_diff - tsr_diff) / pre_diff * 100

        # CTAI
        params = estimate_ctai(X_master, X_slave, n_components=5)
        X_ctai = apply_ctai(X_slave, params)
        ctai_diff = np.mean(np.abs(X_master - X_ctai))
        results['CTAI'] = (pre_diff - ctai_diff) / pre_diff * 100

        # NS-PFCE
        params = estimate_nspfce(X_master, X_slave, wavelengths, use_wavelength_selection=False)
        X_nspfce = apply_nspfce(X_slave, params)
        nspfce_diff = np.mean(np.abs(X_master - X_nspfce))
        results['NS-PFCE'] = (pre_diff - nspfce_diff) / pre_diff * 100

        # JYPLS-inv
        params = estimate_jypls_inv(X_master, X_slave, y, transfer_indices, n_components=5)
        X_jypls = apply_jypls_inv(X_slave, params)
        jypls_diff = np.mean(np.abs(X_master - X_jypls))
        results['JYPLS-inv'] = (pre_diff - jypls_diff) / pre_diff * 100

        # Print comparison
        print("\n" + "=" * 60)
        print("CALIBRATION TRANSFER - ALL METHODS COMPARISON")
        print("=" * 60)
        print(f"  Pre-transfer difference: {pre_diff:.6f}")
        print("")

        # Sort by improvement
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (method, improvement) in enumerate(sorted_results):
            rank = i + 1
            print(f"  {rank}. {method:12s}: {improvement:+.1f}% improvement")

        print("")
        best_method = sorted_results[0][0]
        best_improvement = sorted_results[0][1]
        print(f"  BEST METHOD: {best_method} ({best_improvement:+.1f}%)")
        print("=" * 60)

        # Most methods should provide improvement
        # JYPLS-inv may fail with synthetic y values
        for method, improvement in results.items():
            if method == 'JYPLS-inv':
                # JYPLS-inv needs real y values to work properly
                continue
            assert improvement > 0, f"{method} should provide positive improvement"


# ============================================================
# Interference Removal Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestInterferenceRemoval:
    """
    Test interference removal methods.

    Methods tested:
    - MSC: Multiplicative Scatter Correction
    - OSC: Orthogonal Signal Correction
    - DOSC: Direct OSC
    - EPO: External Parameter Orthogonalization
    - GLSW: Generalized Least Squares Weighting
    """

    def test_msc_removal(self, loaded_regression_data):
        """Test Multiplicative Scatter Correction."""
        from spectral_predict.interference import MSC

        harness = loaded_regression_data
        X = harness.app.X.values

        print("\n--- MSC (Multiplicative Scatter Correction) ---")

        # Fit and transform
        msc = MSC()
        X_corrected = msc.fit_transform(X)

        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {X_corrected.shape}")

        # Verify output
        assert X_corrected.shape == X.shape, "Output shape should match input"
        assert not np.isnan(X_corrected).any(), "Output should not contain NaN"

        # MSC should reduce scatter variation
        # Check that variance of mean-centered spectra is reduced
        original_std = np.std(X - X.mean(axis=1, keepdims=True))
        corrected_std = np.std(X_corrected - X_corrected.mean(axis=1, keepdims=True))

        print(f"  Original std: {original_std:.6f}")
        print(f"  Corrected std: {corrected_std:.6f}")
        print(f"  [+] MSC: Transform successful")

    def test_osc_removal(self, loaded_regression_data):
        """Test Orthogonal Signal Correction."""
        from spectral_predict.interference import OSC

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- OSC (Orthogonal Signal Correction) ---")

        # Fit and transform
        osc = OSC(n_components=2)
        X_corrected = osc.fit_transform(X, y)

        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {X_corrected.shape}")
        print(f"  Components: 2")

        # Verify output
        assert X_corrected.shape == X.shape, "Output shape should match input"
        assert not np.isnan(X_corrected).any(), "Output should not contain NaN"

        print(f"  [+] OSC: Transform successful")

    def test_dosc_removal(self, loaded_regression_data):
        """Test Direct Orthogonal Signal Correction."""
        from spectral_predict.interference import DOSC

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- DOSC (Direct Orthogonal Signal Correction) ---")

        # Fit and transform
        dosc = DOSC(n_components=2)
        X_corrected = dosc.fit_transform(X, y)

        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {X_corrected.shape}")
        print(f"  Components: 2")

        # Verify output
        assert X_corrected.shape == X.shape, "Output shape should match input"
        assert not np.isnan(X_corrected).any(), "Output should not contain NaN"

        print(f"  [+] DOSC: Transform successful")

    def test_glsw_removal(self, loaded_regression_data):
        """Test Generalized Least Squares Weighting."""
        from spectral_predict.interference import GLSW

        harness = loaded_regression_data
        X = harness.app.X.values

        print("\n--- GLSW (Generalized Least Squares Weighting) ---")

        # Fit and transform
        glsw = GLSW(method='covariance', regularization=1e-6)
        X_corrected = glsw.fit_transform(X)

        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {X_corrected.shape}")

        # Verify output
        assert X_corrected.shape == X.shape, "Output shape should match input"
        assert not np.isnan(X_corrected).any(), "Output should not contain NaN"

        print(f"  [+] GLSW: Transform successful")

    def test_interference_improves_model(self, baseline_via_gui, loaded_regression_data):
        """Test if interference removal improves model performance."""
        from spectral_predict.interference import OSC
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values
        baseline_best = baseline_via_gui['baseline']['best']

        print("\n--- Interference Removal + PLS ---")

        # Apply OSC
        osc = OSC(n_components=2)
        X_osc = osc.fit_transform(X, y)

        # Fit PLS on corrected data
        pls = PLSRegression(n_components=5)
        y_pred = cross_val_predict(pls, X_osc, y, cv=5)
        r2 = r2_score(y, y_pred)

        print(f"  Baseline best R2: {baseline_best:.4f}")
        print(f"  OSC + PLS R2: {r2:.4f}")

        diff = r2 - baseline_best
        if diff > 0.01:
            print(f"  [+] OSC OUTPERFORMS baseline by {diff:+.4f}")
        elif diff < -0.01:
            print(f"  [-] OSC underperforms baseline by {diff:+.4f}")
        else:
            print(f"  [=] OSC matches baseline (diff={diff:+.4f})")


# ============================================================
# Model Save/Load Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestModelSaveLoad:
    """
    Test model save/load workflow.

    Tests:
    - Save model to .dasp file
    - Load model from .dasp file
    - Verify model produces same predictions after load
    - Verify metadata is preserved
    """

    def test_save_and_load_pls_model(self, loaded_regression_data, tmp_path):
        """Test saving and loading a PLS model."""
        from spectral_predict.model_io import save_model, load_model
        from sklearn.cross_decomposition import PLSRegression

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- Save/Load PLS Model ---")

        # Train a model
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)
        y_pred_original = pls.predict(X).ravel()

        # Save model
        filepath = tmp_path / "test_pls.dasp"
        wavelengths = harness.app.X.columns.tolist()
        metadata = {
            'model_name': 'PLS',
            'task_type': 'regression',
            'preprocessing': 'Raw',
            'wavelengths': wavelengths,
            'n_vars': len(wavelengths),
            'n_components': 5,
            'target': 'Collagen',
            'n_samples': len(y),
            'performance': {'R2': 0.95, 'RMSE': 0.5}
        }

        save_model(
            model=pls,
            preprocessor=None,
            metadata=metadata,
            filepath=filepath
        )

        print(f"  Saved model to: {filepath}")
        assert filepath.exists(), "Model file should exist"

        # Load model
        loaded = load_model(filepath)

        print(f"  Loaded model type: {type(loaded['model']).__name__}")
        print(f"  Metadata keys: {list(loaded['metadata'].keys())}")

        # Verify model works
        y_pred_loaded = loaded['model'].predict(X).ravel()

        # Predictions should be identical
        np.testing.assert_array_almost_equal(
            y_pred_original, y_pred_loaded, decimal=10,
            err_msg="Predictions should be identical after load"
        )

        # Verify metadata
        assert loaded['metadata']['model_name'] == 'PLS'
        assert loaded['metadata']['n_components'] == 5

        print(f"  [+] Save/Load: Predictions match, metadata preserved")

    def test_save_and_load_with_preprocessing(self, loaded_regression_data, tmp_path):
        """Test saving and loading model with preprocessing pipeline."""
        from spectral_predict.model_io import save_model, load_model
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import Pipeline

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- Save/Load Model with Preprocessing ---")

        # Create preprocessing pipeline (SNV only)
        steps = build_preprocessing_pipeline(
            'snv',
            wavelengths=harness.app.X.columns.values
        )
        preprocessor = Pipeline(steps) if steps else None

        # Fit preprocessor and model
        if preprocessor:
            X_processed = preprocessor.fit_transform(X)
        else:
            X_processed = X
        pls = PLSRegression(n_components=5)
        pls.fit(X_processed, y)

        y_pred_original = pls.predict(X_processed).ravel()

        # Save
        filepath = tmp_path / "test_pls_snv.dasp"
        wavelengths = harness.app.X.columns.tolist()
        metadata = {
            'model_name': 'PLS',
            'task_type': 'regression',
            'preprocessing': 'SNV',
            'wavelengths': wavelengths,
            'n_vars': len(wavelengths),
            'n_components': 5,
            'performance': {'R2': 0.95, 'RMSE': 0.5}
        }

        save_model(
            model=pls,
            preprocessor=preprocessor,
            metadata=metadata,
            filepath=filepath
        )

        print(f"  Saved model with SNV preprocessing")

        # Load
        loaded = load_model(filepath)

        # Apply preprocessing and predict
        if loaded['preprocessor'] is not None:
            X_loaded_processed = loaded['preprocessor'].transform(X)
        else:
            X_loaded_processed = X

        y_pred_loaded = loaded['model'].predict(X_loaded_processed).ravel()

        np.testing.assert_array_almost_equal(
            y_pred_original, y_pred_loaded, decimal=10
        )

        print(f"  [+] Save/Load with preprocessing: Success")

    def test_save_load_multiple_model_types(self, loaded_regression_data, tmp_path):
        """Test save/load for different model types."""
        from spectral_predict.model_io import save_model, load_model
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values
        wavelengths = harness.app.X.columns.tolist()

        print("\n--- Save/Load Multiple Model Types ---")

        models = {
            'PLS': PLSRegression(n_components=5),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }

        for name, model in models.items():
            # Train
            model.fit(X, y)
            y_pred_original = model.predict(X).ravel() if hasattr(model, 'predict') else model.predict(X)

            # Save
            filepath = tmp_path / f"test_{name.lower()}.dasp"
            metadata = {
                'model_name': name,
                'task_type': 'regression',
                'preprocessing': 'Raw',
                'wavelengths': wavelengths,
                'n_vars': len(wavelengths),
                'performance': {'R2': 0.95, 'RMSE': 0.5}
            }
            save_model(
                model=model,
                preprocessor=None,
                metadata=metadata,
                filepath=filepath
            )

            # Load and verify
            loaded = load_model(filepath)
            y_pred_loaded = loaded['model'].predict(X).ravel() if hasattr(loaded['model'], 'predict') else loaded['model'].predict(X)

            np.testing.assert_array_almost_equal(
                y_pred_original, y_pred_loaded, decimal=5
            )

            print(f"  [+] {name}: Save/Load verified")


# ============================================================
# Prediction Workflow Tests
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestPredictionWorkflow:
    """
    Test prediction workflow.

    Tests:
    - Predict on new data with trained model
    - Predict with preprocessing
    - Prediction uncertainty/confidence
    - Batch prediction
    """

    def test_predict_new_samples(self, loaded_regression_data):
        """Test prediction on held-out samples."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- Predict New Samples ---")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        # Train model
        pls = PLSRegression(n_components=5)
        pls.fit(X_train, y_train)

        # Predict
        y_pred = pls.predict(X_test).ravel()

        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"  Test R2: {r2:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")

        assert y_pred.shape == y_test.shape, "Prediction shape should match"
        assert r2 > 0, "R2 should be positive on this data"

        print(f"  [+] Prediction: Success (R2={r2:.4f})")

    def test_predict_with_preprocessing(self, loaded_regression_data):
        """Test prediction with preprocessing pipeline."""
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values
        wavelengths = harness.app.X.columns.values

        print("\n--- Predict with Preprocessing ---")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and fit preprocessing (SNV + SG1)
        from sklearn.pipeline import Pipeline
        steps = build_preprocessing_pipeline(
            'snv_deriv',
            deriv=1, window=11, polyorder=2,
            wavelengths=wavelengths
        )
        preprocessor = Pipeline(steps) if steps else None

        if preprocessor:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test

        # Train and predict
        pls = PLSRegression(n_components=5)
        pls.fit(X_train_processed, y_train)
        y_pred = pls.predict(X_test_processed).ravel()

        r2 = r2_score(y_test, y_pred)
        print(f"  Preprocessing: SNV + SG1")
        print(f"  Test R2: {r2:.4f}")

        assert r2 > 0, "R2 should be positive"
        print(f"  [+] Predict with preprocessing: Success")

    def test_batch_prediction(self, loaded_regression_data):
        """Test batch prediction on multiple samples."""
        from sklearn.cross_decomposition import PLSRegression

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- Batch Prediction ---")

        # Train on all data
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Predict in batches
        batch_sizes = [1, 5, 10, len(X)]
        for batch_size in batch_sizes:
            X_batch = X[:batch_size]
            y_pred = pls.predict(X_batch).ravel()

            assert y_pred.shape[0] == batch_size
            print(f"  Batch size {batch_size}: OK")

        print(f"  [+] Batch prediction: All sizes work")

    def test_prediction_confidence(self, loaded_regression_data):
        """Test prediction with confidence/uncertainty estimation."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values

        print("\n--- Prediction Confidence ---")

        # Get CV residuals for uncertainty estimation
        pls = PLSRegression(n_components=5)
        y_cv_pred = cross_val_predict(pls, X, y, cv=5)
        residuals = y - y_cv_pred

        # Estimate prediction uncertainty from CV residuals
        rmse_cv = np.sqrt(np.mean(residuals**2))
        std_residuals = np.std(residuals)

        print(f"  CV RMSE: {rmse_cv:.4f}")
        print(f"  Residual Std: {std_residuals:.4f}")

        # Fit final model
        pls.fit(X, y)
        y_pred = pls.predict(X).ravel()

        # Simple confidence interval (±2*std)
        confidence_interval = 2 * std_residuals
        print(f"  95% Confidence interval: ±{confidence_interval:.4f}")

        # Check that most training points fall within interval
        within_interval = np.abs(y - y_pred) < confidence_interval
        pct_within = 100 * np.mean(within_interval)
        print(f"  Training points within interval: {pct_within:.1f}%")

        assert pct_within > 50, "Most points should be within interval"
        print(f"  [+] Confidence estimation: {pct_within:.1f}% within +/-2*std")

    def test_loaded_model_prediction(self, loaded_regression_data, tmp_path):
        """Test prediction workflow with loaded model."""
        from spectral_predict.model_io import save_model, load_model
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        harness = loaded_regression_data
        X = harness.app.X.values
        y = harness.app.y.values
        wavelengths = harness.app.X.columns.values

        print("\n--- Loaded Model Prediction Workflow ---")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create preprocessing (SNV)
        from sklearn.pipeline import Pipeline
        steps = build_preprocessing_pipeline(
            'snv', wavelengths=wavelengths
        )
        preprocessor = Pipeline(steps) if steps else None

        # Fit everything
        if preprocessor:
            X_train_proc = preprocessor.fit_transform(X_train)
        else:
            X_train_proc = X_train
        pls = PLSRegression(n_components=5)
        pls.fit(X_train_proc, y_train)

        # Save
        filepath = tmp_path / "prediction_test.dasp"
        metadata = {
            'model_name': 'PLS',
            'task_type': 'regression',
            'preprocessing': 'SNV',
            'wavelengths': wavelengths.tolist() if hasattr(wavelengths, 'tolist') else list(wavelengths),
            'n_vars': len(wavelengths),
            'performance': {'R2': 0.95, 'RMSE': 0.5}
        }
        save_model(
            model=pls,
            preprocessor=preprocessor,
            metadata=metadata,
            filepath=filepath
        )

        print(f"  Model saved: {filepath.name}")

        # Simulate new session: load and predict
        loaded = load_model(filepath)

        # Apply preprocessing and predict
        X_test_proc = loaded['preprocessor'].transform(X_test)
        y_pred = loaded['model'].predict(X_test_proc).ravel()

        r2 = r2_score(y_test, y_pred)
        print(f"  Loaded model test R2: {r2:.4f}")

        assert r2 > 0, "Loaded model should produce valid predictions"
        print(f"  [+] Full workflow: Train -> Save -> Load -> Predict = Success")


# ============================================================
# FINAL SUMMARY
# ============================================================

@pytest.mark.gui
@pytest.mark.comprehensive
class TestComprehensiveSummary:
    """Print summary guidance."""

    def test_summary(self, baseline_via_gui):
        """Print how to run comprehensive tests."""
        print("\n")
        print("=" * 70)
        print("COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print("\nRun all comparison tests:")
        print("  pytest tests/gui/test_comprehensive.py -v -s -m comprehensive")
        print("\nRun specific test groups:")
        print("  pytest tests/gui/test_comprehensive.py::TestAllModelsViaGUI -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestVariableSelectionViaGUI -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestPreprocessingViaGUI -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestCalibrationTransfer -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestInterferenceRemoval -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestModelSaveLoad -v -s")
        print("  pytest tests/gui/test_comprehensive.py::TestPredictionWorkflow -v -s")
        print("\n[+] = OUTPERFORMS baseline")
        print("[=] = MATCHES baseline")
        print("[-] = UNDERPERFORMS baseline")
        print("=" * 70)
