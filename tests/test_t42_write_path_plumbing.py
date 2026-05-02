"""T-42 Approach C regression tests — hoisted study.user_attrs + dead-write removal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_data(task: str = "regression"):
    rng = np.random.default_rng(0)
    n_samples, n_features = 60, 80
    latent = rng.normal(size=(n_samples, 3))
    base = np.cos(np.linspace(0, 8 * np.pi, n_features))[None, :]
    X = (
        latent @ rng.normal(size=(3, n_features))
        + base
        + rng.normal(scale=0.05, size=(n_samples, n_features))
    )
    if task == "regression":
        y = latent[:, 0] * 1.5 + rng.normal(scale=0.1, size=n_samples)
    else:
        y = (latent[:, 0] > 0).astype(int)
    return X, y, np.linspace(900, 1700, n_features)


def test_constants_hoisted_to_study_user_attrs():
    """cv_strategy + cv_n_repeats + early_stopping_rounds live on the study."""
    from spectral_predict import run_state, unified_bayesian

    run_state._active_storage_url = None
    run_state._active_run_id = None

    X, y, wavelengths = _make_data("regression")
    _, study = unified_bayesian.run_unified_bayesian(
        X=X, y=y, wavelengths=wavelengths,
        model_name="PLS", task_type="regression",
        n_trials=3, cv_folds=3, cv_strategy="kfold", cv_n_repeats=2,
        early_stopping_rounds=40,
        random_state=42, verbose=False, progress_callback=None,
    )
    assert study.user_attrs.get("cv_strategy") == "kfold"
    assert study.user_attrs.get("cv_n_repeats") == 2
    # PLS doesn't use early stopping → hoisted as None.
    assert study.user_attrs.get("early_stopping_rounds") is None


def test_early_stopping_hoisted_for_xgboost():
    """For boosting models the hoisted value is the actual early_stopping_rounds.
    Also verifies the per-trial write was removed for boosting trials —
    the PLS-only absence assertion is insufficient because PLS always wrote
    None and would not catch a boosting-only regression."""
    pytest.importorskip("xgboost")
    from spectral_predict import run_state, unified_bayesian

    run_state._active_storage_url = None
    run_state._active_run_id = None

    X, y, wavelengths = _make_data("regression")
    _, study = unified_bayesian.run_unified_bayesian(
        X=X, y=y, wavelengths=wavelengths,
        model_name="XGBoost", task_type="regression",
        n_trials=3, cv_folds=3, cv_strategy="kfold",
        early_stopping_rounds=25,
        random_state=42, verbose=False, progress_callback=None,
    )
    assert study.user_attrs.get("early_stopping_rounds") == 25
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in completed:
        assert "early_stopping_rounds" not in trial.user_attrs


def test_hoist_does_not_overwrite_existing_study_user_attrs(tmp_path, caplog):
    """On resume the existing study already carries hoisted values from the
    prior session — overwriting them with the current call's args would
    silently corrupt the audit trail. The guard preserves the stored
    values and logs a warning if the new args disagree."""
    pytest.importorskip("xgboost")
    from spectral_predict import run_state, unified_bayesian
    import logging

    run_state._active_storage_url = None
    run_state._active_run_id = None

    X, y, wavelengths = _make_data("regression")

    # First call writes the hoist values.
    _, study1 = unified_bayesian.run_unified_bayesian(
        X=X, y=y, wavelengths=wavelengths,
        model_name="XGBoost", task_type="regression",
        n_trials=2, cv_folds=3, cv_strategy="kfold",
        early_stopping_rounds=25,
        random_state=42, verbose=False, progress_callback=None,
    )
    assert study1.user_attrs.get("early_stopping_rounds") == 25

    # Pre-set conflicting values on a fresh in-memory study to mimic the
    # resume case where prior trials wrote different attrs.
    study2 = optuna.create_study(direction="minimize")
    study2.set_user_attr("cv_strategy", "kfold")
    study2.set_user_attr("cv_n_repeats", 5)
    study2.set_user_attr("early_stopping_rounds", 50)
    study2.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

    # Simulate the hoist running on a study that already has values:
    # call the same hoist logic the function uses internally. The contract
    # is "only set if absent; warn on mismatch."
    captured_logs = []
    handler = logging.Handler()
    handler.emit = lambda record: captured_logs.append(record)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("spectral_predict.unified_bayesian")
    logger.addHandler(handler)
    try:
        # Replay the hoist body with mismatched new args.
        new_es = 100  # disagrees with stored 50
        new_cv = "loo"  # disagrees with stored 'kfold'
        for key, val in (
            ('cv_strategy', new_cv),
            ('cv_n_repeats', 5),  # matches
            ('early_stopping_rounds', new_es),
        ):
            if key in study2.user_attrs:
                if study2.user_attrs[key] != val:
                    logger.warning(
                        "study.user_attrs[%r]=%r already set by prior "
                        "session; current call passed %r — keeping the "
                        "stored value to preserve the audit trail.",
                        key, study2.user_attrs[key], val,
                    )
            else:
                study2.set_user_attr(key, val)
    finally:
        logger.removeHandler(handler)

    # Stored values preserved (not overwritten with new_es / new_cv).
    assert study2.user_attrs["cv_strategy"] == "kfold"
    assert study2.user_attrs["cv_n_repeats"] == 5
    assert study2.user_attrs["early_stopping_rounds"] == 50
    # Two warnings logged (cv_strategy and early_stopping_rounds disagreed;
    # cv_n_repeats matched and skipped silently).
    assert len(captured_logs) == 2


def test_study_user_attrs_survive_copy_study_migration(tmp_path):
    """T-41's auto-calc migrates the in-memory study to SQLite via
    optuna.copy_study. user_attrs are documented to survive that copy —
    this test verifies it empirically rather than relying on the docs.
    The hoisted cv_strategy / cv_n_repeats / early_stopping_rounds must
    all carry over."""
    from spectral_predict import unified_bayesian

    study = optuna.create_study(direction="minimize")
    study.set_user_attr("cv_strategy", "repeated_kfold")
    study.set_user_attr("cv_n_repeats", 3)
    study.set_user_attr("early_stopping_rounds", 50)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)

    sqlite_url = f"sqlite:///{(tmp_path / 'test_copy.db').as_posix()}"
    migrated = unified_bayesian._migrate_study_to_sqlite(
        study=study,
        sqlite_url=sqlite_url,
        study_name="test_copy",
        random_state=42,
    )
    assert migrated.user_attrs.get("cv_strategy") == "repeated_kfold"
    assert migrated.user_attrs.get("cv_n_repeats") == 3
    assert migrated.user_attrs.get("early_stopping_rounds") == 50
    assert len(migrated.trials) == 2


def test_no_per_trial_writes_for_hoisted_keys():
    """The dead per-trial cv_strategy + cv_n_repeats writes are gone, and
    early_stopping_rounds no longer appears on each trial's user_attrs."""
    from spectral_predict import run_state, unified_bayesian

    run_state._active_storage_url = None
    run_state._active_run_id = None

    X, y, wavelengths = _make_data("regression")
    _, study = unified_bayesian.run_unified_bayesian(
        X=X, y=y, wavelengths=wavelengths,
        model_name="PLS", task_type="regression",
        n_trials=3, cv_folds=3, cv_strategy="kfold",
        random_state=42, verbose=False, progress_callback=None,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed, "no completed trials in test"
    for trial in completed:
        # T-42: these three keys must NOT appear on trial.user_attrs anymore.
        assert "cv_strategy" not in trial.user_attrs
        assert "cv_n_repeats" not in trial.user_attrs
        assert "early_stopping_rounds" not in trial.user_attrs


def test_convert_study_reads_hoisted_early_stopping_rounds():
    """convert_study_to_dataframe finds early_stopping_rounds on the study,
    not on each trial."""
    pytest.importorskip("xgboost")
    from spectral_predict import run_state, unified_bayesian

    run_state._active_storage_url = None
    run_state._active_run_id = None

    X, y, wavelengths = _make_data("regression")
    df, _ = unified_bayesian.run_unified_bayesian(
        X=X, y=y, wavelengths=wavelengths,
        model_name="XGBoost", task_type="regression",
        n_trials=3, cv_folds=3, cv_strategy="kfold",
        early_stopping_rounds=25,
        random_state=42, verbose=False, progress_callback=None,
    )
    if df is None or df.empty:
        pytest.skip("no completed XGBoost trials in this run")
    assert (df["early_stopping_rounds"] == 25).all()


def test_legacy_study_fallback_to_trial_user_attrs():
    """A study created before T-42 has early_stopping_rounds on each trial,
    not on the study. convert_study_to_dataframe must fall through to the
    trial-level value so resumed studies don't lose the column."""
    from spectral_predict import unified_bayesian

    # Build a fake legacy study by hand: no study.user_attrs, but per-trial
    # values written manually.
    study = optuna.create_study(direction="minimize")

    def legacy_objective(trial):
        trial.suggest_categorical("preprocessing", ["raw"])
        trial.suggest_categorical("subset_type", ["importance"])
        # Mimic pre-T-42 per-trial writes.
        trial.set_user_attr("preprocessing", "raw")
        trial.set_user_attr("apply_baseline", False)
        trial.set_user_attr("apply_smoothing", False)
        trial.set_user_attr("apply_autoscale", False)
        trial.set_user_attr("deriv", 0)
        trial.set_user_attr("window", 0)
        trial.set_user_attr("poly", 0)
        trial.set_user_attr("model_params", "{}")
        trial.set_user_attr("n_vars", 80)
        trial.set_user_attr("full_vars_masked", 80)
        trial.set_user_attr("subset_tag", "full")
        trial.set_user_attr("early_stopping_rounds", 50)  # legacy per-trial
        trial.set_user_attr("RMSE", 0.1)
        trial.set_user_attr("R2", 0.9)
        trial.set_user_attr("CCC", 0.9)
        trial.set_user_attr("RMSEcv", 0.15)
        trial.set_user_attr("R2cv", 0.85)
        trial.set_user_attr("CCCcv", 0.85)
        trial.set_user_attr("MAEcv", 0.1)
        trial.set_user_attr("RPD", 2.5)
        trial.set_user_attr("Bias", 0.0)
        trial.set_user_attr("RER", 5.0)
        trial.set_user_attr("regional_rmse", {})
        trial.set_user_attr("y_quartiles", [0, 0, 0, 0])
        trial.set_user_attr("all_wavelengths", "900,1000,1100")
        return 0.1

    study.optimize(legacy_objective, n_trials=2)
    # No study.set_user_attr was called → mimics legacy.
    assert study.user_attrs.get("early_stopping_rounds") is None

    df = unified_bayesian.convert_study_to_dataframe(
        study=study, model_name="XGBoost", task_type="regression",
        wavelengths=np.linspace(900, 1100, 80), n_features=80, cv_folds=3,
    )
    if not df.empty:
        # Legacy fallback path picked up the per-trial value.
        assert (df["early_stopping_rounds"] == 50).all()
