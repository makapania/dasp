"""T-17 F7: save -> load -> predict round-trip for multi-target models.

Mirrors ``tests/test_autoscale_save_load_roundtrip.py``. Verifies that a
multi-target (multi-Y) ``.dasp`` file persists everything needed to reproduce
per-target RAW-unit predictions after reload:

* a JOINT multi-target model is fit on Y scaled by a full-training-set
  ``FoldYScaler``; the scaler's per-target mean/std must survive save->load so
  ``inverse_transform`` still yields RAW units (verification item 5c(b));
* the reloaded estimator's raw ``predict()`` is bit-identical to pre-save
  (5c(a));
* ``target_names`` + ``per_target_metrics`` survive in order (5c(c));
* ``predict_with_model`` returns RAW target units for a JOINT model (applies the
  persisted Y-scaler) but stays byte-identical for single-target models (they
  carry no Y-scaler, so the inverse-transform branch is skipped).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from spectral_predict.model_io import save_model, load_model, predict_with_model
from spectral_predict.multi_y import FoldYScaler, multi_y_metrics


@pytest.fixture
def synthetic_multi_y():
    """Deterministic 3-target spectral regression dataset with divergent scales."""
    rng = np.random.default_rng(7)
    n_samples, n_features = 45, 24
    latent = rng.normal(size=(n_samples, 2))
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    X = np.outer(latent[:, 0], base) + rng.normal(scale=0.1, size=(n_samples, n_features))
    # Deliberately divergent target variances (std ~0.1, ~1, ~100) to catch a
    # units bug: a scaled-vs-raw mistake shows up in RMSE (scale-sensitive).
    y0 = 0.1 * latent[:, 0] + 0.01 * rng.normal(size=n_samples)
    y1 = 1.0 * latent[:, 0] + 0.1 * latent[:, 1] + 0.1 * rng.normal(size=n_samples)
    y2 = 100.0 * latent[:, 0] + 5.0 * rng.normal(size=n_samples)
    Y = np.column_stack([y0, y1, y2])
    return X, Y


def _train_joint_pls(X, Y):
    """Fit a JOINT PLS on full-training-set-scaled Y; return (model, y_scaler)."""
    y_scaler = FoldYScaler().fit(Y)
    model = PLSRegression(n_components=3, scale=False)
    model.fit(X, y_scaler.transform(Y))
    return model, y_scaler


def _metadata(target_names, per_target_metrics, mode, n_features):
    return {
        "model_name": "PLS",
        "task_type": "regression",
        "preprocessing": "raw",
        "wavelengths": [float(i) for i in range(n_features)],
        "n_vars": n_features,
        "n_samples": 45,
        "performance": {"joint_q2": float(np.mean([m["q2"] for m in per_target_metrics]))},
        "multitarget_mode": mode,
        "n_targets": len(target_names),
        "target_names": list(target_names),
        "prediction_columns": [f"{t}_pred" for t in target_names],
        "per_target_metrics": per_target_metrics,
    }


class TestMultiTargetSaveLoadRoundtrip:
    def test_reloaded_model_predict_bit_identical(self, synthetic_multi_y, tmp_path):
        """5c(a): reloaded estimator raw predict() == pre-save predict()."""
        X, Y = synthetic_multi_y
        model, y_scaler = _train_joint_pls(X, Y)
        target_names = ["yield", "irsf", "carbonate"]
        metrics = multi_y_metrics(Y, model.predict(X), target_names=target_names)

        pred_before = model.predict(X)
        filepath = tmp_path / "mt_joint.dasp"
        save_model(
            model=model, preprocessor=None,
            metadata=_metadata(target_names, metrics["per_target"], "JOINT", X.shape[1]),
            filepath=filepath, y_scaler=y_scaler,
        )
        loaded = load_model(filepath)
        pred_after = loaded["model"].predict(X)
        np.testing.assert_array_equal(pred_before, pred_after)

    def test_y_scaler_stats_reload_and_inverse_transform(self, synthetic_multi_y, tmp_path):
        """5c(b): per-target Y-scaler mean/std reload; inverse_transform matches."""
        X, Y = synthetic_multi_y
        model, y_scaler = _train_joint_pls(X, Y)
        target_names = ["yield", "irsf", "carbonate"]
        metrics = multi_y_metrics(Y, model.predict(X), target_names=target_names)

        filepath = tmp_path / "mt_scaler.dasp"
        save_model(
            model=model, preprocessor=None,
            metadata=_metadata(target_names, metrics["per_target"], "JOINT", X.shape[1]),
            filepath=filepath, y_scaler=y_scaler,
        )
        loaded = load_model(filepath)
        loaded_scaler = loaded["y_scaler"]
        assert loaded_scaler is not None
        np.testing.assert_array_equal(loaded_scaler.mean_, y_scaler.mean_)
        np.testing.assert_array_equal(loaded_scaler.std_, y_scaler.std_)

        scaled_pred = model.predict(X)
        raw_before = y_scaler.inverse_transform(scaled_pred)
        raw_after = loaded_scaler.inverse_transform(scaled_pred)
        np.testing.assert_array_equal(raw_before, raw_after)

    def test_predict_with_model_returns_raw_units_for_joint(self, synthetic_multi_y, tmp_path):
        """predict_with_model applies the persisted Y-scaler -> RAW units, and the
        RAW-unit per-target RMSE matches recomputing from inverse-transformed
        predictions (NOT the scaled-unit value). Pins inverse-transform precedes
        per-target reporting (verification item 4)."""
        X, Y = synthetic_multi_y
        model, y_scaler = _train_joint_pls(X, Y)
        target_names = ["yield", "irsf", "carbonate"]
        metrics = multi_y_metrics(Y, model.predict(X), target_names=target_names)

        filepath = tmp_path / "mt_predict.dasp"
        save_model(
            model=model, preprocessor=None,
            metadata=_metadata(target_names, metrics["per_target"], "JOINT", X.shape[1]),
            filepath=filepath, y_scaler=y_scaler,
        )
        loaded = load_model(filepath)
        raw_pred = predict_with_model(loaded, X, validate_wavelengths=False)

        expected_raw = y_scaler.inverse_transform(model.predict(X))
        np.testing.assert_array_equal(raw_pred, expected_raw)

        # Per-target RMSE in RAW units tracks each target's own scale (~0.1, ~1,
        # ~100) — impossible if predictions were left in scaled (~unit) space.
        raw_rmse = np.sqrt(np.mean((raw_pred - Y) ** 2, axis=0))
        assert raw_rmse[2] > raw_rmse[1] > raw_rmse[0]
        assert raw_rmse[2] > 1.0  # high-variance target has large RAW RMSE

    def test_target_names_and_metrics_survive_in_order(self, synthetic_multi_y, tmp_path):
        """5c(c): target names + per-target metrics survive save->load in order."""
        X, Y = synthetic_multi_y
        model, y_scaler = _train_joint_pls(X, Y)
        target_names = ["yield", "irsf", "carbonate"]
        metrics = multi_y_metrics(Y, model.predict(X), target_names=target_names)

        filepath = tmp_path / "mt_meta.dasp"
        save_model(
            model=model, preprocessor=None,
            metadata=_metadata(target_names, metrics["per_target"], "JOINT", X.shape[1]),
            filepath=filepath, y_scaler=y_scaler,
        )
        loaded = load_model(filepath)
        md = loaded["metadata"]
        assert md["target_names"] == target_names
        assert md["multitarget_mode"] == "JOINT"
        assert md["n_targets"] == 3
        assert md["prediction_columns"] == ["yield_pred", "irsf_pred", "carbonate_pred"]
        reloaded_targets = [m["target"] for m in md["per_target_metrics"]]
        assert reloaded_targets == target_names
        for orig, reloaded in zip(metrics["per_target"], md["per_target_metrics"]):
            for key in ("r2", "rmse", "q2", "rpd", "rer", "ccc", "bias"):
                assert reloaded[key] == pytest.approx(orig[key])

    def test_independent_model_persists_no_y_scaler(self, synthetic_multi_y, tmp_path):
        """INDEPENDENT multi-target models fit on RAW Y -> no Y-scaler persisted,
        and predict_with_model returns the estimator output unchanged."""
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import Ridge

        X, Y = synthetic_multi_y
        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
        model.fit(X, Y)  # RAW Y
        target_names = ["yield", "irsf", "carbonate"]
        metrics = multi_y_metrics(Y, model.predict(X), target_names=target_names)

        filepath = tmp_path / "mt_independent.dasp"
        save_model(
            model=model, preprocessor=None,
            metadata=_metadata(target_names, metrics["per_target"], "INDEPENDENT", X.shape[1]),
            filepath=filepath, y_scaler=None,
        )
        loaded = load_model(filepath)
        assert loaded["y_scaler"] is None
        assert loaded["metadata"]["has_y_scaler"] is False
        raw_pred = predict_with_model(loaded, X, validate_wavelengths=False)
        np.testing.assert_array_equal(raw_pred, model.predict(X))


class TestSingleYByteIdentityGuardrail:
    def test_single_target_model_has_no_y_scaler_and_predict_unchanged(self, tmp_path):
        """A single-target model carries no Y-scaler, so load_model returns
        y_scaler=None and predict_with_model skips the inverse-transform branch —
        the single-Y prediction path stays byte-identical."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 12))
        y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=30)
        model = PLSRegression(n_components=2)
        model.fit(X, y)

        metadata = {
            "model_name": "PLS", "task_type": "regression", "preprocessing": "raw",
            "wavelengths": [float(i) for i in range(12)], "n_vars": 12,
            "n_samples": 30, "performance": {"R2": 0.9},
        }
        filepath = tmp_path / "single_y.dasp"
        save_model(model=model, preprocessor=None, metadata=metadata, filepath=filepath)
        loaded = load_model(filepath)
        assert loaded["y_scaler"] is None
        assert loaded["metadata"]["has_y_scaler"] is False

        pred = predict_with_model(loaded, X, validate_wavelengths=False)
        np.testing.assert_array_equal(pred, model.predict(X))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
