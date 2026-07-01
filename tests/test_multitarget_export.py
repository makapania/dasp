"""T-17 F7: exported multi-Y script reproduces in-app multi-target predictions.

The multi-target code-export path (``code_generator`` +
``templates/header.py`` + ``templates/validation.py``) emits a standalone
script that transcribes :func:`spectral_predict.multi_y.multi_y_cv_pool` — fold
Y-scaling + inverse-transform for JOINT models, raw Y for INDEPENDENT — plus
per-target RAW-unit metrics. This suite pins that the exported script's pooled
CV predictions, per-target metrics, and joint Q² match the in-app
:func:`spectral_predict.multitarget_search.run_multitarget_search`.

It also pins the byte-identity guardrail: a single-target config never triggers
the multi-target generator (``is_multitarget`` False), so the legacy single-Y
export stays untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.multitarget_search import (
    INDEPENDENT_PRECISE_NOTE,
    run_multitarget_search,
)


def _correlated_multi_y(seed: int = 0, n_samples: int = 40, n_features: int = 15):
    """Two correlated numeric targets over a shared latent structure."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    latent = rng.standard_normal((n_samples, 2))
    y0 = 2.0 * latent[:, 0] + 0.1 * rng.standard_normal(n_samples)
    y1 = 3.0 * latent[:, 0] + latent[:, 1] + 0.1 * rng.standard_normal(n_samples)
    Y = np.column_stack([y0, y1])
    return X, Y


def _exec_generated_multitarget(model_name, params, X, Y, target_names):
    """Generate a multi-target script with embedded data, exec it, return ns."""
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "regression",
        "params": params,
        "cv_folds": 5,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "target_names": target_names,
        "wavelengths": list(range(X.shape[1])),
    }
    opts = ExportOptions(
        format="script",
        include_data=True,
        data_X=X,
        data_y=Y,
        wavelengths=np.arange(X.shape[1]),
        include_visualization=False,
        include_prediction_template=False,
        include_cross_validation=True,
    )
    gen = CodeGenerator(config, opts)
    assert gen.is_multitarget, "multi-target config must activate the multi-Y generator"
    script = gen.generate_script()
    ns: dict = {}
    exec(compile(script, "<multitarget_export>", "exec"), ns)
    return ns, script


@pytest.mark.parametrize(
    "model_name,params,expected_mode",
    [
        ("PLS", {"n_components": 3}, "JOINT"),          # JOINT flagship
        ("Ridge", {"alpha": 1.0}, "INDEPENDENT"),        # INDEPENDENT native
        ("SVR", {}, "INDEPENDENT"),                      # INDEPENDENT MOR-wrapped
    ],
)
def test_exported_script_reproduces_inapp_pooled_predictions(model_name, params, expected_mode):
    X, Y = _correlated_multi_y()
    target_names = ["A", "B"]

    ns, script = _exec_generated_multitarget(model_name, params, X, Y, target_names)

    out = run_multitarget_search(
        X, Y, [{"model_name": model_name, "params": params}],
        cv="kfold", n_folds=5, n_repeats=5, target_names=target_names,
        optimization_method="grid",
    )
    res = out.results[0]

    assert res.mode == expected_mode
    # Header honestly labels the coupling mode.
    assert f"MultiTarget Mode: {expected_mode}" in script

    # Pooled RAW-unit CV predictions reproduce the in-app run bit-for-bit.
    np.testing.assert_allclose(
        ns["Y_pred_cv"], res.y_pred_pooled, rtol=0, atol=1e-9,
        err_msg="exported multi-Y pooled predictions diverge from in-app run",
    )
    np.testing.assert_allclose(ns["Y_true_cv"], res.y_true_pooled, rtol=0, atol=1e-9)

    # Joint Q² and per-target metrics reproduce the in-app metrics.
    assert ns["Y_pred_cv"].shape == (X.shape[0], 2)
    np.testing.assert_allclose(ns["joint_q2"], res.joint_q2, rtol=0, atol=1e-9)
    export_per = {m["target"]: m for m in ns["per_target_metrics"]}
    for tgt in target_names:
        for key in ("r2", "rmse", "rpd", "rer", "ccc", "bias"):
            in_app = next(d for d in res.metrics["per_target"] if d["target"] == tgt)[key]
            np.testing.assert_allclose(
                export_per[tgt][key], in_app, rtol=0, atol=1e-9,
                err_msg=f"{model_name} {tgt} {key} diverges",
            )


def test_joint_scale_y_true_independent_false_in_emitted_source():
    """JOINT emits SCALE_Y=True (fold Y-scaling); INDEPENDENT emits SCALE_Y=False."""
    X, Y = _correlated_multi_y()
    _, joint_script = _exec_generated_multitarget("PLS", {"n_components": 3}, X, Y, ["A", "B"])
    _, indep_script = _exec_generated_multitarget("Ridge", {"alpha": 1.0}, X, Y, ["A", "B"])
    assert "SCALE_Y = True" in joint_script
    assert "SCALE_Y = False" in indep_script


def test_exported_final_model_predicts_raw_units_for_joint():
    """The JOINT final model fits on full-data-scaled Y and predict_raw() maps
    back to RAW units (inverse-transform via the full-training FoldYScaler)."""
    X, Y = _correlated_multi_y()
    ns, _ = _exec_generated_multitarget("PLS", {"n_components": 3}, X, Y, ["A", "B"])
    # Calibration predictions live in RAW units (same scale as Y), so their
    # column means track Y's column means far better than scaled (~0-mean) units.
    y_pred_cal = ns["Y_pred_cal"]
    assert y_pred_cal.shape == Y.shape
    assert np.all(np.abs(y_pred_cal.mean(axis=0) - Y.mean(axis=0)) < 1.0)
    assert ns["final_y_scaler"] is not None  # JOINT keeps a full-data Y-scaler


def test_multitarget_notebook_reproduces_inapp_predictions():
    """The notebook path (title markdown + one code cell) reproduces the in-app
    pooled predictions too — a multi-target config must not silently emit a
    single-Y notebook."""
    import contextlib
    import io as _io

    X, Y = _correlated_multi_y()
    config = {
        "model_name": "PLS", "preprocessing": "raw", "task_type": "regression",
        "params": {"n_components": 3}, "cv_folds": 5, "cv_strategy": "kfold",
        "cv_n_repeats": 5, "target_names": ["A", "B"], "wavelengths": list(range(15)),
    }
    opts = ExportOptions(
        format="notebook", include_data=True, data_X=X, data_y=Y,
        wavelengths=np.arange(15), include_visualization=False,
        include_prediction_template=False,
    )
    nb = CodeGenerator(config, opts).generate_notebook()
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code"]
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ns: dict = {}
    with contextlib.redirect_stdout(_io.StringIO()):
        exec(compile(code, "<multitarget_nb>", "exec"), ns)
    out = run_multitarget_search(
        X, Y, [{"model_name": "PLS", "params": {"n_components": 3}}],
        cv="kfold", n_folds=5, n_repeats=5, target_names=["A", "B"],
        optimization_method="grid",
    )
    np.testing.assert_allclose(
        ns["Y_pred_cv"], out.results[0].y_pred_pooled, rtol=0, atol=1e-9)


def test_single_target_config_does_not_activate_multitarget():
    """Byte-identity guardrail: a single target keeps the legacy single-Y path."""
    X, Y = _correlated_multi_y()
    config = {
        "model_name": "PLS",
        "preprocessing": "raw",
        "task_type": "regression",
        "target_name": "protein",
        "params": {"n_components": 3},
        "cv_folds": 5,
    }
    # One-element target_names must NOT flip the multi-target switch.
    config_one = dict(config, target_names=["protein"])
    gen = CodeGenerator(config_one, ExportOptions(include_data=False))
    assert gen.is_multitarget is False
    script = gen.generate_script()
    # Legacy single-Y markers present; multi-target markers absent.
    assert "cross_val_predict" in script
    assert "MultiTarget Mode" not in script
    assert "Y_pred_cv" not in script


def _generate_multitarget_script(model_name, params, target_names, n_features=15):
    """Generate a multi-target script WITHOUT embedding/exec (fast, generation
    only) — used to pin header text and to confirm a builder exists."""
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "regression",
        "params": params,
        "cv_folds": 5,
        "target_names": target_names,
        "wavelengths": list(range(n_features)),
    }
    gen = CodeGenerator(config, ExportOptions(include_data=False))
    assert gen.is_multitarget
    return gen.generate_script()


def test_independent_header_carries_exact_precise_note():
    """Honest-labeling guardrail: an INDEPENDENT export header must contain the
    exact INDEPENDENT_PRECISE_NOTE verbatim (pins the string against drift),
    while a JOINT header must NOT (it is genuine coupling)."""
    indep_script = _generate_multitarget_script("Ridge", {"alpha": 1.0}, ["A", "B"])
    joint_script = _generate_multitarget_script("PLS", {"n_components": 3}, ["A", "B"])

    assert INDEPENDENT_PRECISE_NOTE in indep_script, (
        "INDEPENDENT export header must carry the exact precise note verbatim"
    )
    assert "MultiTarget Mode: INDEPENDENT" in indep_script
    assert INDEPENDENT_PRECISE_NOTE not in joint_script
    assert "MultiTarget Mode: JOINT" in joint_script


def test_neuralboosted_multitarget_export_has_builder():
    """NeuralBoosted is a runtime-supported INDEPENDENT multi-target model
    (multitarget_search._build_independent_base), so its export must NOT raise
    NotImplementedError; the script wraps NeuralBoostedRegressor in a
    MultiOutputRegressor."""
    script = _generate_multitarget_script("NeuralBoosted", {}, ["A", "B"])
    assert "MultiOutputRegressor" in script
    assert "NeuralBoostedRegressor" in script
    assert "from spectral_predict.neural_boosted import NeuralBoostedRegressor" in script
    # INDEPENDENT model -> honest note present.
    assert INDEPENDENT_PRECISE_NOTE in script


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
