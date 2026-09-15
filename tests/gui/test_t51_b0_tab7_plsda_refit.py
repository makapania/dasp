"""T-51 PR B0: Tab 7 (Model Development) refit keeps the PLS-DA logistic head.

Drives ``_run_refined_model_thread`` synchronously on a real (withdrawn) Tk app for
{canonical ``lr__*`` row, legacy ``lr_*`` row} x {default, non-default transformer
settings}, with a non-default head (C, solver, max_iter). Before B0 a legacy row's
``lr_C`` / ``lr_solver`` / ``lr_max_iter`` were pushed into ``PLSTransformer.set_params``
(which stores any key as a junk attribute) and never reached the LogisticRegression, so
the refit used C=1.0. Canonical rows were already right in Tab 7, only because the
pipeline-level ``_apply_pipeline_params_to_pipe`` re-applies ``lr__*`` after the build.

Ensemble reconstruction (``_reconstruct_models_from_results``) hard-coded the head and
dropped every transformer param except ``n_components`` for both spellings.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from tests.test_t51_b0_plsda_head_params import (
    HEAD,
    MATRIX,
    TRANSFORMER_SETTINGS,
    X_TEST,
    X_TRAIN,
    Y_TRAIN,
    _assert_pipeline_matches,
    _reference_proba,
    _row_params,
    canonical_grid_rows,  # noqa: F401  (pytest fixture)
)

pytestmark = pytest.mark.gui


def _refit(app, params: dict, n_components: int):
    wavelengths = [float(c) for c in X_TRAIN.columns]
    X_df = X_TRAIN.copy()
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    y = pd.Series(Y_TRAIN.values, index=X_df.index)

    app.X_original = X_df
    app.X = X_df
    app.y = y
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = {
        "Model": "PLS-DA",
        "Task": "classification",
        "Params": str(params),
        "LVs": n_components,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    app._original_wavelength_order = wavelengths
    app.refine_task_type.set("classification")
    app.refine_model_type.set("PLS-DA")
    app.refine_preprocess.set("raw")
    app.refine_folds.set(3)
    app.refine_cv_strategy.set("kfold")
    app.model_loaded_from_results = True
    app.refine_hyperparams_modified = False
    app.refined_model = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._run_refined_model_thread()
    app.root.update()
    assert app.refined_model is not None, buf.getvalue()[-4000:]
    return app.refined_model


@pytest.mark.parametrize("style,setting", MATRIX)
def test_tab7_refit_plsda_head_matrix(gui_app, style, setting, canonical_grid_rows):  # noqa: F811
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)

    model = _refit(gui_app, params, transformer["n_components"])

    _assert_pipeline_matches(model, transformer)
    assert np.allclose(model.predict_proba(X_TEST), _reference_proba(transformer))
    assert model.get_params()["lr__C"] == HEAD["C"]


@pytest.mark.parametrize("style,setting", MATRIX)
def test_ensemble_reconstruction_plsda_head_matrix(
    gui_app, style, setting, canonical_grid_rows  # noqa: F811
):
    """``_reconstruct_models_from_results`` (ensemble training) is a fifth consumer."""
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)
    top_models_df = pd.DataFrame(
        [
            {
                "Model": "PLS-DA",
                "Params": str(params),
                "Preprocess": "raw",
                "Deriv": 0,
                "Window": 17,
                "Poly": 2,
            }
        ]
    )

    reconstructed = gui_app._reconstruct_models_from_results(
        top_models_df, X_TRAIN, Y_TRAIN.values, "classification"
    )

    assert len(reconstructed) == 1
    fitted, _, _ = reconstructed[0]
    _assert_pipeline_matches(getattr(fitted, "pipeline", fitted), transformer)
    assert np.allclose(fitted.predict_proba(X_TEST), _reference_proba(transformer))
