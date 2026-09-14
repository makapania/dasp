"""T-51 PR B (plan T9): Tab 7 refit reproduces a model searched with extra-axis bundles.

Drives ``_run_refined_model_thread`` synchronously on the session Tk app, one
representative per family, from a results row carrying non-default bundle values
(built and checked in ``tests/test_t51_supervised_bundles.py``). The refit must carry
every bundle value and constant and predict like the search-time model.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from tests.test_t51_supervised_bundles import (
    ROUND_TRIPS,
    assert_estimator_carries,
    predictions,
    round_trip_case,
)

pytestmark = pytest.mark.gui


def _refit(app, case: dict):
    X_df = case["X"].copy()
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    y = pd.Series(case["y"], index=X_df.index)

    app.X_original = X_df
    app.X = X_df
    app.y = y
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = dict(case["row"])
    app._original_wavelength_order = [float(c) for c in X_df.columns]
    app.refine_task_type.set(case["task"])
    app.refine_model_type.set(case["family"])
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


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_tab7_refit_keeps_bundle_params(gui_app, name):
    case = round_trip_case(name)

    model = _refit(gui_app, case)

    assert_estimator_carries(model, case)
    np.testing.assert_allclose(
        predictions(model, case["X_test"], case["task"]),
        predictions(case["reference"], case["X_test"], case["task"]),
        rtol=1e-6,
        atol=1e-8,
    )
