"""Tests for the class_weight discriminator in the validation-rebuild path.

The bug class fixed by PR #38 (4 sister sites in the optimization-loop training
paths) had a downstream consumer that was silently broken: the validation-set
metric computation in ``compute_validation_metrics_for_top_models`` rebuilt
each top-N model via ``_rebuild_model_from_row`` and called ``model.fit(...)``
without re-applying the imbalance discriminator. Pre-fix:

* **XGBoost** retrained UNWEIGHTED for the validation set. XGBoost's
  class_weight handling is fit-time ``sample_weight`` (no constructor kwarg),
  so it isn't in the result-row ``Params`` string captured during search.
  ``_rebuild_model_from_row`` reconstructs from that string, losing the
  weighting silently.
* **PLS-DA** retrained UNWEIGHTED for the validation set. PLS-DA's
  ``class_weight`` lives in the LR sub-step's constructor inside
  ``_run_single_config`` (the ``lr_kwargs['class_weight'] = 'balanced'``
  block) but the LR step's params are NOT serialized into the result-row
  ``Params`` dict — only the PLS step's ``n_components`` ends up there.
  ``_rebuild_model_from_row`` reconstructs LR without ``class_weight``.

Reached when ``validation_count > 0`` in ``run_search`` (Grid) or
``run_unified_bayesian`` (Bayesian) on every classification job with
``imbalance_method='class_weight'`` (or ``'auto'`` resolving to it). The user
sees wrong ``val_*`` columns in the Results panel; the silent shape means
the user has no warning that the validation model was unweighted.

Test strategy
-------------
The fix is verified at two layers:

1. **Behavioral** — apply the discriminator to a rebuilt XGBoost model on
   imbalanced data, assert that ``fit_kwargs`` contains a non-trivial
   ``sample_weight`` array. Spy-style: we don't run a full search, we
   exercise the helper directly with a representative model.
2. **Structural** — apply the discriminator to a rebuilt PLS-DA Pipeline,
   assert ``model.get_params()['lr__class_weight'] == 'balanced'``. Per
   Codex's design pass, this is the cleanest invariant to pin because the
   ``get_params(deep=True)`` probe pattern is the actual fix mechanism.

Note on optional-dependency skip: ``importlib.util.find_spec`` boolean probe
in ``pytest.mark.skipif`` rather than ``pytest.importorskip`` inside
``pytest.param`` (the latter raises ``Skipped`` at collection time and kills
the entire parametrize set; see ``test_class_weight_sister_sites.py`` header
for the rationale).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from spectral_predict.search import (
    _apply_class_weight_discriminator_for_rebuilt_model,
    _rebuild_model_from_row,
)

_HAS_CATBOOST = importlib.util.find_spec("catboost") is not None
_HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


@pytest.fixture
def imbalanced_y():
    """Imbalanced binary classification labels (10:1 ratio)."""
    rng = np.random.default_rng(42)
    y = np.concatenate([np.zeros(90), np.ones(10)])
    rng.shuffle(y)
    return y.astype(int)


# =========================================================================
# Helper short-circuit invariants
# =========================================================================


@pytest.mark.unit
class TestDiscriminatorShortCircuits:
    """The helper must return ``{}`` for cases where no class_weight applies."""

    def test_regression_short_circuits(self, imbalanced_y):
        """Regression task → helper is a no-op regardless of imbalance_method."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "Linear", "regression", imbalanced_y, imbalance_method="class_weight"
        )
        assert result == {}

    def test_no_imbalance_method_short_circuits(self, imbalanced_y):
        """imbalance_method=None → helper is a no-op even for classification."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "LogReg", "classification", imbalanced_y, imbalance_method=None
        )
        assert result == {}
        assert model.get_params()["class_weight"] is None

    def test_resampling_method_short_circuits(self, imbalanced_y):
        """imbalance_method='smote' (a resampling method, not class_weight) → no-op."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "LogReg", "classification", imbalanced_y, imbalance_method="smote"
        )
        assert result == {}
        assert model.get_params()["class_weight"] is None

    def test_auto_normalizes_to_class_weight(self, imbalanced_y):
        """imbalance_method='auto' → treated as class_weight (defense-in-depth
        for direct GUI callers that pass raw GUI state without backend
        resolution)."""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "LogReg", "classification", imbalanced_y, imbalance_method="auto"
        )
        assert result == {}  # set_params path, no fit_kwargs
        assert model.get_params()["class_weight"] == "balanced"


# =========================================================================
# Per-model class_weight application
# =========================================================================


@pytest.mark.unit
class TestClassWeightConstructorPath:
    """Models whose class_weight is a constructor kwarg are mutated in place."""

    def test_sklearn_bare_model(self, imbalanced_y):
        """RandomForestClassifier exposes ``class_weight`` directly on the
        bare estimator. set_params should mutate it to 'balanced'."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=2)
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "RandomForest", "classification", imbalanced_y, imbalance_method="class_weight"
        )
        assert result == {}
        assert model.get_params()["class_weight"] == "balanced"

    def test_pls_da_pipeline(self, imbalanced_y):
        """PLS-DA Pipeline (PLS + scaler + LR) — class_weight lives on the
        'lr' sub-step. The discriminator's get_params probe finds
        ``lr__class_weight`` and sets it. This is the structural invariant
        the validation-rebuild fix pins."""
        # Build a PLS-DA result row and rebuild via the production path
        row = pd.Series({"Model": "PLS-DA", "Params": "{}", "LVs": 3})
        model = _rebuild_model_from_row(row, "classification", autoscale=False)

        # Confirm the rebuilt pipeline has the expected structure
        assert hasattr(model, "named_steps"), "PLS-DA rebuild should be a Pipeline"
        assert "lr" in model.named_steps, "PLS-DA pipeline should have an 'lr' step"
        assert model.get_params()["lr__class_weight"] is None, (
            "Pre-discriminator: LR class_weight should be None (the bug shape)"
        )

        # Apply discriminator
        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "PLS-DA", "classification", imbalanced_y, imbalance_method="class_weight"
        )
        assert result == {}
        assert model.get_params()["lr__class_weight"] == "balanced", (
            "Post-discriminator: LR class_weight must be 'balanced'"
        )

    def test_scale_sensitive_pipeline_routes_via_model_step(self, imbalanced_y):
        """Scale-sensitive models (SVC) are wrapped as ``('scaler', ...),
        ('model', SVC)`` by ``_rebuild_model_from_row``. class_weight should
        route via ``model__class_weight``. Uses 'SVC' which is in
        SCALE_SENSITIVE_MODELS at search.py:113."""
        row = pd.Series({"Model": "SVC", "Params": "{}", "LVs": None})
        # get_model('SVC', task_type='classification') returns SVC.
        # But the rebuild wrapper checks SCALE_SENSITIVE_MODELS which contains 'SVC'.
        # If get_model raises (e.g., 'SVC' not in registry), construct manually.
        try:
            model = _rebuild_model_from_row(row, "classification", autoscale=False)
        except ValueError:
            # Fall back: build the SCALE_SENSITIVE_MODELS shape manually so the
            # discriminator's `model__class_weight` route is still exercised.
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            model = Pipeline([("scaler", StandardScaler()), ("model", SVC(random_state=42))])

        # Confirm structure (the test invariant — Pipeline with 'model' step)
        assert hasattr(model, "named_steps"), "Test fixture should be a Pipeline"
        assert "model" in model.named_steps, "Pipeline should have a 'model' step"

        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "SVC", "classification", imbalanced_y, imbalance_method="class_weight"
        )
        assert result == {}
        assert model.get_params()["model__class_weight"] == "balanced"


# =========================================================================
# Sample-weight fallback path (XGBoost — the headline bug)
# =========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_XGBOOST, reason="xgboost not installed")
class TestSampleWeightFallback:
    """XGBoost has no class_weight constructor kwarg — its imbalance handling
    is fit-time sample_weight. Pre-fix, the validation-rebuild path called
    ``model.fit(X, y)`` and the model trained UNWEIGHTED. Post-fix, the
    discriminator returns ``{'sample_weight': ...}`` and the caller passes it.
    """

    def test_xgboost_returns_sample_weight_kwarg(self, imbalanced_y):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=2, use_label_encoder=False, eval_metric="logloss")

        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "XGBoost", "classification", imbalanced_y, imbalance_method="class_weight"
        )

        assert "sample_weight" in result, (
            "XGBoost should fall through to sample_weight kwarg (no class_weight ctor kwarg)"
        )
        sw = result["sample_weight"]
        assert isinstance(sw, np.ndarray)
        assert sw.shape == imbalanced_y.shape
        # Imbalanced labels → minority class gets larger weights
        expected = compute_sample_weight("balanced", imbalanced_y)
        np.testing.assert_array_equal(sw, expected)

    def test_xgboost_no_class_weight_attribute(self):
        """Document the bug-class invariant: XGBoost truly has no
        class_weight constructor kwarg. If sklearn-xgboost ever adds one,
        this test fails and the discriminator's priority-order probe will
        catch it via the constructor path instead of sample_weight fallback."""
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=2)
        assert "class_weight" not in model.get_params(deep=True), (
            "XGBoost was assumed to have no class_weight kwarg. If this assertion "
            "fires, the discriminator's priority-order probe will start using the "
            "constructor path; verify the change is intentional."
        )


# =========================================================================
# CatBoost — uses auto_class_weights (different mechanism)
# =========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_CATBOOST, reason="catboost not installed")
class TestCatBoostPath:
    """CatBoost uses ``auto_class_weights='Balanced'`` (different param name from
    sklearn's ``class_weight='balanced'``). The discriminator's CatBoost branch
    fires after the constructor-probe loop misses ``class_weight``."""

    def test_catboost_sets_auto_class_weights(self, imbalanced_y):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=2, verbose=0)

        result = _apply_class_weight_discriminator_for_rebuilt_model(
            model, "CatBoost", "classification", imbalanced_y, imbalance_method="class_weight"
        )

        assert result == {}, "CatBoost uses constructor kwarg, no fit_kwargs"
        assert model.get_params()["auto_class_weights"] == "Balanced"


# =========================================================================
# Caller-threading: confirm imbalance_method parameter is in the signature
# =========================================================================


@pytest.mark.unit
class TestCallerThreading:
    """Structural test that `compute_validation_metrics_for_top_models` accepts
    imbalance_method, and that the helper is reachable from the production
    function. Catches a refactor that drops the parameter or stops calling
    the helper."""

    def test_compute_validation_metrics_accepts_imbalance_method(self):
        import inspect
        from spectral_predict.search import compute_validation_metrics_for_top_models
        sig = inspect.signature(compute_validation_metrics_for_top_models)
        assert "imbalance_method" in sig.parameters, (
            "compute_validation_metrics_for_top_models must accept imbalance_method "
            "so that XGBoost / PLS-DA validation rebuilds receive the discriminator."
        )
        # Default must be None for backward-compat with non-imbalance callers.
        assert sig.parameters["imbalance_method"].default is None

    def test_helper_referenced_in_validation_function_source(self):
        """Catches a refactor that drops the helper call from the validation
        function body. The helper exists; this test ensures it's wired in."""
        import inspect
        from spectral_predict.search import compute_validation_metrics_for_top_models
        source = inspect.getsource(compute_validation_metrics_for_top_models)
        assert "_apply_class_weight_discriminator_for_rebuilt_model" in source, (
            "The validation-rebuild discriminator helper must be called from "
            "compute_validation_metrics_for_top_models — without it, the rebuild "
            "path silently produces unweighted XGBoost and PLS-DA validation models."
        )

    def test_fit_site_splats_fit_kwargs(self):
        """Pin the fit-site that the helper's return value is splatted into
        ``model.fit``. The helper can return ``{'sample_weight': ...}`` for
        XGBoost or ``{'model__sample_weight': ...}`` for Pipeline-wrapped
        estimators — without the ``**fit_kwargs`` splat at the call site,
        XGBoost validation rebuild silently regresses to UNWEIGHTED training
        (the original bug shape). Catches a refactor that drops the splat
        while leaving the helper call in place. Per pr-test-analyzer
        recommendation on PR #41 (the same silent-failure shape as the
        original bug)."""
        import inspect
        from spectral_predict.search import compute_validation_metrics_for_top_models
        source = inspect.getsource(compute_validation_metrics_for_top_models)
        assert "**fit_kwargs" in source, (
            "Fit site must splat **fit_kwargs from the discriminator's return "
            "value. Without it, XGBoost (and any other model whose class_weight "
            "is fit-time sample_weight) silently trains UNWEIGHTED in validation "
            "rebuild — the exact failure shape this PR fixes."
        )
        # Stronger pin: verify the literal fit() call carries the splat.
        # The fit site immediately follows the helper call.
        assert "model.fit(X_train_final, y_train, **fit_kwargs)" in source, (
            "Fit site must invoke `model.fit(X_train_final, y_train, **fit_kwargs)` "
            "exactly. A refactor that splits the splat into a separate variable "
            "or moves the kwargs application elsewhere should re-verify the "
            "splat pattern is preserved."
        )

    @pytest.mark.parametrize(
        "func_name",
        ["run_search"],
    )
    def test_search_callers_pass_imbalance_method(self, func_name):
        """Per-caller structural pin (Codex finding on this PR). The previous
        signature-only test would pass even if a refactor dropped
        ``imbalance_method=imbalance_method`` from one of the callers, silently
        regressing the validation rebuild for that path. Pin each caller's
        source individually."""
        import inspect
        import spectral_predict.search as search_mod
        func = getattr(search_mod, func_name)
        source = inspect.getsource(func)
        # The caller must invoke compute_validation_metrics_for_top_models AND
        # pass imbalance_method=imbalance_method (the local variable from the
        # function's own signature).
        assert "compute_validation_metrics_for_top_models" in source, (
            f"{func_name} must call compute_validation_metrics_for_top_models "
            f"to populate val_* columns when validation_count > 0."
        )
        assert "imbalance_method=imbalance_method" in source, (
            f"{func_name} must pass imbalance_method=imbalance_method to "
            f"compute_validation_metrics_for_top_models. Without it, the validation "
            f"rebuild silently produces unweighted XGBoost / PLS-DA models."
        )

    def test_gui_callers_pass_imbalance_method(self):
        """Per-caller structural pin for the 2 GUI direct callers at gui:27914
        and gui:28069. These were caught by Codex's design pass; the
        continuation prompt only flagged the 2 backend callers in search.py.
        Source-level grep because the GUI module is too large to import in
        unit tests."""
        import os
        gui_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "spectral_predict_gui_optimized.py",
        )
        with open(gui_path, "r", encoding="utf-8") as f:
            gui_source = f.read()

        # Find each compute_validation_metrics_for_top_models invocation in the
        # GUI module. Each must immediately precede an
        # "imbalance_method=imbalance_method" line within the call args.
        invocations = gui_source.count("compute_validation_metrics_for_top_models(")
        with_imbalance = gui_source.count("imbalance_method=imbalance_method,")
        # The GUI also passes imbalance_method=imbalance_method to other backend
        # calls (unified_bayesian, nsga2_search), so with_imbalance >= invocations
        # is the floor; the strict check is per-invocation.
        assert invocations >= 2, (
            f"Expected at least 2 GUI calls to compute_validation_metrics_for_top_models "
            f"(Bayesian validation panel + NSGA-II validation panel); found {invocations}."
        )
        # Strict check: each compute_validation_metrics_for_top_models call must
        # be in a context that includes imbalance_method=imbalance_method within
        # the argument list. Slice each call's arg block and verify.
        cursor = 0
        checked = 0
        while True:
            idx = gui_source.find("compute_validation_metrics_for_top_models(", cursor)
            if idx == -1:
                break
            # Find the matching close paren for the call (naive: scan forward
            # tracking depth). For our purposes a 2KB window suffices since the
            # call args are short.
            depth = 0
            i = idx
            end = idx
            while i < len(gui_source):
                c = gui_source[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                i += 1
            call_block = gui_source[idx:end + 1]
            cursor = end + 1
            # The multi-class SIMCA validation call (T-31 Task 11) is exempt:
            # class-modeling is per-class unsupervised (no class_weight /
            # sample_weight), so imbalance_method has no meaning and is not in
            # scope on that path.
            if 'task_type="multiclass_simca"' in call_block:
                continue
            assert "imbalance_method=imbalance_method" in call_block, (
                f"GUI call to compute_validation_metrics_for_top_models at offset "
                f"{idx} does not pass imbalance_method=imbalance_method. The "
                f"validation rebuild for that path will silently produce unweighted "
                f"XGBoost / PLS-DA validation models."
            )
            checked += 1
        assert checked >= 2, f"Expected to check at least 2 call sites, checked {checked}."
