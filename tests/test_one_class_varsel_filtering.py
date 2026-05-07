"""Regression tests for one-class variable-selection method filtering.

Closes Codex finding on PR #57: the GUI disables UVE-family + iPLS-family
varsel + apply_uve_prefilter for one-class (per CLAUDE.md:66 — UVE-on-y_oc
is a discrimination method, not one-class modeling, per Pomerantsev et al.
2025 LOVE), but `run_one_class_search` accepted them anyway because:

1. `implemented_oc_varsel` whitelist included uve, uve_spa, uve_cars,
   uve_cars_tree, uve_cars_spa.
2. `apply_uve_prefilter=True` was honored without question.

Both backend gaps are now closed; this file pins them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from spectral_predict.search import run_one_class_search


@pytest.fixture
def synthetic_one_class_data():
    rng = np.random.RandomState(42)
    n_features = 30
    X_clean = rng.randn(30, n_features) * 0.3
    X_contam = rng.randn(8, n_features) + 3.0
    X = np.vstack([X_clean, X_contam])
    y = np.array(["clean"] * 30 + ["contaminated"] * 8)
    wavelengths = [f"{400 + i * 10:.1f}" for i in range(n_features)]
    return pd.DataFrame(X, columns=wavelengths), pd.Series(y)


def _run_oc_search(X, y, **kwargs):
    return run_one_class_search(
        X=X,
        y=y,
        inlier_class_label="clean",
        folds=3,
        preprocessing_methods=["raw"],
        window_sizes=[17],
        enabled_models=["IsolationForest"],
        variable_counts=[10],
        **kwargs,
    )


class TestOneClassVarselFiltering:
    def test_uve_family_methods_filtered_out(self, synthetic_one_class_data, caplog):
        """All UVE-family methods must be rejected for one-class; spa and
        importance must still run."""
        X, y = synthetic_one_class_data
        with caplog.at_level(logging.WARNING):
            results = _run_oc_search(
                X,
                y,
                variable_selection_methods=[
                    "uve",
                    "uve_spa",
                    "uve_cars",
                    "uve_cars_tree",
                    "uve_cars_spa",
                    "spa",
                    "importance",
                ],
            )

        subset_tags = set(results["SubsetTag"].astype(str).unique())
        # No row should have a SubsetTag that mentions a UVE-family method.
        for forbidden in ("uve", "uve_spa", "uve_cars", "uve_cars_tree", "uve_cars_spa"):
            offending = [t for t in subset_tags if t.startswith(f"{forbidden}_top")]
            assert not offending, (
                f"UVE-family method '{forbidden}' must be filtered for one-class "
                f"(per CLAUDE.md:66), but found result rows with SubsetTag "
                f"prefix '{forbidden}_top': {offending}"
            )
        # The valid methods must still produce results.
        valid_tags = [t for t in subset_tags if t.startswith(("spa_top", "importance_top"))]
        assert valid_tags, (
            "Valid one-class methods (spa, importance) must still produce "
            f"varsel result rows; got SubsetTags: {sorted(subset_tags)}"
        )
        # Warning must be logged listing the unsupported methods.
        warning_text = " ".join(r.getMessage() for r in caplog.records)
        assert "uve" in warning_text.lower(), (
            "A warning must be logged enumerating filtered UVE-family methods; "
            f"caplog records: {[r.getMessage() for r in caplog.records]}"
        )

    def test_ipls_family_methods_filtered_out(self, synthetic_one_class_data, caplog):
        """iPLS-family methods (require PLS internals not available for
        one-class) must be rejected; spa must still run."""
        X, y = synthetic_one_class_data
        with caplog.at_level(logging.WARNING):
            results = _run_oc_search(
                X,
                y,
                variable_selection_methods=[
                    "ipls",
                    "ipls_forward",
                    "ipls_backward",
                    "mc_sipls",
                    "mwpls",
                    "spa",
                ],
            )

        subset_tags = set(results["SubsetTag"].astype(str).unique())
        for forbidden in ("ipls", "ipls_forward", "ipls_backward", "mc_sipls", "mwpls"):
            offending = [t for t in subset_tags if t.startswith(f"{forbidden}_top")]
            assert not offending, (
                f"iPLS-family method '{forbidden}' must be filtered for one-class, "
                f"but found result rows with SubsetTag prefix '{forbidden}_top': "
                f"{offending}"
            )
        valid_tags = [t for t in subset_tags if t.startswith("spa_top")]
        assert valid_tags, (
            "Valid one-class method 'spa' must still produce varsel result rows; "
            f"got SubsetTags: {sorted(subset_tags)}"
        )

    def test_apply_uve_prefilter_forced_false(self, synthetic_one_class_data, caplog):
        """`apply_uve_prefilter=True` must be coerced to False for one-class
        with a warning. UVE-on-y_oc is a discrimination method, not a one-class
        method (CLAUDE.md:66)."""
        X, y = synthetic_one_class_data
        with caplog.at_level(logging.WARNING):
            _run_oc_search(
                X,
                y,
                apply_uve_prefilter=True,
                variable_selection_methods=["spa"],
            )

        log_text = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "uve prefilter" in log_text and "one-class" in log_text, (
            "Passing apply_uve_prefilter=True for one-class must log a warning "
            "explaining the coercion; caplog: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        # GLM 5.1 cycle 4 MEDIUM: pin "fires AT MOST ONCE" so a future
        # refactor moving the coercion inside a loop body can't silently
        # re-spam. Today the local-mutation prevents loop-spam, but that's
        # an implementation detail; the contract is one warning per call.
        coercion_msgs = [
            r for r in caplog.records
            if "uve prefilter" in r.getMessage().lower()
            and "one-class" in r.getMessage().lower()
        ]
        assert len(coercion_msgs) == 1, (
            "Coercion warning must fire exactly once per call; got "
            f"{len(coercion_msgs)}: {[r.getMessage() for r in coercion_msgs]}"
        )
        # And the prefilter must not have actually run — no eliminate-variables
        # info-log should appear.
        info_text = " ".join(
            r.getMessage() for r in caplog.records if r.levelno <= logging.INFO
        )
        assert "UVE prefilter:" not in info_text, (
            "UVE prefilter must NOT execute for one-class even when caller "
            "passes apply_uve_prefilter=True"
        )


class TestCreateUnifiedObjectiveInnerGuard:
    """Direct exercise of the defense-in-depth guard inside
    ``create_unified_objective``. The outer ``run_unified_bayesian`` guard
    short-circuits ``enable_uve`` before this inner guard is ever reached
    via the normal path, so without a direct test the inner branch is
    "dead from a coverage perspective" and a future refactor could remove
    it with no test failure (pr-test-analyzer cycle 4 IMPORTANT).
    """

    def test_inner_guard_coerces_when_called_directly(self, caplog):
        """Calling ``create_unified_objective`` directly with
        ``task_type='one_class', enable_uve=True`` must coerce to False
        and exclude ``'uve'`` from the closure's available_methods."""
        from spectral_predict.unified_bayesian import create_unified_objective

        rng = np.random.RandomState(7)
        n_samples, n_features = 30, 20
        X = rng.randn(n_samples, n_features)
        y_str = np.array(["clean"] * 22 + ["contaminated"] * 8)

        with caplog.at_level(logging.WARNING):
            objective = create_unified_objective(
                X_raw=X,
                y=np.where(y_str == "clean", 1, -1).astype(float),
                wavelengths=np.linspace(1000, 2500, n_features),
                model_name="PCA-SIMCA",
                task_type="one_class",
                cv_folds=3,
                random_state=42,
                enable_uve=True,
                inlier_class_label="clean",
                y_original=y_str,
            )

        assert callable(objective), "Should return a callable objective"

        # The inner guard must have fired exactly once (its warning is
        # the only one emitted at create-time; the outer guard is not
        # invoked when calling create_unified_objective directly).
        coercion_msgs = [
            r for r in caplog.records
            if "enable_uve" in r.getMessage()
            and "one-class" in r.getMessage()
        ]
        assert len(coercion_msgs) == 1, (
            "Inner guard must fire exactly once when create_unified_objective "
            f"is called directly with enable_uve=True; got {len(coercion_msgs)}: "
            f"{[r.getMessage() for r in coercion_msgs]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
