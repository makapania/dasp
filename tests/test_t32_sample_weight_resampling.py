"""T-32 regression: sample_weight + imblearn resampler must not crash.

Pre-fix bug: in _run_single_fold's classification-sample-weight path
(search.py:~4109), after an imblearn resampler ran, the final model.fit()
call passed the original (pre-resampling) y_train alongside the resampled
X_train_transformed and the resampler-aware sample_weight_train. sklearn's
check_X_y caught the length mismatch and raised ValueError, so any user who
combined SMOTE-family resampling with a sample_weight-supporting classifier
(Ridge, LogisticRegression, ...) got a hard crash on the very first
training fold.

Fix: thread y_train_for_model through the whole post-loop fit() path so X,
y, and sample_weight all stay length-consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("imblearn")

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import RidgeClassifier

from spectral_predict.search import _run_single_fold


def _imbalanced_binary(seed=0):
    """Imbalanced binary dataset where SMOTE will actually upsample.

    30 majority + 12 minority. After the test split (4 held out from the
    minority side), the train fold has 30 + 8 = 38 samples with 8 minority
    — enough for SMOTE's k_neighbors=3 (needs k+1=4 minority) without
    fallback.
    """
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(0, 1, (30, 8))
    X_min = rng.normal(2, 1, (12, 8))
    X = np.vstack([X_maj, X_min]).astype(np.float64)
    y = np.array([0] * 30 + [1] * 12, dtype=np.int64)
    return X, y


def test_t32_smote_plus_ridge_sample_weight_does_not_crash():
    """Pre-fix: this raised ValueError due to X/y/sample_weight length mismatch
    after SMOTE resampled the training fold but y_train stayed at the
    pre-resample length. Now: fit succeeds and we get fold metrics back."""
    X, y = _imbalanced_binary()

    # SMOTE resampler followed by a sample_weight-supporting classifier.
    # RidgeClassifier supports sample_weight at fit time but not class_weight,
    # which is what triggers use_sample_weight_for_classification=True.
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=0, k_neighbors=3)),
        ("model", RidgeClassifier()),
    ])

    # Hold out the last 4 minority samples for test so the train fold has
    # 30 majority + 8 minority — SMOTE k_neighbors=3 needs ≥4 minority.
    train_idx = np.arange(len(y) - 4)
    test_idx = np.arange(len(y) - 4, len(y))

    metrics = _run_single_fold(
        pipe=pipe,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        task_type="classification",
        is_binary_classification=True,
        use_sample_weight_for_classification=True,
    )

    assert isinstance(metrics, dict)
    # The fold ran successfully — predictions are present and length-consistent
    # with the held-out test set.
    assert "y_pred" in metrics
    assert len(metrics["y_pred"]) == len(test_idx)


def test_t32_resampler_extends_y_for_downstream_fit():
    """End-to-end pin on the underlying invariant: after a resampler step,
    the y the model is trained on must reflect the resampled length, not the
    original. Verify by capturing what the model receives via a fit() spy."""
    X, y = _imbalanced_binary()
    captured: dict[str, object] = {}

    class _SpyRidge(RidgeClassifier):
        def fit(self, X_inner, y_inner, sample_weight=None):
            captured["X_len"] = len(X_inner)
            captured["y_len"] = len(y_inner)
            captured["sw_len"] = (
                len(sample_weight) if sample_weight is not None else None
            )
            return super().fit(X_inner, y_inner, sample_weight=sample_weight)

    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=0, k_neighbors=3)),
        ("model", _SpyRidge()),
    ])

    # Same train/test split rationale as the first test.
    train_idx = np.arange(len(y) - 4)
    test_idx = np.arange(len(y) - 4, len(y))

    _run_single_fold(
        pipe=pipe,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        task_type="classification",
        is_binary_classification=True,
        use_sample_weight_for_classification=True,
    )

    # All three lengths must agree post-resample. SMOTE upsamples the
    # minority to match the majority count (20+20 = 40 from the original
    # 18+4 train fold) — but the exact count depends on SMOTE's internals,
    # so the regression test pins length-equality rather than the exact value.
    assert captured["X_len"] == captured["y_len"], (
        f"X length {captured['X_len']} != y length {captured['y_len']} — "
        "the T-32 mismatch bug has regressed"
    )
    assert captured["sw_len"] == captured["y_len"], (
        f"sample_weight length {captured['sw_len']} != y length "
        f"{captured['y_len']} — the T-32 sample_weight mismatch bug has regressed"
    )
    # And the resample actually happened — y is longer than the pre-resample
    # train fold (which was 38 samples = 30 majority + 8 minority; SMOTE
    # upsamples minority to majority count → 30 + 30 = 60).
    assert captured["y_len"] > len(train_idx), (
        f"SMOTE didn't actually resample: y_len={captured['y_len']} <= "
        f"train_idx len {len(train_idx)} — test premise broken"
    )
