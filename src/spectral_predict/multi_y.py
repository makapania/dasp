"""Model-agnostic multi-target (multi-Y) foundation for T-17.

This module is the shared, estimator-agnostic core consumed by the
multi-target orchestrator, all models, v1 variable selection, VIP, and
(later, unchanged) the deferred UVE/CARS multi-Y work.

Design invariants (from the T-17 plan):

* **Fold Y-scaling is JOINT fitting only** — :class:`FoldYScaler` standardizes
  each target column using train-fold statistics and is used ONLY to fit
  JOINT models (PLS covariance, RF impurity, MLP summed-MSE, CatBoost
  MultiRMSE, XGBoost multi_output_tree, MultiTask linear). INDEPENDENT
  (per-target-separable) estimators fit on RAW Y.
* **Metrics are always on RAW units.** JOINT predictions are inverse-
  transformed to raw *inside* :func:`multi_y_cv_pool` before pooling, so
  :func:`multi_y_metrics` only ever sees raw-unit arrays. Per-target
  Q2 is scale-invariant, so joint Q2 == mean per-target Q2, and at
  ``n_targets == 1`` per-target Q2 equals the legacy
  ``r2_score(all_y_test, all_y_pred)`` (the consolidation pin).
* **Task-agnostic seam.** Orchestration, :func:`multi_y_cv_pool`, and the
  correlation guardrail are task-agnostic. Only the metric layer
  (:func:`multi_y_metrics`) and Y-scaling are regression-specific; they are
  structured as a swappable layer so multi-label classification can slot in
  later as a task mode of THIS module.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import clone

from .cv_utils import build_cv_splitter
from .scoring import lins_ccc

__all__ = [
    "FoldYScaler",
    "multi_y_cv_pool",
    "multi_y_metrics",
    "reduce_multi_y_score",
    "extract_pls_multi_y",
    "aggregate_importance",
    "inter_target_correlation",
    "cap_components",
]


def _as_2d(Y: Any) -> np.ndarray:
    """Return ``Y`` as a float ``(n_samples, n_targets)`` array.

    A 1-D input is treated as a single target column ``(n_samples, 1)``.
    """
    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class FoldYScaler:
    """Per-target (per-column) train-fold Y autoscaler for JOINT models.

    Fits one mean/std per target column on the training fold and applies the
    same transform to train and validation targets. JOINT models are fit on
    the scaled Y so no single high-variance target dominates the shared
    fitting criterion; predictions are inverse-transformed back to raw units
    before any metric is computed.

    This is JOINT-model machinery only. INDEPENDENT (per-target-separable)
    estimators must fit on RAW Y and must NOT use this scaler, otherwise the
    "separate per-target models" guarantee (scale-sensitive ``alpha`` /
    ``epsilon``) breaks.

    Zero-variance columns get ``std = 1.0`` so the transform is a pure
    centering (no divide-by-zero).
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, Y: Any) -> "FoldYScaler":
        """Fit per-column mean/std jointly on the full target block.

        Args:
            Y: Target block, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.

        Returns:
            self
        """
        arr = _as_2d(Y)
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        std = np.where(std == 0.0, 1.0, std)
        self.std_ = std
        return self

    def transform(self, Y: Any) -> np.ndarray:
        """Standardize ``Y`` to zero-mean/unit-std per target column.

        Returns a ``(n_samples, n_targets)`` array regardless of input rank.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FoldYScaler.transform called before fit.")
        arr = _as_2d(Y)
        return (arr - self.mean_) / self.std_

    def fit_transform(self, Y: Any) -> np.ndarray:
        """Convenience: :meth:`fit` then :meth:`transform`."""
        return self.fit(Y).transform(Y)

    def inverse_transform(self, Y: Any) -> np.ndarray:
        """Map scaled predictions back to raw target units.

        Returns a ``(n_samples, n_targets)`` array regardless of input rank.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FoldYScaler.inverse_transform called before fit.")
        arr = _as_2d(Y)
        return arr * self.std_ + self.mean_


def _slice_fit_params(fit_params: Optional[dict], train_idx: np.ndarray, n_samples: int) -> dict:
    """Slice per-sample array fit params to a training fold; pass scalars through."""
    if not fit_params:
        return {}
    sliced: dict[str, Any] = {}
    for key, value in fit_params.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == n_samples:
            sliced[key] = value[train_idx]
        else:
            sliced[key] = value
    return sliced


def multi_y_cv_pool(
    estimator: Any,
    X: Any,
    Y: Any,
    cv: Any,
    *,
    scale_y: bool = True,
    fit_params: Optional[dict] = None,
    task_type: str = "regression",
    n_folds: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Shape-aware cross-validated predictions pooled to RAW target units.

    This is the multi-target replacement for the 1-D
    :func:`cv_utils.cross_val_predict_pooled`, whose repeated-CV branch
    allocates ``pred_sum = np.zeros(n_samples)`` and ravels predictions —
    unusable for ``(n, n_targets)`` output.

    For JOINT models (``scale_y=True``) a fresh :class:`FoldYScaler` is fit on
    each training fold's Y, the model is fit on scaled Y, and predictions are
    inverse-transformed to raw units *before* pooling — so the returned arrays
    (and hence all downstream metrics) are always in raw units. For
    INDEPENDENT models (``scale_y=False``) the estimator is fit on raw Y.

    Predictions are pooled per sample (averaged across repeats for repeated
    CV); for non-repeated K-fold / LOO each sample is tested exactly once, so
    the pooled arrays are equivalent to legacy concatenation of fold outputs.

    **Single-target byte-identity.** When ``Y`` resolves to one target column,
    ``scale_y`` is ignored and predictions are produced by the exact legacy
    :func:`cv_utils.cross_val_predict_pooled` on the RAW 1-D target. With one
    target there is no coupling to preserve, and the JOINT scale/inverse-
    transform round-trip would otherwise perturb pooled predictions at the
    ~1e-15 level, breaking ``np.array_equal`` byte-identity with the legacy
    single-Y path (the T-17 consolidation guardrail).

    Args:
        estimator: An unfitted sklearn-compatible estimator; cloned per fold.
        X: Feature matrix, shape ``(n_samples, n_features)``.
        Y: Target block, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
        cv: A fitted CV splitter object, or a strategy string
            (``'kfold'`` / ``'repeated_kfold'`` / ``'loo'``) which is built via
            :func:`cv_utils.build_cv_splitter`.
        scale_y: If True (JOINT), fit a per-fold :class:`FoldYScaler` and
            inverse-transform predictions. If False (INDEPENDENT), fit on raw Y.
        fit_params: Optional per-fold fit kwargs; array values whose length
            matches ``X`` are sliced per train index.
        task_type: Passed to :func:`build_cv_splitter` when ``cv`` is a string.
        n_folds: Passed to :func:`build_cv_splitter` when ``cv`` is a string.
        n_repeats: Passed to :func:`build_cv_splitter` when ``cv`` is a string.
        random_state: Passed to :func:`build_cv_splitter` when ``cv`` is a string.

    Returns:
        ``(Y_true_pooled, Y_pred_pooled)`` both shape ``(n_tested, n_targets)``
        in raw target units, aligned row-for-row.
    """
    X_arr = np.asarray(X, dtype=float)
    Y_arr = _as_2d(Y)
    n_samples, n_targets = Y_arr.shape

    if isinstance(cv, str):
        cv = build_cv_splitter(
            cv, n_folds, task_type, n_repeats=n_repeats, random_state=random_state, y=None
        )

    # Single-target byte-identity branch (T-17 consolidation guardrail). With
    # one target there is no coupling to preserve, so JOINT fold Y-scaling would
    # only introduce a scale/inverse-transform round-trip that perturbs the
    # pooled predictions at the ~1e-15 level (breaking byte-identity with the
    # legacy raw-Y path). Route straight through the exact legacy pooler on the
    # RAW 1-D target so single-Y predictions are np.array_equal to legacy.
    if n_targets == 1:
        from .cv_utils import cross_val_predict_pooled

        pooled = cross_val_predict_pooled(
            estimator, X_arr, Y_arr[:, 0], cv, fit_params=fit_params
        )
        Y_pred = np.asarray(pooled, dtype=float).reshape(-1, 1)
        return Y_arr, Y_pred

    pred_sum = np.zeros((n_samples, n_targets), dtype=float)
    pred_count = np.zeros(n_samples, dtype=float)

    for train_idx, test_idx in cv.split(X_arr, Y_arr):
        est = clone(estimator)
        Y_train = Y_arr[train_idx]
        if scale_y:
            scaler = FoldYScaler().fit(Y_train)
            Y_fit = scaler.transform(Y_train)
        else:
            scaler = None
            Y_fit = Y_train
        est.fit(
            X_arr[train_idx],
            Y_fit,
            **_slice_fit_params(fit_params, train_idx, n_samples),
        )
        pred = np.asarray(est.predict(X_arr[test_idx]), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if scaler is not None:
            pred = scaler.inverse_transform(pred)
        pred_sum[test_idx] += pred
        pred_count[test_idx] += 1

    mask = pred_count > 0
    Y_pred = pred_sum[mask] / pred_count[mask, None]
    return Y_arr[mask], Y_pred


def multi_y_metrics(
    Y_true: Any,
    Y_pred: Any,
    target_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Per-target RAW-unit metrics plus the joint-Q2 selection criterion.

    All inputs are assumed to already be in RAW target units (JOINT
    predictions are inverse-transformed inside :func:`multi_y_cv_pool`).

    Per target, in raw units:

    * ``r2`` — scale-invariant; ``q2`` equals ``1 - PRESS/SSY`` with SSY
      referenced to the pooled-test mean, so ``q2 == r2`` here and, at
      ``n_targets == 1``, equals legacy ``r2_score(all_y_test, all_y_pred)``.
    * ``rmse`` — RAW units (scale-sensitive; catches a units bug that R2 hides).
    * ``rpd`` — ``std(y_true) / RMSEcv`` (Williams RPD tiers).
    * ``rer`` — ``range(y_true) / RMSEcv``.
    * ``ccc`` — Lin's concordance CCCcv.
    * ``bias`` — ``mean(pred - true)`` (positive = systematic overprediction).

    The joint selection criterion is ``joint_q2 = mean(per-target q2)``.

    Args:
        Y_true: Raw pooled truths, shape ``(n, n_targets)`` or ``(n,)``.
        Y_pred: Raw pooled predictions, same shape as ``Y_true``.
        target_names: Optional per-target labels; defaults to ``target_0`` etc.

    Returns:
        Dict with per-target arrays (``r2``, ``rmse``, ``q2``, ``rpd``,
        ``rer``, ``ccc``, ``bias``), a ``per_target`` list of dicts, the
        scalar ``joint_q2``, and ``target_names``.
    """
    yt = _as_2d(Y_true)
    yp = _as_2d(Y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Y_true shape {yt.shape} != Y_pred shape {yp.shape} in multi_y_metrics.")
    n_targets = yt.shape[1]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]
    elif len(target_names) != n_targets:
        raise ValueError(
            f"target_names has {len(target_names)} entries but Y has {n_targets} targets."
        )

    r2 = np.empty(n_targets)
    rmse = np.empty(n_targets)
    q2 = np.empty(n_targets)
    rpd = np.empty(n_targets)
    rer = np.empty(n_targets)
    ccc = np.empty(n_targets)
    bias = np.empty(n_targets)

    for s in range(n_targets):
        ts = yt[:, s]
        ps = yp[:, s]
        residual = ps - ts
        press = float(np.sum(residual**2))
        ssy = float(np.sum((ts - ts.mean()) ** 2))
        if ssy > 0:
            q2_s = 1.0 - press / ssy
        else:
            # Constant target (SSY == 0): match sklearn r2_score(force_finite=True)
            # exactly -- perfect prediction (PRESS == 0) scores 1.0, otherwise 0.0.
            # Preserves the "per-target Q2 == legacy r2_score at n_targets==1" contract.
            q2_s = 1.0 if press == 0.0 else 0.0
        q2[s] = q2_s
        r2[s] = q2_s  # raw-unit R2 == Q2 (pooled-mean reference)
        rmse_s = float(np.sqrt(press / len(ts)))
        rmse[s] = rmse_s
        y_std = float(np.std(ts))
        rpd[s] = y_std / rmse_s if rmse_s > 0 else 0.0
        rer[s] = float(np.ptp(ts)) / rmse_s if rmse_s > 0 else 0.0
        ccc[s] = lins_ccc(ts, ps)
        bias[s] = float(np.mean(residual))

    per_target = [
        {
            "target": target_names[s],
            "r2": float(r2[s]),
            "rmse": float(rmse[s]),
            "q2": float(q2[s]),
            "rpd": float(rpd[s]),
            "rer": float(rer[s]),
            "ccc": float(ccc[s]),
            "bias": float(bias[s]),
        }
        for s in range(n_targets)
    ]

    return {
        "target_names": list(target_names),
        "r2": r2,
        "rmse": rmse,
        "q2": q2,
        "rpd": rpd,
        "rer": rer,
        "ccc": ccc,
        "bias": bias,
        "per_target": per_target,
        "joint_q2": float(np.mean(q2)),
    }


def reduce_multi_y_score(per_target: Any, rule: str = "mean") -> float:
    """Reduce a per-target CV score vector to a single scalar.

    Available helper for collapsing a per-target Q2/RMSE vector to one number
    (varsel/component-selection paths currently compute their joint criterion
    inline rather than calling this). ``mean`` is the correct default because
    per-target Q2 is scale-invariant on raw units, so equal weighting is
    automatic.

    Args:
        per_target: Array-like of per-target scores. A scalar is returned as-is.
        rule: One of ``'mean'``, ``'median'``, ``'min'``, ``'max'``, ``'sum'``.

    Returns:
        Scalar reduction of ``per_target``.
    """
    arr = np.asarray(per_target, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("reduce_multi_y_score got an empty per_target vector.")
    reducers = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
    }
    if rule not in reducers:
        raise ValueError(f"Unknown reduce rule {rule!r}. Expected one of {sorted(reducers)}.")
    return float(reducers[rule](arr))


def extract_pls_multi_y(pls: Any) -> np.ndarray:
    """Return the PLS regression B-matrix as ``(n_features, n_targets)``.

    sklearn >= 1.1 (verified on pinned 1.8) exposes
    ``PLSRegression.coef_`` as ``(n_targets, n_features)`` — the transpose of
    the pre-1.1 layout — so the canonical chemometrics B-matrix
    ``(n_features, n_targets)`` is ``pls.coef_.T``. Single-target PLS yields
    ``(n_features, 1)``.

    Accepts either a raw ``PLSRegression`` or a ``PLSTransformer`` wrapper
    (unwrapped via ``.pls_``).

    Args:
        pls: A fitted PLS estimator exposing ``coef_``.

    Returns:
        B-matrix, shape ``(n_features, n_targets)``.
    """
    inner = getattr(pls, "pls_", pls)
    coef = np.asarray(inner.coef_)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return coef.T


def aggregate_importance(matrix: Any, rule: str = "mean") -> np.ndarray:
    """Aggregate a per-target importance matrix to one score per feature.

    Single home for cross-target aggregation rules. Generic reductions plus the
    two method-specific T-17 rules:

    - ``uve_stability`` (UVE multi-Y): mean across targets of the per-target
      reliability ratio ``mean(|coef|)/std(coef)``. The ratio is already
      scale-invariant per target, so the mean weights every target's stability
      equally without any Y-scaling. (Methodology choice -- ``max`` / ``l2``
      would instead reward variables reliable for a single target; flagged for
      an A/B.)
    - ``cars_scaled_coef`` (CARS multi-Y): l2-norm across targets of the
      column-scaled PLS-2 ``|coef|`` matrix. Column scaling is load-bearing --
      raw coefficient magnitudes across differently-scaled targets are
      incomparable. l2 (vs ``mean``) lets a variable strongly informative for
      one target survive the reweighting. (Methodology choice; flagged for an
      A/B.)

    Args:
        matrix: Importance matrix, shape ``(n_features, n_targets)`` (a 1-D
            input is treated as a single target column).
        rule: One of ``'mean'``, ``'sum'``, ``'max'``, ``'l2'`` (root sum of
            squares across targets), ``'uve_stability'`` (== mean), or
            ``'cars_scaled_coef'`` (== l2).

    Returns:
        Per-feature scores, shape ``(n_features,)``.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if rule in ("mean", "uve_stability"):
        return arr.mean(axis=1)
    if rule == "sum":
        return arr.sum(axis=1)
    if rule == "max":
        return arr.max(axis=1)
    if rule in ("l2", "cars_scaled_coef"):
        return np.sqrt(np.sum(arr**2, axis=1))
    raise ValueError(
        f"Unknown aggregate rule {rule!r}. Expected 'mean', 'sum', 'max', 'l2', "
        f"'uve_stability', or 'cars_scaled_coef'."
    )


def inter_target_correlation(Y: Any, weak_threshold: float = 0.35) -> dict[str, Any]:
    """Pairwise Pearson correlation guardrail across targets.

    Weak mean absolute pairwise correlation among targets suggests separate
    PLS-1 models are likely better than a coupled JOINT model. No-op at
    ``n_targets == 1`` (``mean_abs_corr = 0.0``, never weak).

    Args:
        Y: Target block, shape ``(n_samples, n_targets)`` or ``(n_samples,)``.
        weak_threshold: Mean-abs-correlation below which ``is_weak`` is True.

    Returns:
        Dict with ``corr_matrix`` (``(n_targets, n_targets)``),
        ``mean_abs_corr`` (mean of absolute off-diagonal entries), and
        ``is_weak`` (bool).
    """
    arr = _as_2d(Y)
    n_targets = arr.shape[1]
    if n_targets < 2:
        return {
            "corr_matrix": np.ones((n_targets, n_targets)),
            "mean_abs_corr": 0.0,
            "is_weak": False,
        }
    corr = np.corrcoef(arr, rowvar=False)
    off = ~np.eye(n_targets, dtype=bool)
    mean_abs = float(np.nanmean(np.abs(corr[off])))
    return {
        "corr_matrix": corr,
        "mean_abs_corr": mean_abs,
        "is_weak": mean_abs < weak_threshold,
    }


def cap_components(n_samples: int, n_features: int, requested: Optional[int] = None) -> int:
    """Cap PLS latent components at ``A <= min(n_samples - 1, n_features)``.

    Args:
        n_samples: Number of training samples (``N``).
        n_features: Number of features (``p``).
        requested: Desired component count; if None the maximum cap is returned.

    Returns:
        A valid component count ``>= 1`` (clamped to the cap, floored at 1).
    """
    cap = min(int(n_samples) - 1, int(n_features))
    cap = max(cap, 1)
    if requested is None:
        return cap
    return max(1, min(int(requested), cap))
