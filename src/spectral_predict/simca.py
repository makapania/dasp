"""Multi-class class modeling (T-31).

Implements :class:`MultiClassClassModel` — K independent per-class membership
models (SIMCA flagship) evaluated at a *global* level ``alpha``, producing a
per-sample x per-class decision matrix and a single / multiple / novel summary
label.

This is task **A2 + A3 + A4** of the T-31 build: the core orchestrator with
the SIMCA (``"pca-simca"``) engine, three column-scaling modes
(``scaling="per_class"`` is the textbook default), and four non-SIMCA
per-class engines (``"ocsvm"`` / ``"isolation-forest"`` / ``"lof"`` /
``"elliptic-envelope"``) each calibrated to a per-class empirical p-value
(spec section 5.3). Per-class CV ``n_components`` tuning (A5), novelty
evaluation (A6), dedicated metrics (A7), and ``.dasp`` persistence (A8) arrive
in subsequent tasks.

References
----------
- Pomerantsev, A.L. & Rodionova, O.Y. DD-SIMCA (Hotelling T^2 + Q / SPE with
  data-driven chi-squared fits and Fisher-combined joint p-value).
- Oliveri, P. & Downey, G. (2012). Multivariate class modeling for the
  authentication of foods — the belongs-to-none / one / several decision rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from spectral_predict.contamination import (
    PCASIMCA,
    get_one_class_model,
    get_one_class_model_grids,
    run_one_class_cv,
)


# Non-SIMCA per-class engines (spec section 5.3 / task A4). Maps a
# MultiClassClassModel engine name to ``(builder_name, score_method)`` where
# ``builder_name`` is understood by ``contamination.get_one_class_model``
# (sensible NIR defaults: LOF novelty=True, IsolationForest random_state=42)
# and ``score_method`` is the sklearn method returning the "higher = more
# normal" score. PINNED per engine: IsolationForest uses ``score_samples``
# AS-IS (do NOT negate -- sklearn already returns higher=more-normal;
# GPT-5.5 verified). Negating breaks test_isolationforest_direction_not_inverted.
_NON_SIMCA_ENGINES: dict[str, tuple[str, str]] = {
    "ocsvm": ("OneClassSVM", "decision_function"),
    "isolation-forest": ("IsolationForest", "score_samples"),
    "lof": ("LOF", "decision_function"),
    "elliptic-envelope": ("EllipticEnvelope", "decision_function"),
}


class MultiClassClassModel(BaseEstimator, ClassifierMixin):
    """Multi-class class-modeling orchestrator.

    Fits one membership model per class on that class's training rows only,
    then evaluates every sample against every modeled class at a *global*
    level ``alpha``. A sample may be accepted by zero, one, or several class
    models, yielding the class-modeling outcomes ``"novel"`` / single-class /
    ``"multiple"`` (Oliveri & Downey 2012; spec section 5.2).

    Parameters
    ----------
    engine : str, default="pca-simca"
        Per-class membership engine. One of:

        - ``"pca-simca"`` (DD-SIMCA via
          :class:`~spectral_predict.contamination.PCASIMCA`; flagship).
        - ``"ocsvm"`` / ``"isolation-forest"`` / ``"lof"`` / ``"elliptic-envelope"``
          -- non-SIMCA one-class engines (built via
          :func:`~spectral_predict.contamination.get_one_class_model`), each
          calibrated to a per-class empirical p-value via a cross-fit null on
          that class's training rows (spec section 5.3 / task A4).
    alpha : float, default=0.05
        Global significance level shared by every per-class model, so the
        ">= 2 of K" multiple-membership rule is coherent across classes
        (spec decision #6).
    n_components : int, dict, or "per_class_cv", default="per_class_cv"
        PCA components for the ``"pca-simca"`` engine. An int applies the same
        value to every class; a ``{class_label: int}`` mapping sets it per
        class; the string ``"per_class_cv"`` (default, task A5) tunes each
        modeled class's components by one-vs-rest CV via
        :func:`~spectral_predict.contamination.run_one_class_cv` (candidate
        grid = distinct int ``n_components`` from
        :func:`~spectral_predict.contamination.get_one_class_model_grids`;
        alpha stays global / fixed). Unused by non-SIMCA engines.
    scaling : {"none", "per_class", "global"}, default="per_class"
        Column-scaling mode (spec section 5.5). All scalers are
        :class:`~sklearn.preprocessing.StandardScaler` instances fit train-only
        inside :meth:`fit`.

        - ``"per_class"`` (SIMCA-textbook default): one scaler per class, fit on
          THAT class's rows only, then used to autoscale that class's rows
          before the per-class PCA. At decision time, column ``k`` transforms X
          with class ``k``'s scaler before scoring. Stored in ``scalers_``.
        - ``"global"``: a single scaler fit on ALL training rows across classes
          (same scaler applied to every class column — useful for cross-engine
          comparability). Stored in ``global_scaler_``.
        - ``"none"``: X is passed through unchanged (functionally identical to
          a bare :class:`PCASIMCA` per class).
    min_class_samples : int, default=10
        Minimum training rows a class must have to be modeled (the MoM chi^2
        fit / calibration is brittle below this; spec section 5.1 / 8). Classes
        below the threshold are marked unmodelable: their decision-matrix
        column is preserved (NaN / never-accept) and their label is recorded in
        ``unmodelable_`` — never silently dropped.
    engine_params : dict, optional
        Forward-compatibility hook for per-engine hyperparameters (unused by
        the ``"pca-simca"`` engine in A2).

    Attributes
    ----------
    classes_ : ndarray
        Sorted unique class labels of ``y`` (original dtype preserved).
    models_ : dict
        Fitted per-class engines keyed by class label. Unmodelable classes are
        absent from this dict.
    nulls_ : dict
        Per-class cross-fit null score arrays (non-SIMCA engines only, task
        A4), keyed by class label. Each value is a sorted-ascending ndarray of
        out-of-fold "higher = more normal" scores used by the empirical
        p-value calibration. Empty for the ``"pca-simca"`` engine.
    unmodelable_ : set
        Class labels with fewer than ``min_class_samples`` training rows.
    scalers_ : dict
        Per-class :class:`~sklearn.preprocessing.StandardScaler` instances,
        keyed by class label. Populated only under ``scaling="per_class"``;
        empty dict otherwise.
    global_scaler_ : StandardScaler or None
        Single scaler fit on all training rows. Populated only under
        ``scaling="global"``; ``None`` otherwise.
    """

    def __init__(
        self,
        engine: str = "pca-simca",
        alpha: float = 0.05,
        n_components: int | dict | str = "per_class_cv",
        scaling: str = "per_class",
        min_class_samples: int = 10,
        engine_params: dict | None = None,
    ):
        self.engine = engine
        self.alpha = alpha
        self.n_components = n_components
        self.scaling = scaling
        self.min_class_samples = min_class_samples
        self.engine_params = engine_params

    def fit(self, X, y):
        """Fit one per-class membership model on each class's rows only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training spectra.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : MultiClassClassModel
            Fitted orchestrator with ``classes_``, ``models_``, and
            ``unmodelable_`` populated.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # --- validate constructor choices (sklearn-style: in fit) ------------
        if self.scaling not in ("none", "per_class", "global"):
            raise ValueError(
                f"Unknown scaling={self.scaling!r}; "
                'expected "none", "per_class", or "global".'
            )

        if self.engine != "pca-simca" and self.engine not in _NON_SIMCA_ENGINES:
            raise ValueError(
                f"Unknown engine={self.engine!r}; expected 'pca-simca' or one of "
                f"{sorted(_NON_SIMCA_ENGINES)}."
            )

        # --- scaling setup (all scalers fit train-only inside fit) ------------
        # scalers_: {class_label: StandardScaler} populated only under per_class;
        # global_scaler_: single StandardScaler populated only under global.
        self.scalers_: dict = {}
        self.global_scaler_ = None
        if self.scaling == "global":
            self.global_scaler_ = StandardScaler().fit(X)

        # --- class registry (sorted, dtype-preserving) -----------------------
        self.classes_ = np.unique(y)
        self.unmodelable_: set = set()
        self.models_: dict = {}
        # Per-class cross-fit null arrays (non-SIMCA engines only, task A4).
        # Each value is sorted ascending so _empirical_p can use searchsorted.
        self.nulls_: dict = {}
        # Per-class resolved PCA n_components (task A5). {class_label: int};
        # populated for modeled SIMCA classes only (empty for non-SIMCA engines
        # and unmodelable classes).
        self.n_components_: dict = {}

        is_simca = self.engine == "pca-simca"

        for c in self.classes_:
            X_c = X[y == c]
            if X_c.shape[0] < self.min_class_samples:
                # Too few rows for a reliable MoM chi^2 fit / cross-fit null:
                # mark unmodelable, keep the decision-matrix column, fit no
                # scaler/engine (spec section 8).
                self.unmodelable_.add(c)
                continue

            # Apply this class's scaling mode (train-only fit) before the engine.
            if self.scaling == "per_class":
                self.scalers_[c] = StandardScaler().fit(X_c)
                X_c_scaled = self.scalers_[c].transform(X_c)
            elif self.scaling == "global":
                X_c_scaled = self.global_scaler_.transform(X_c)
            else:  # "none" -- X passes through unchanged
                X_c_scaled = X_c

            if is_simca:
                if self.n_components == "per_class_cv":
                    # Tune this class's n_components by one-vs-rest CV (task A5).
                    # X_s = the SAME column-scaling space the final model uses
                    # for class c: per_class -> transform ALL X by class c's
                    # scaler; global -> the global scaler; none -> raw X.
                    if self.scaling == "per_class":
                        X_s = self.scalers_[c].transform(X)
                    elif self.scaling == "global":
                        X_s = self.global_scaler_.transform(X)
                    else:  # "none"
                        X_s = X
                    y_oc = np.where(y == c, 1, -1)
                    n_components_c = self._tune_per_class_n_components(
                        X_s, y_oc, X_c.shape[0], X.shape[1]
                    )
                else:
                    n_components_c = self._n_components_for(c)
                self.n_components_[c] = n_components_c
                self.models_[c] = PCASIMCA(
                    n_components=n_components_c, alpha=self.alpha
                ).fit(X_c_scaled)
            else:
                builder_name, score_method = _NON_SIMCA_ENGINES[self.engine]
                final_engine = get_one_class_model(builder_name)
                final_engine.fit(X_c_scaled)
                self.models_[c] = final_engine
                # Cross-fit null on this class's scaled rows (train-only) so the
                # empirical p-value is a real level-alpha test (spec section 5.3).
                self.nulls_[c] = self._cross_fit_null(
                    X_c_scaled, builder_name, score_method
                )

        return self

    def decision_matrix(self, X):
        """Per-sample x per-class p-values and acceptance flags.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        P : ndarray of shape (n_samples, K), dtype float
            Calibrated per-class p-values. For the ``"pca-simca"`` engine this
            is ``PCASIMCA.p_joint`` (the chi^2(4) upper-tail of the Fisher-combined
            T^2 / Q statistic). Unmodelable columns are ``NaN``.
        A : ndarray of shape (n_samples, K), dtype bool
            ``P[:, k] >= alpha`` for modeled classes; always ``False`` for
            unmodelable columns. Columns are ordered by ``classes_``
            (``K = len(classes_)``).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        K = len(self.classes_)
        P = np.full((n_samples, K), np.nan, dtype=np.float64)
        A = np.zeros((n_samples, K), dtype=bool)

        is_simca = self.engine == "pca-simca"
        if not is_simca:
            _, score_method = _NON_SIMCA_ENGINES[self.engine]

        for k, c in enumerate(self.classes_):
            if c in self.unmodelable_:
                continue  # leave P[:, k] = NaN and A[:, k] = False
            # Apply the same scaling mode used at fit time before scoring.
            if self.scaling == "per_class":
                X_eval = self.scalers_[c].transform(X)
            elif self.scaling == "global":
                X_eval = self.global_scaler_.transform(X)
            else:  # "none"
                X_eval = X
            if is_simca:
                P[:, k] = self.models_[c].p_joint(X_eval)
            else:
                scores = getattr(self.models_[c], score_method)(X_eval)
                P[:, k] = self._empirical_p(scores, self.nulls_[c])
            A[:, k] = P[:, k] >= self.alpha

        return P, A

    def predict(self, X):
        """Summary label per sample: single class, ``"multiple"``, or ``"novel"``.

        Implements the Oliveri & Downey (2012) decision rule at global level
        ``alpha``: with ``n = A[i].sum()`` acceptances for row ``i``,

        - ``n == 0`` -> ``"novel"`` (belongs to none of the modeled classes),
        - ``n == 1`` -> the single accepted class label (original dtype),
        - ``n >= 2`` -> ``"multiple"`` (ambiguous; accepted by >= 2 of K models).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=object
            Per-row summary label (see the three cases above).
        """
        _, A = self.decision_matrix(X)
        n_samples = A.shape[0]
        out = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
            accepted = np.where(A[i])[0]
            n_accept = accepted.size
            if n_accept == 0:
                out[i] = "novel"
            elif n_accept == 1:
                out[i] = self.classes_[accepted[0]]
            else:
                out[i] = "multiple"

        return out

    def cross_validate(self, X, y, n_splits: int = 5) -> dict:
        """Nested leakage-safe outer CV (task A5).

        For each outer fold a FRESH :class:`MultiClassClassModel` is fit on the
        fold's training rows only (clone-style: same engine / alpha /
        n_components / scaling / min_class_samples / engine_params as this
        instance), so any ``"per_class_cv"`` tuning runs on fold-train only —
        no outer leakage. The fold's held-out rows are then scored into
        out-of-fold (OOF) ``P`` / ``A`` arrays and summary labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_splits : int, default=5
            Outer fold count. Reduced to ``min(n_splits, <smallest class
            count>)`` when a class has fewer rows than ``n_splits``
            (StratifiedKFold requirement); clamped to a minimum of 2.

        Returns
        -------
        dict
            With exactly these keys:

            - ``"labels"``: ``(n_samples,)`` object OOF summary labels.
            - ``"decision_matrix"``: ``(P, A)`` tuple of ``(n_samples, K)``
              OOF p-values (float; ``NaN`` where a class is unmodelable or
              absent from a fold) and acceptance flags (bool).
            - ``"classes"``: global sorted unique class labels (column order).
            - ``"test_indices"``: list of per-fold test-index ndarrays.
            - ``"train_indices"``: list of per-fold train-index ndarrays (same
              fold order as ``test_indices``).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        K = len(classes)
        n = X.shape[0]

        min_class_count = int(min(np.sum(y == c) for c in classes))
        n_splits = max(2, min(n_splits, min_class_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

        P = np.full((n, K), np.nan, dtype=np.float64)
        A = np.zeros((n, K), dtype=bool)
        labels = np.empty(n, dtype=object)

        test_indices: list[np.ndarray] = []
        train_indices: list[np.ndarray] = []

        for train_idx, test_idx in skf.split(X, y):
            fold_model = MultiClassClassModel(
                engine=self.engine,
                alpha=self.alpha,
                n_components=self.n_components,
                scaling=self.scaling,
                min_class_samples=self.min_class_samples,
                engine_params=self.engine_params,
            )
            fold_model.fit(X[train_idx], y[train_idx])

            P_fold, A_fold = fold_model.decision_matrix(X[test_idx])
            fold_classes = fold_model.classes_
            for k_global, c in enumerate(classes):
                if c in fold_model.unmodelable_:
                    continue  # leave NaN / False
                matches = np.where(fold_classes == c)[0]
                if matches.size == 0:
                    continue  # class absent from fold-train -> leave NaN / False
                k_fold = int(matches[0])
                P[test_idx, k_global] = P_fold[:, k_fold]
                A[test_idx, k_global] = A_fold[:, k_fold]

            labels[test_idx] = fold_model.predict(X[test_idx])

            test_indices.append(test_idx)
            train_indices.append(train_idx)

        return {
            "labels": labels,
            "decision_matrix": (P, A),
            "classes": classes,
            "test_indices": test_indices,
            "train_indices": train_indices,
        }

    def evaluate_novelty(
        self, X, y, mode: str = "loco", external_X=None
    ) -> dict | float:
        """Leakage-safe novelty evaluation (task A6, spec section 5.4).

        Fits FRESH :class:`MultiClassClassModel` instances (constructed with
        the same ``engine`` / ``alpha`` / ``n_components`` / ``scaling`` /
        ``min_class_samples`` / ``engine_params`` as this instance) so no
        already-fitted state is reused -- the same leakage discipline as
        :meth:`cross_validate`.

        Two modes:

        - ``mode="loco"`` (leave-one-class-out): for each class ``c`` in
          ``np.unique(y)``, fit a fresh model on the ``K - 1`` REMAINING
          classes (``X[y != c]``, ``y[y != c]``) -- the held-out class NEVER
          enters the fitting data -- then predict the held-out class's rows
          ``X[y == c]``. Its novelty rate is the fraction of those rows
          accepted by NONE of the ``K - 1`` remaining class models,
          ``mean(predict(X[y == c]) == "novel")``.
        - ``mode="external"`` (requires ``external_X``): fit a fresh model on
          ALL of ``(X, y)`` and predict ``external_X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Known-class spectra.
        y : array-like of shape (n_samples,)
            Known class labels.
        mode : {"loco", "external"}, default="loco"
        external_X : array-like of shape (n_external, n_features), optional
            Required for ``mode="external"``; ignored for ``mode="loco"``.

        Returns
        -------
        dict or float
            - ``mode="loco"``: ``{class_label: novelty_rate}`` whose keys are
              exactly ``set(np.unique(y))``; each value is a float in
              ``[0, 1]``.
            - ``mode="external"``: a float in ``[0, 1]`` -- the fraction of
              ``external_X`` flagged ``"novel"``.

        Raises
        ------
        ValueError
            If ``mode`` is neither ``"loco"`` nor ``"external"``, or if
            ``mode="external"`` and ``external_X`` is ``None``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if mode == "loco":
            classes = np.unique(y)
            rates: dict = {}
            for c in classes:
                train_mask = y != c
                held_out = MultiClassClassModel(
                    engine=self.engine,
                    alpha=self.alpha,
                    n_components=self.n_components,
                    scaling=self.scaling,
                    min_class_samples=self.min_class_samples,
                    engine_params=self.engine_params,
                )
                held_out.fit(X[train_mask], y[train_mask])
                preds = held_out.predict(X[y == c])
                rates[c] = float(np.mean(preds == "novel"))
            return rates

        if mode == "external":
            if external_X is None:
                raise ValueError(
                    'mode="external" requires external_X '
                    "(a separate held-out set to score)."
                )
            external_X = np.asarray(external_X, dtype=np.float64)
            model = MultiClassClassModel(
                engine=self.engine,
                alpha=self.alpha,
                n_components=self.n_components,
                scaling=self.scaling,
                min_class_samples=self.min_class_samples,
                engine_params=self.engine_params,
            )
            model.fit(X, y)
            preds = model.predict(external_X)
            return float(np.mean(preds == "novel"))

        raise ValueError(
            f"Unknown mode={mode!r}; expected 'loco' or 'external'."
        )

    def _n_components_for(self, label) -> int:
        """Resolve ``n_components`` for a single class label.

        ``n_components`` may be an int (same for all classes) or a
        ``{class_label: int}`` dict. ``label`` is a scalar from ``classes_``;
        numpy integer scalars hash-equal to Python ints, so dict lookup works
        regardless of whether the user keyed the dict with Python or numpy ints.
        """
        nc = self.n_components
        if isinstance(nc, dict):
            return int(nc[label])
        return int(nc)

    def _tune_per_class_n_components(
        self, X_s: np.ndarray, y_oc: np.ndarray, n_class: int, n_features: int
    ) -> int:
        """Select this class's PCA ``n_components`` by one-vs-rest CV (task A5).

        Candidate grid = the distinct integer ``n_components`` values from the
        ``"PCA-SIMCA"`` entry of
        :func:`~spectral_predict.contamination.get_one_class_model_grids` (its
        alpha axis is ignored — alpha is fixed at ``self.alpha``). Each
        candidate is capped at ``min(n_class - 1, n_features)``; non-integer
        (e.g. variance-fraction floats), non-positive, and duplicate candidates
        are dropped. For each surviving candidate a one-vs-rest CV is run via
        :func:`~spectral_predict.contamination.run_one_class_cv` on this class's
        column-scaled space ``X_s`` and the candidate with the highest mean
        balanced_accuracy wins (ties broken toward the smallest
        ``n_components`` since candidates are visited ascending with strict
        ``>``). Falls back to ``min(5, n_class - 1)`` if the grid is empty or
        every candidate is skipped.

        Parameters
        ----------
        X_s : ndarray of shape (n_samples, n_features)
            Full ``X`` projected into the SAME column-scaling space the final
            model uses for this class (per_class/global scaler applied, or raw
            under ``scaling="none"``).
        y_oc : ndarray of shape (n_samples,)
            One-vs-rest labels: ``+1`` for this class, ``-1`` otherwise.
        n_class : int
            Number of training rows in this class (used for the cap).
        n_features : int
            Number of features (used for the cap).

        Returns
        -------
        n_components : int
            The chosen int ``n_components`` for this class.
        """
        cap = min(n_class - 1, n_features)
        grid = get_one_class_model_grids().get("PCA-SIMCA", [])
        candidates: list[int] = []
        seen: set[int] = set()
        for entry in grid:
            nc = entry.get("n_components")
            if isinstance(nc, bool) or not isinstance(nc, (int, np.integer)):
                continue  # drop variance-fraction (float) candidates
            nc = min(int(nc), cap)
            if nc < 1 or nc in seen:
                continue
            seen.add(nc)
            candidates.append(nc)
        candidates.sort()
        if not candidates:
            return min(5, max(1, n_class - 1))

        best_nc: int | None = None
        best_score = -np.inf
        for nc in candidates:
            try:
                result = run_one_class_cv(
                    X_s,
                    y_oc,
                    "PCA-SIMCA",
                    {"n_components": nc, "alpha": self.alpha},
                    compute_calibration=False,
                )
            except Exception:
                continue
            if result.get("skipped"):
                continue
            score = result.get("mean_metrics", {}).get("balanced_accuracy", np.nan)
            if not np.isfinite(score):
                continue
            if score > best_score:  # strict > -> ties keep the smallest nc
                best_score = score
                best_nc = nc
        if best_nc is None:
            return min(5, max(1, n_class - 1))
        return best_nc

    def _cross_fit_null(
        self, X_scaled: np.ndarray, builder_name: str, score_method: str
    ) -> np.ndarray:
        """Build the cross-fit "normality" null array for one class (train-only).

        K-fold (5-fold, or fewer if the class is small) on the class's *scaled*
        training rows; for each fold, fit a fresh engine (built via
        :func:`~spectral_predict.contamination.get_one_class_model`) on the
        other folds and score the held-out fold with the pinned
        ``score_method``. Collects the out-of-fold "higher = more normal"
        scores as the empirical null (spec section 5.3 / task A4). Failed fold
        fits are silently skipped (spec).

        Parameters
        ----------
        X_scaled : ndarray of shape (n_class, n_features)
            This class's already-scaled training rows (scaling was applied by
            :meth:`fit` upstream, train-only).
        builder_name : str
            One-class model name understood by ``get_one_class_model``.
        score_method : str
            sklearn method name returning the engine's "higher = more normal"
            score (pinned per engine in :data:`_NON_SIMCA_ENGINES`).

        Returns
        -------
        null : ndarray of shape (m,), sorted ascending
            Out-of-fold scores; ``m`` ~= ``n_class`` (fewer if folds failed).
        """
        n = X_scaled.shape[0]
        n_splits = min(5, n)
        kf = KFold(n_splits=n_splits, shuffle=False)
        scores_list = []
        for train_idx, test_idx in kf.split(X_scaled):
            try:
                fold_engine = get_one_class_model(builder_name)
                fold_engine.fit(X_scaled[train_idx])
                fold_scores = getattr(fold_engine, score_method)(X_scaled[test_idx])
            except Exception:
                continue  # spec: skip failed folds
            scores_list.extend(np.asarray(fold_scores).ravel().tolist())
        return np.asarray(sorted(scores_list), dtype=np.float64)

    @staticmethod
    def _empirical_p(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
        """Add-one-smoothed empirical p-value (spec section 5.3 / task A4).

        ``p = (1 + #{null <= s}) / (m + 1)`` -- lower-tail is anomalous since
        "higher score = more normal". ``null`` must be sorted ascending; uses
        :func:`numpy.searchsorted` (``side="right"``) for vectorized
        ``O((n + m) log m)`` evaluation.

        Returns NaN where ``null`` is empty (defensive -- modeled classes have
        ``min_class_samples`` rows, so a fully-empty null implies every fold
        failed; treated as uninformative).
        """
        m = null.shape[0]
        scores = np.asarray(scores, dtype=np.float64)
        if m == 0:
            return np.full(scores.shape, np.nan, dtype=np.float64)
        counts = np.searchsorted(null, scores, side="right")
        return (1.0 + counts) / (m + 1.0)
