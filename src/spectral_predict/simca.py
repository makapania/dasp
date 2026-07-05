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

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
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
    variable_selection : str or None, default=None
        Optional per-class prefilter computed TRAIN-ONLY inside :meth:`fit`
        and applied to every per-class scaler / PCA / engine (task B3). When
        ``None`` (default) no selection is applied and behavior is
        byte-identical to the un-prefiltered model. Supported values:

        - ``"wold_modeling"`` / ``"wold_discriminating"`` / ``"wold_balanced"``:
          classical Wold (1976) modeling / discriminating / balanced power
          (task B1) via :func:`wold_variable_selection`. Tagged
          ``varsel_path_ = "wold"``.
        - ``"importance"``: supervised per-feature importance from
          :func:`~spectral_predict.unified_bayesian.compute_importances` on the
          genuine multi-class label (``task_type="classification"``). Tagged
          ``varsel_path_ = "supervised"``. NOTE: supervised selection optimizes
          for DISCRIMINATION between the known classes, so it can miss a future
          novel class that is distinctive ONLY on the low-importance features it
          discards; the novelty guard verifies no degradation on representative
          novels but cannot cover that adversarial case (spec §5 guardrail).
        - a boolean ``ndarray`` of shape ``(n_features,)``: a precomputed mask
          used directly (``varsel_path_ = "precomputed"``) — the hook for the C
          search layer to wire any external selection method.
        - Any other string raises :class:`NotImplementedError` at fit time
          (the fuller supervised method set — spa/cars/ga/... — is enumerated
          in the C search layer, not the model layer).
    n_select : int or None, default=None
        Number of variables to keep when ``variable_selection`` is set. If an
        int, the top-``n_select`` variables are kept (by Wold score / by
        importance). If ``None``, a variable is kept when its score is
        ``>= mean(score)``.
    varsel_model_name : str, default="RandomForest"
        Model name forwarded to :func:`compute_importances` for the supervised
        path (unused by the Wold modes).

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
    varsel_mask_ : ndarray or None
        Boolean variable-selection mask of shape ``(n_features,)`` computed
        train-only inside :meth:`fit` when ``variable_selection`` is set; the
        per-class scalers / PCA / engines are fit on ``X[:, varsel_mask_]``
        and :meth:`decision_matrix` masks its input the same way. ``None`` when
        ``variable_selection is None`` (no prefiltering).
    varsel_path_ : str
        Tag identifying which variable-selection path ran at fit time:
        ``"none"`` (no selection), ``"wold"`` (Wold MPOW/DPOW/Balanced), or
        ``"supervised"`` (importance-based prefilter).
    """

    def __init__(
        self,
        engine: str = "pca-simca",
        alpha: float = 0.05,
        n_components: int | dict | str = "per_class_cv",
        scaling: str = "per_class",
        min_class_samples: int = 10,
        engine_params: dict | None = None,
        variable_selection: str | None = None,
        n_select: int | None = None,
        varsel_model_name: str = "RandomForest",
    ):
        self.engine = engine
        self.alpha = alpha
        self.n_components = n_components
        self.scaling = scaling
        self.min_class_samples = min_class_samples
        self.engine_params = engine_params
        self.variable_selection = variable_selection
        self.n_select = n_select
        self.varsel_model_name = varsel_model_name

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

        # --- variable-selection prefilter (task B3, train-only) --------------
        # Compute the mask on the FULL-width (X, y) BEFORE any per-class work;
        # then every per-class scaler / PCA / engine / calibration below is fit
        # on the SELECTED subspace (X_fit). mask=None means no selection and is
        # byte-identical to the pre-B3 behavior.
        self.varsel_mask_: np.ndarray | None = None
        self.varsel_path_: str = "none"
        if self.variable_selection is not None:
            n_features = X.shape[1]
            # n_select sanity (Phase-B gate, Kimi H1/L8) — an empty (n_select=0)
            # or negative-slice mask would surface as a confusing 0-feature
            # DD-SIMCA error deep downstream.
            if self.n_select is not None and int(self.n_select) <= 0:
                raise ValueError(
                    f"n_select must be a positive int or None, got {self.n_select!r}."
                )
            if isinstance(self.variable_selection, np.ndarray):
                # Precomputed boolean-mask hook (Phase-B gate): the C search layer
                # can wire ANY supervised method by computing the mask externally
                # (on fold-train) and passing it in.
                mask = np.asarray(self.variable_selection)
                if mask.dtype != bool or mask.shape != (n_features,):
                    raise ValueError(
                        "precomputed variable_selection must be a boolean array "
                        f"of shape ({n_features},), got dtype={mask.dtype} "
                        f"shape={mask.shape}."
                    )
                self.varsel_mask_ = mask.copy()
                self.varsel_path_ = "precomputed"
            elif self.variable_selection in (
                "wold_modeling",
                "wold_discriminating",
                "wold_balanced",
            ):
                # Wold needs an int n_components; "per_class_cv"/dict falls back to
                # 5 for the varsel PCA (warn — that is not each class's tuned nk).
                _varsel_nc = (
                    self.n_components
                    if isinstance(self.n_components, (int, np.integer))
                    else 5
                )
                if not isinstance(self.n_components, (int, np.integer)):
                    warnings.warn(
                        f"Wold varsel needs an int n_components; n_components="
                        f"{self.n_components!r} -> using {_varsel_nc} for the "
                        "varsel PCA (not each class's tuned value)."
                    )
                # Estimate Wold power on MODELABLE classes only (Phase-B gate:
                # Codex HIGH / Kimi M3) — a below-floor class cannot be PCA/KFold
                # modeled and would crash varsel; its decision-matrix column is
                # still preserved as unmodelable in the per-class loop below.
                X_vs, y_vs = self._varsel_modelable_subset(X, y, _varsel_nc + 1)
                wold_mode = self.variable_selection.split("_", 1)[1]
                self.varsel_mask_ = wold_variable_selection(
                    X_vs,
                    y_vs,
                    mode=wold_mode,
                    n_components=_varsel_nc,
                    n_select=self.n_select,
                    scaling=self.scaling,
                )
                self.varsel_path_ = "wold"
            elif self.variable_selection == "importance":
                from spectral_predict.unified_bayesian import compute_importances

                X_vs, y_vs = self._varsel_modelable_subset(X, y, 2)
                imp = np.asarray(
                    compute_importances(
                        X_vs,
                        y_vs,
                        "importance",
                        self.varsel_model_name,
                        task_type="classification",
                    ),
                    dtype=np.float64,
                )
                # Guard non-finite importances (Phase-B gate, Kimi #7) — a NaN
                # would make `imp >= mean(imp)` produce an arbitrary mask.
                if not np.all(np.isfinite(imp)):
                    raise ValueError(
                        "supervised variable_selection produced non-finite "
                        f"importances (varsel_model_name={self.varsel_model_name!r})."
                    )
                if isinstance(self.n_select, (int, np.integer)):
                    n_sel = int(self.n_select)
                    top_idx = np.argsort(imp, kind="stable")[-n_sel:]
                    mask = np.zeros(imp.shape[0], dtype=bool)
                    mask[top_idx] = True
                    self.varsel_mask_ = mask
                else:
                    self.varsel_mask_ = imp >= np.mean(imp)
                self.varsel_path_ = "supervised"
            else:
                raise NotImplementedError(
                    f"variable_selection={self.variable_selection!r} is not "
                    "wired at the model layer yet; the full supervised method "
                    "set is enumerated in the C search layer. Model-layer "
                    "supervised currently supports 'importance' (or pass a "
                    "precomputed boolean mask array)."
                )
            # Empty-mask guard (Phase-B gate, Kimi H1).
            if self.varsel_mask_ is not None and not self.varsel_mask_.any():
                raise ValueError(
                    "variable_selection produced an empty mask (no variable "
                    f"selected); check n_select={self.n_select!r}."
                )
        X_fit = X[:, self.varsel_mask_] if self.varsel_mask_ is not None else X

        # --- scaling setup (all scalers fit train-only inside fit) ------------
        # scalers_: {class_label: StandardScaler} populated only under per_class;
        # global_scaler_: single StandardScaler populated only under global.
        self.scalers_: dict = {}
        self.global_scaler_ = None
        if self.scaling == "global":
            self.global_scaler_ = StandardScaler().fit(X_fit)

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
            X_c = X_fit[y == c]
            n_class = X_c.shape[0]
            if n_class < self.min_class_samples:
                # Too few rows for a reliable MoM chi^2 fit / cross-fit null:
                # mark unmodelable, keep the decision-matrix column, fit no
                # scaler/engine (spec section 8).
                self.unmodelable_.add(c)
                continue

            # --- layered n_components-aware calibration floor (Phase-A fix 1) ---
            # Non-SIMCA engines: the empirical p-value floor is 1/(m+1) with
            # m ~= n_class; at m < 20 the floor exceeds alpha=0.05 so no sample
            # can ever be rejected -> such a class is unmodelable for these
            # engines (do not fit). SIMCA (DD-SIMCA): warn at small n but still
            # model, since DD-SIMCA over-rejects rather than failing to reject.
            if not is_simca and n_class < 20:
                self.unmodelable_.add(c)
                warnings.warn(
                    f"class {c}: n={n_class} < 20; non-SIMCA empirical p-value "
                    f"floor 1/(m+1) exceeds alpha={self.alpha} so no sample can "
                    f"be rejected; marking this class unmodelable."
                )
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
                        X_s = self.scalers_[c].transform(X_fit)
                    elif self.scaling == "global":
                        X_s = self.global_scaler_.transform(X_fit)
                    else:  # "none"
                        X_s = X_fit
                    y_oc = np.where(y == c, 1, -1)
                    n_components_c = self._tune_per_class_n_components(
                        X_s, y_oc, n_class, X_fit.shape[1]
                    )
                else:
                    n_components_c = self._n_components_for(c)
                self.n_components_[c] = n_components_c

                # DD-SIMCA small-n calibration warning (Phase-A fix 1): the MoM
                # chi^2 fit is high-variance at small n, so the model may
                # over-reject. Still model the class (do not mark unmodelable).
                warn_floor = max(20, 5 * n_components_c)
                if n_class < warn_floor:
                    warnings.warn(
                        f"class {c}: n={n_class} < {warn_floor}; DD-SIMCA "
                        f"calibration may over-reject at small n"
                    )

                self.models_[c] = PCASIMCA(
                    n_components=n_components_c, alpha=self.alpha
                ).fit(X_c_scaled)
            else:
                builder_name, score_method = _NON_SIMCA_ENGINES[self.engine]
                # Forward user-supplied engine hyperparameters (Phase-A fix 2).
                final_engine = get_one_class_model(
                    builder_name, **(self.engine_params or {})
                )
                final_engine.fit(X_c_scaled)
                self.models_[c] = final_engine
                # Cross-fit null on this class's rows so the empirical p-value
                # is a real level-alpha test (spec section 5.3). Per-fold
                # scaling for ``scaling="per_class"`` (Phase-A fix 6): pass the
                # RAW class rows and the scaling mode so each null fold fits a
                # fresh scaler on fold-train only (no leakage of the held-out
                # null-fold row into the scaler that scores it). For
                # ``scaling="global"`` reuse the already-fit global scaler; for
                # ``scaling="none"`` no scaling is applied.
                reuse_scaler = self.global_scaler_ if self.scaling == "global" else None
                self.nulls_[c] = self._cross_fit_null(
                    X_c, builder_name, score_method, self.scaling, reuse_scaler
                )

        # If every class was unmodelable, there is nothing to score with.
        # Raise rather than silently produce an all-NaN decision matrix
        # (Phase-A fix 1).
        if not self.models_:
            raise ValueError(
                "no class had enough samples to model; every class was below "
                "the calibration floor (min_class_samples="
                f"{self.min_class_samples}, or <20 for non-SIMCA engines)."
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
        if getattr(self, "varsel_mask_", None) is not None:
            X = X[:, self.varsel_mask_]
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
        # Phase-A fix 5: StratifiedKFold cannot split a singleton class. If any
        # class has fewer than 2 rows, raise a clear error rather than letting
        # sklearn emit an opaque "n_splits=1" message downstream.
        if min_class_count < 2:
            raise ValueError(
                f"cross_validate needs at least 2 samples per class to split "
                f"(StratifiedKFold); smallest class has {min_class_count}."
            )
        # If a class is smaller than the requested n_splits, OOF coverage is
        # reduced (clamped below); warn so the user knows.
        if min_class_count < n_splits:
            warnings.warn(
                f"cross_validate: smallest class has {min_class_count} samples "
                f"< requested n_splits={n_splits}; reducing to "
                f"{max(2, min(n_splits, min_class_count))} folds "
                f"(OOF coverage reduced)."
            )
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
                variable_selection=self.variable_selection,
                n_select=self.n_select,
                varsel_model_name=self.varsel_model_name,
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
                    variable_selection=self.variable_selection,
                    n_select=self.n_select,
                    varsel_model_name=self.varsel_model_name,
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
                variable_selection=self.variable_selection,
                n_select=self.n_select,
                varsel_model_name=self.varsel_model_name,
            )
            model.fit(X, y)
            preds = model.predict(external_X)
            return float(np.mean(preds == "novel"))

        raise ValueError(
            f"Unknown mode={mode!r}; expected 'loco' or 'external'."
        )

    def _varsel_modelable_subset(self, X, y, min_rows: int):
        """Rows of classes with >= ``min_rows`` samples, for varsel estimation.

        A below-floor class cannot be PCA/KFold-modeled, so Wold power estimation
        (and, for symmetry, the supervised importance model) is computed on the
        modelable classes only — its decision-matrix column is still preserved as
        unmodelable in :meth:`fit`'s per-class loop (Phase-B gate: Codex HIGH /
        Kimi M3). Raises if no class clears ``min_rows`` (the same degenerate case
        the per-class loop would otherwise raise on, surfaced earlier and
        clearer).
        """
        classes, counts = np.unique(y, return_counts=True)
        modelable = classes[counts >= min_rows]
        if modelable.size == 0:
            raise ValueError(
                "no class has enough samples for variable selection "
                f"(need >= {min_rows} rows per class)."
            )
        keep = np.isin(y, modelable)
        return X[keep], y[keep]

    def _n_components_for(self, label) -> int:
        """Resolve ``n_components`` for a single class label.

        ``n_components`` may be an int (same for all classes) or a
        ``{class_label: int}`` dict. ``label`` is a scalar from ``classes_``;
        numpy integer scalars hash-equal to Python ints, so dict lookup works
        regardless of whether the user keyed the dict with Python or numpy ints.
        """
        nc = self.n_components
        if isinstance(nc, dict):
            if label not in nc:
                raise ValueError(
                    f"n_components dict missing key {label!r} "
                    f"(have: {sorted(nc.keys())})"
                )
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
        self,
        X_raw: np.ndarray,
        builder_name: str,
        score_method: str,
        scaling: str,
        reuse_scaler=None,
    ) -> np.ndarray:
        """Build the cross-fit "normality" null array for one class (train-only).

        K-fold (5-fold, or fewer if the class is small) on the class's RAW
        training rows; for each fold, fit a fresh engine (built via
        :func:`~spectral_predict.contamination.get_one_class_model`, forwarding
        ``self.engine_params``) on the other folds and score the held-out fold
        with the pinned ``score_method``. Collects the out-of-fold "higher =
        more normal" scores as the empirical null (spec section 5.3 / task A4).
        Failed fold fits are silently skipped (spec).

        Per-fold scaling (Phase-A fix 6, leakage fix for ``scaling="per_class"``)
        is applied INSIDE each fold so a held-out null-fold row never influenced
        the scaler that scores it:

        - ``scaling="per_class"``: a fresh :class:`StandardScaler` is fit on the
          fold-train rows only, then used to transform BOTH fold-train and
          fold-test before the engine is fit / scores are computed.
        - ``scaling="global"``: the already-fit ``reuse_scaler`` (the train-level
          global scaler) is applied to both fold-train and fold-test.
        - ``scaling="none"``: rows are passed through unscaled.

        Parameters
        ----------
        X_raw : ndarray of shape (n_class, n_features)
            This class's RAW (unscaled) training rows.
        builder_name : str
            One-class model name understood by ``get_one_class_model``.
        score_method : str
            sklearn method name returning the engine's "higher = more normal"
            score (pinned per engine in :data:`_NON_SIMCA_ENGINES`).
        scaling : str
            Scaling mode (``"per_class"`` / ``"global"`` / ``"none"``).
        reuse_scaler : fitted StandardScaler, optional
            The already-fit global scaler, used only when ``scaling="global"``;
            ignored otherwise.

        Returns
        -------
        null : ndarray of shape (m,), sorted ascending
            Out-of-fold scores; ``m`` ~= ``n_class`` (fewer if folds failed).
        """
        n = X_raw.shape[0]
        n_splits = min(5, n)
        kf = KFold(n_splits=n_splits, shuffle=False)
        scores_list = []
        for train_idx, test_idx in kf.split(X_raw):
            X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
            try:
                if scaling == "per_class":
                    fold_scaler = StandardScaler().fit(X_tr)
                    X_tr_s = fold_scaler.transform(X_tr)
                    X_te_s = fold_scaler.transform(X_te)
                elif scaling == "global":
                    X_tr_s = reuse_scaler.transform(X_tr)
                    X_te_s = reuse_scaler.transform(X_te)
                else:  # "none"
                    X_tr_s, X_te_s = X_tr, X_te
                fold_engine = get_one_class_model(
                    builder_name, **(self.engine_params or {})
                )
                fold_engine.fit(X_tr_s)
                fold_scores = getattr(fold_engine, score_method)(X_te_s)
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


def multiclass_simca_metrics(y_true, A, classes) -> dict:
    """Dedicated multilabel / class-modeling metrics (spec section 7, task A7).

    Evaluates an acceptance matrix against ground-truth labels using the SIMCA
    class-modeling metric family (the belongs-to-none / one / several decision
    rule of Oliveri & Downey 2012). This is NOT the inverted
    ``one_class_metrics`` and NOT the single-label ``compute_imbalance_metrics``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels. Labels NOT in ``classes`` are treated as truly novel.
    A : ndarray of shape (n_samples, K), dtype bool
        Acceptance matrix; columns ordered by ``classes``. ``A[i, k]`` is True
        iff sample ``i`` is accepted by class ``classes[k]``.
    classes : list-like of length K
        Class labels defining the column order of ``A``.

    Returns
    -------
    dict
        Keys: ``per_class_sensitivity``, ``per_class_specificity``,
        ``novelty_detection_rate``, ``no_class_rate``, ``ambiguity_rate``,
        ``exact_set_rate``, ``efficiency``, ``pairwise_confusion``. Empty
        denominators yield ``float("nan")`` (never divides by zero).
    """
    y_true = np.asarray(y_true)
    A = np.asarray(A, dtype=bool)
    classes = list(classes)
    K = len(classes)
    n = A.shape[0]
    class_to_col = {c: k for k, c in enumerate(classes)}

    novel_mask = ~np.isin(y_true, classes)
    n_accepts = A.sum(axis=1)

    # --- per-class sensitivity / specificity -------------------------------
    per_class_sensitivity: dict = {}
    per_class_specificity: dict = {}
    for c in classes:
        k = class_to_col[c]
        is_k = y_true == c
        not_k = ~is_k
        denom_sens = int(np.sum(is_k))
        denom_spec = int(np.sum(not_k))
        if denom_sens == 0:
            per_class_sensitivity[c] = float("nan")
        else:
            per_class_sensitivity[c] = float(np.sum(is_k & A[:, k]) / denom_sens)
        if denom_spec == 0:
            per_class_specificity[c] = float("nan")
        else:
            per_class_specificity[c] = float(np.sum(not_k & ~A[:, k]) / denom_spec)

    # --- novelty detection rate (novel samples flagged as 0-accept) ---------
    n_novel = int(np.sum(novel_mask))
    if n_novel == 0:
        novelty_detection_rate = float("nan")
    else:
        novelty_detection_rate = float(
            np.sum(novel_mask & (n_accepts == 0)) / n_novel
        )

    # --- no-class / ambiguity / exact-set rates -----------------------------
    no_class_rate = float(np.sum(n_accepts == 0) / n) if n > 0 else float("nan")
    ambiguity_rate = float(np.sum(n_accepts >= 2) / n) if n > 0 else float("nan")

    if n == 0:
        exact_set_rate = float("nan")
    else:
        exact = np.zeros(n, dtype=bool)
        for i in range(n):
            accepted_cols = frozenset(np.where(A[i])[0].tolist())
            if novel_mask[i]:
                true_cols: frozenset = frozenset()
            else:
                true_cols = frozenset({class_to_col[y_true[i]]})
            exact[i] = accepted_cols == true_cols
        exact_set_rate = float(np.mean(exact))

    # --- efficiency: geomean of mean sensitivity / mean specificity --------
    # Phase-A fix 3: use nanmean so an absent class (NaN sensitivity) does not
    # collapse efficiency to NaN. Return NaN only if ALL sensitivities or ALL
    # specificities are NaN.
    sens_values = np.asarray(list(per_class_sensitivity.values()), dtype=np.float64)
    spec_values = np.asarray(list(per_class_specificity.values()), dtype=np.float64)
    if K == 0 or np.all(np.isnan(sens_values)) or np.all(np.isnan(spec_values)):
        efficiency = float("nan")
    else:
        mean_sens = float(np.nanmean(sens_values))
        mean_spec = float(np.nanmean(spec_values))
        if np.isfinite(mean_sens) and np.isfinite(mean_spec):
            efficiency = float(np.sqrt(mean_sens * mean_spec))
        else:
            efficiency = float("nan")

    # --- pairwise confusion: fraction of class-i accepted by class-j --------
    pairwise_confusion: dict = {}
    for i_cls in classes:
        for j_cls in classes:
            if i_cls == j_cls:
                continue
            col_j = class_to_col[j_cls]
            is_i = y_true == i_cls
            denom = int(np.sum(is_i))
            if denom == 0:
                pairwise_confusion[(i_cls, j_cls)] = float("nan")
            else:
                pairwise_confusion[(i_cls, j_cls)] = float(
                    np.sum(is_i & A[:, col_j]) / denom
                )

    return {
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "novelty_detection_rate": novelty_detection_rate,
        "no_class_rate": no_class_rate,
        "ambiguity_rate": ambiguity_rate,
        "exact_set_rate": exact_set_rate,
        "efficiency": efficiency,
        "pairwise_confusion": pairwise_confusion,
    }


def wilson_ci(k, n, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (task A7).

    Parameters
    ----------
    k : int or float
        Number of successes.
    n : int or float
        Number of trials.
    z : float, default=1.96
        Z-score for the confidence level (1.96 = 95% two-sided).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` bounds clamped to ``[0, 1]``. Returns
        ``(float("nan"), float("nan"))`` when ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def novelty_tradeoff_auc(y_true, P, classes) -> float:
    """AUC of the novelty-vs-false-rejection tradeoff curve (task A7).

    Sweeps the acceptance threshold ``alpha`` and integrates the tradeoff
    between detecting truly-novel samples and falsely rejecting known ones.

    For each threshold ``alpha``, acceptance is ``A(alpha) = P >= alpha``:

    - ``false_rejection_rate(alpha)`` = fraction of KNOWN samples whose
      own-class p-value falls below ``alpha`` (rejected by their own class).
    - ``novelty_rate(alpha)`` = fraction of NOVEL samples accepted by NO class
      (``P[j, :] < alpha`` for all columns).

    The ``(false_rejection_rate, novelty_rate)`` points are sorted by
    false-rejection rate and trapezoid-integrated; the AUC is clamped to
    ``[0, 1]``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels. Labels NOT in ``classes`` are truly novel.
    P : ndarray of shape (n_samples, K)
        Per-class p-values (columns ordered by ``classes``); acceptance at
        threshold ``alpha`` is ``P >= alpha``.
    classes : list-like of length K
        Class labels defining the column order of ``P``.

    Returns
    -------
    float
        AUC of ``novelty_rate`` over ``false_rejection_rate`` across the swept
        threshold set (``np.unique`` of ``P`` plus 0 and 1), clamped to
        ``[0, 1]``. Returns ``float("nan")`` if there are no known or no novel
        samples.
    """
    y_true = np.asarray(y_true)
    P = np.asarray(P, dtype=np.float64)
    classes = list(classes)
    class_to_col = {c: k for k, c in enumerate(classes)}

    known_mask = np.isin(y_true, classes)
    novel_mask = ~known_mask
    if not np.any(known_mask) or not np.any(novel_mask):
        return float("nan")

    known_idx = np.where(known_mask)[0]
    own_cols = np.array([class_to_col[y_true[i]] for i in known_idx], dtype=int)
    known_own_p = P[known_idx, own_cols]

    # Phase-A fix 4: exclude known samples whose own-class p-value is NaN
    # (e.g. their own class is unmodelable) from the false-rejection rate —
    # otherwise a NaN own-p would never count as a false rejection but would
    # still inflate the denominator.
    known_finite_mask = np.isfinite(known_own_p)
    known_own_p_finite = known_own_p[known_finite_mask]

    novel_idx = np.where(novel_mask)[0]
    novel_P = P[novel_idx]

    # Build the threshold sweep from FINITE p-values only so an all-NaN
    # (unmodelable) column does not pollute np.unique with a NaN threshold
    # (which would never compare correctly and collapse the AUC to 0).
    finite_p = P[np.isfinite(P)]
    thresholds = np.unique(np.concatenate([finite_p.ravel(), [0.0, 1.0]]))

    false_rej_rates = np.empty(thresholds.shape[0], dtype=np.float64)
    novelty_rates = np.empty(thresholds.shape[0], dtype=np.float64)
    for t, alpha in enumerate(thresholds):
        if known_own_p_finite.shape[0] == 0:
            false_rej_rates[t] = 0.0
        else:
            false_rej_rates[t] = float(np.mean(known_own_p_finite < alpha))
        # Treat NaN p-values as "never accept": a NaN is anomalous, so it
        # contributes True to the per-sample "rejected by every class" novelty
        # test. Without this an all-NaN unmodelable column forces all() to
        # False and collapses the novelty rate to 0.
        novelty_rates[t] = float(
            np.mean(np.all(np.isnan(novel_P) | (novel_P < alpha), axis=1))
        )

    order = np.argsort(false_rej_rates, kind="stable")
    fr = false_rej_rates[order]
    nr = novelty_rates[order]

    # numpy 2.0+ renamed trapz -> trapezoid; fall back for older versions.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(nr, fr))
    return float(min(1.0, max(0.0, auc)))


# ============================================================================
# Task B1: Wold (1976) variable selection — modeling power + discriminating
# power (spec §5.6). Class-modeling-native diagnostics on the genuine
# multi-class label (the one-class UVE/iPLS exclusion does NOT apply here).
# ============================================================================


def wold_modeling_power(X, n_components: int) -> np.ndarray:
    r"""Classical Wold (1976) per-variable modeling power for one class (B1).

    Fits ``PCA(n_components)`` on the rows of ``X`` (a single class),
    reconstructs, and returns the fraction of each variable's variance captured
    by the model:

    .. math:: \mathrm{MPOW}_j = 1 - \frac{s_{\mathrm{resid},j}}{s_{\mathrm{total},j}}

    where ``s_resid`` is the population (``ddof=0``) standard deviation of the
    PCA residual ``E = X - \hat{X}`` and ``s_total`` is the population standard
    deviation of the raw variable. Values near 1 mean the variable is well
    described by the PCA subspace; values near 0 mean mostly unmodeled noise.

    This is the CLASSICAL Wold 1976 per-variable modeling power and is DISTINCT
    from the DD-SIMCA per-variable :math:`T^2 / Q` decomposition used by
    :class:`~spectral_predict.contamination.PCASIMCA` (spec §5.6).

    .. note::
        ``ddof=0`` is used for both ``s_resid`` and ``s_total``. The residual has
        fewer effective degrees of freedom (``n - n_components - 1``) than the
        raw variable (``n - 1``), so the two do NOT fully cancel and MPOW is
        biased slightly UPWARD at small ``n`` / large ``n_components`` (~5 pp at
        n=30/J=2, larger below that). This affects the absolute MPOW level (a
        diagnostic), not — to first order — the variable RANKING that drives
        selection. A dof-corrected residual std is a deferred refinement (see
        the Phase-B gate notes / SESSION_LOG).

    Zero-variance columns (``s_total == 0``) return ``MPOW = 0.0`` (no variance
    to model) -- never NaN/inf (``np.errstate`` + mask guarded).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        ONE class's rows.
    n_components : int
        Requested PCA components; capped at ``min(n_samples - 1, n_features)``.

    Returns
    -------
    ndarray of shape (n_features,)
        Per-variable modeling power (constant columns return 0.0).
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    nc = max(1, min(int(n_components), n_samples - 1, n_features))
    pca = PCA(n_components=nc, random_state=0).fit(X)
    Xhat = pca.inverse_transform(pca.transform(X))
    E = X - Xhat
    s_resid = E.std(axis=0, ddof=0)
    s_total = X.std(axis=0, ddof=0)
    MPOW = np.zeros(n_features, dtype=np.float64)
    nz = s_total != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        MPOW[nz] = 1.0 - s_resid[nz] / s_total[nz]
    return MPOW


def _wold_cross_fit_own_rms(
    Xc_raw: np.ndarray,
    n_components: int,
    scaling: str,
    global_scaler,
) -> np.ndarray:
    """Cross-fit per-variable RMS residual (about zero) of one class on its own PCA.

    K-fold (``n_splits = min(5, n_class)``, no shuffle) over the class's RAW
    rows; each fold fits a fresh PCA on the fold-train rows (with per-fold
    train-only scaling under ``scaling="per_class"``) and reconstructs the
    fold-test rows. Out-of-fold residuals are collected into a full
    ``(n_class, n_features)`` array and reduced as
    ``sqrt(mean(resid**2, axis=0))`` -- the RMS ABOUT ZERO, i.e. the SIMCA
    "residual standard deviation" convention (RMS of residuals, dof aside), NOT
    the statistical ``std`` about the residual's own mean. This matters only for
    the DPOW cross term (own-model residuals are ~zero-mean by construction, so
    RMS == std there): when class-``c`` rows are scored on a foreign class-``j``
    model, the residual carries a constant between-class mean OFFSET per variable,
    which is exactly the discriminating signal. Taking ``std`` would subtract
    that offset out, leaving only within-class scatter (≈ the own-model residual)
    so the cross/own ratio collapses toward 1 — empirically verified on
    mean-separated synthetic data (Spearman rank-stability fell to ~0 with
    ``std`` vs ~0.93 with RMS). (Phase-B gate note: this is the intended
    resolution of the std-vs-RMS ambiguity in the classical Wold formula; it is
    surfaced to the user as a methodology decision.)

    Parameters
    ----------
    Xc_raw : ndarray of shape (n_class, n_features)
        ONE class's RAW (unscaled) rows.
    n_components : int
        Requested PCA components (capped per fold at
        ``min(n_train - 1, n_features)``).
    scaling : {"none", "per_class", "global"}
        Scaling mode. ``"per_class"`` fits a fresh :class:`StandardScaler` on
        each fold-train only (no leakage of the held-out fold row into the
        scaler that scores it); ``"global"`` reuses the already-fit
        ``global_scaler``; ``"none"`` passes rows through unscaled.
    global_scaler : fitted StandardScaler or None
        The train-level global scaler, used only when ``scaling="global"``.

    Returns
    -------
    ndarray of shape (n_features,)
        Per-variable cross-fit RMS residual about zero.
    """
    n, n_features = Xc_raw.shape
    # A class with < 2 rows cannot be K-fold cross-fit (KFold needs n_splits >= 2)
    # nor PCA-modeled; return a zero RMS (its DPOW ratio then guards to 0). The
    # fit-path modelable filter normally prevents this; this guards direct calls
    # (Phase-B gate: Kimi M3).
    if n < 2:
        return np.zeros(n_features, dtype=np.float64)
    n_splits = min(5, n)
    kf = KFold(n_splits=n_splits, shuffle=False)
    residuals = np.zeros((n, n_features), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    for train_idx, test_idx in kf.split(Xc_raw):
        X_tr_raw = Xc_raw[train_idx]
        X_te_raw = Xc_raw[test_idx]
        if scaling == "per_class":
            fold_scaler = StandardScaler().fit(X_tr_raw)
            X_tr = fold_scaler.transform(X_tr_raw)
            X_te = fold_scaler.transform(X_te_raw)
        elif scaling == "global":
            X_tr = global_scaler.transform(X_tr_raw)
            X_te = global_scaler.transform(X_te_raw)
        else:  # "none"
            X_tr = X_tr_raw
            X_te = X_te_raw
        n_tr = X_tr.shape[0]
        nc_fold = max(1, min(int(n_components), n_tr - 1, n_features))
        pca = PCA(n_components=nc_fold, random_state=0).fit(X_tr)
        recon = pca.inverse_transform(pca.transform(X_te))
        residuals[test_idx] = X_te - recon
        filled[test_idx] = True
    res = residuals[filled] if not filled.all() else residuals
    return np.sqrt(np.mean(res**2, axis=0))


def wold_variable_powers(
    X, y, n_components: int = 5, scaling: str = "none"
) -> dict:
    r"""Wold per-variable modeling power + discriminating power (B1, spec §5.6).

    Computes classical Wold (1976) modeling power (MPOW) and one-vs-rest
    discriminating power (DPOW) for every variable, per class and macro-averaged
    across classes. Both are class-modeling-native diagnostics computed on the
    genuine multi-class label (the one-class UVE/iPLS exclusion does NOT apply
    here).

    Discriminating power (Wold): for each class ``c`` and variable ``v``,

    .. math:: \mathrm{DPOW}_{c,v} = \frac{1}{K-1}\sum_{j \neq c}
        \frac{\mathrm{cross\_rms}_{c\to j,v}}{\mathrm{own\_rms}_{c,v}}

    where ``own_rms`` is the CROSS-FIT RMS residual of class ``c`` on its own PCA
    model (K-fold out-of-fold, about zero -- see :func:`_wold_cross_fit_own_rms`),
    and ``cross_rms[c->j]`` is the RMS residual of class-``c`` rows reconstructed
    by class-``j`` 's PCA model. The RMS is taken ABOUT ZERO
    (``sqrt(mean(resid**2))``) -- NOT ``std`` -- so the between-class mean offset
    survives (see :func:`_wold_cross_fit_own_rms` for the full rationale).

    **Aggregation and asymmetry (Phase-B gate — documented per spec §5.6):**

    - **K>2 = one-vs-rest, non-symmetric.** ``DPOW[c]`` is class ``c``'s residual
      inflation averaged over the ``K-1`` foreign models; ``DPOW[c]`` and
      ``DPOW[j]`` for a pair ``(c, j)`` are computed independently and are NOT
      symmetrized. The reported per-class DPOW is therefore "how class ``c`` is
      distinguished from the rest", and the macro-average across classes weights
      each ordered pair once. A symmetric variant (geomean of the two directions)
      is a possible future option.
    - **own = cross-fit, cross = full model — deliberate.** ``own_rms`` is
      out-of-fold (cross-fit) while ``cross_rms`` uses class ``j``'s full PCA. This
      removes the LARGE in-sample optimism that a full own-model would give (own
      residuals on the exact rows the PCA was fit on are artificially tiny, which
      would inflate every ratio); the residual ``(n-fold)/n`` sample-size
      asymmetry is small by comparison. This matches the spec §5.6 formula (own =
      cross-fit) and is NOT the classical in-sample Wold ratio.
    - Where ``own_rms == 0`` (a variable perfectly modeled within class ``c``) the
      ratio is set to 0.0 to avoid ``inf``; a genuinely discriminating variable
      that is also perfectly modeled within-class is therefore under-reported in
      that degenerate case (rare; documented). Non-finite values also map to 0.0.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Class labels.
    n_components : int, default=5
        PCA components per class (capped at ``min(n_class - 1, n_features)``).
    scaling : {"none", "per_class", "global"}, default="none"
        Column-scaling mode. All scalers are fit train-only. ``"per_class"``
        fits one :class:`StandardScaler` per class on that class's rows; the
        DPOW cross term transforms class-``c`` rows by class-``j`` 's scaler
        before scoring on model_j (mirroring
        :meth:`MultiClassClassModel.decision_matrix`). ``"global"`` fits one
        scaler across all rows. ``"none"`` passes ``X`` through unchanged.

    Returns
    -------
    dict
        Keys:

        - ``"modeling_power"``: ``(n_features,)`` macro-average of per-class MPOW.
        - ``"discriminating_power"``: ``(n_features,)`` macro-average of
          per-class DPOW.
        - ``"modeling_power_per_class"``: ``{class_label: (n_features,) ndarray}``.
        - ``"discriminating_power_per_class"``:
          ``{class_label: (n_features,) ndarray}``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]

    # --- scaling setup (all scalers fit train-only) -----------------------
    if scaling == "global":
        global_scaler = StandardScaler().fit(X)
    else:
        global_scaler = None
    per_class_scalers: dict = {}
    if scaling == "per_class":
        for c in classes:
            per_class_scalers[c] = StandardScaler().fit(X[y == c])

    def _to_class_space(Xc, c):
        if scaling == "none":
            return Xc
        if scaling == "per_class":
            return per_class_scalers[c].transform(Xc)
        return global_scaler.transform(Xc)

    def _to_other_space(Xc, j):
        # Mirrors MultiClassClassModel.decision_matrix: a sample scored against
        # class j is transformed by class j's scaler (per_class) / the global
        # scaler (global) / raw (none) before scoring on model_j.
        if scaling == "none":
            return Xc
        if scaling == "per_class":
            return per_class_scalers[j].transform(Xc)
        return global_scaler.transform(Xc)

    # --- per-class MPOW + full PCA models (models feed the DPOW cross term) -
    modeling_power_per_class: dict = {}
    models: dict = {}
    for c in classes:
        Xc = X[y == c]
        Xc_s = _to_class_space(Xc, c)
        nc = max(1, min(int(n_components), Xc.shape[0] - 1, n_features))
        models[c] = PCA(n_components=nc, random_state=0).fit(Xc_s)
        modeling_power_per_class[c] = wold_modeling_power(Xc_s, n_components)

    # --- per-class DPOW (one-vs-rest cross residual-RMS ratio) -------------
    discriminating_power_per_class: dict = {}
    for c in classes:
        Xc = X[y == c]
        own_rms = _wold_cross_fit_own_rms(
            Xc, n_components, scaling, global_scaler
        )
        ratios = []
        for j in classes:
            if j == c:
                continue
            Xc_in_j = _to_other_space(Xc, j)
            model_j = models[j]
            recon_j = model_j.inverse_transform(model_j.transform(Xc_in_j))
            cross_rms = np.sqrt(np.mean((Xc_in_j - recon_j) ** 2, axis=0))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(own_rms == 0.0, 0.0, cross_rms / own_rms)
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            ratios.append(ratio)
        if ratios:
            discriminating_power_per_class[c] = np.mean(np.vstack(ratios), axis=0)
        else:
            discriminating_power_per_class[c] = np.zeros(
                n_features, dtype=np.float64
            )

    modeling_power = np.mean(
        np.vstack([modeling_power_per_class[c] for c in classes]), axis=0
    )
    discriminating_power = np.mean(
        np.vstack([discriminating_power_per_class[c] for c in classes]), axis=0
    )

    return {
        "modeling_power": modeling_power,
        "discriminating_power": discriminating_power,
        "modeling_power_per_class": modeling_power_per_class,
        "discriminating_power_per_class": discriminating_power_per_class,
    }


def _minmax_unit(a: np.ndarray) -> np.ndarray:
    """Min-max scale an array to [0, 1]; a constant array maps to all zeros."""
    a = np.asarray(a, dtype=np.float64)
    lo = np.min(a)
    hi = np.max(a)
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def wold_variable_selection(
    X,
    y,
    mode: str = "balanced",
    n_components: int = 5,
    n_select: int | None = None,
    scaling: str = "none",
) -> np.ndarray:
    """Wold modeling/discriminating-power variable selection (B1, spec §5.6).

    Scores every variable by the chosen ``mode`` and keeps the top variables.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Class labels.
    mode : {"modeling", "discriminating", "balanced"}, default="balanced"
        Scoring rule: ``"modeling"`` -> MPOW; ``"discriminating"`` -> DPOW;
        ``"balanced"`` -> ``minmax(MPOW) * minmax(DPOW)`` (both min-max scaled to
        [0, 1] before multiplying, so the unbounded DPOW scale cannot dominate).
    n_components : int, default=5
        Forwarded to :func:`wold_variable_powers`.
    n_select : int or None, default=None
        If an int, keep exactly the top-``n_select`` variables by score (ties
        broken toward the lower variable index via a stable sort on the
        negated score). If ``None``, keep every variable whose score is
        ``>= mean(score)``.
    scaling : {"none", "per_class", "global"}, default="none"
        Forwarded to :func:`wold_variable_powers`.

    Returns
    -------
    ndarray of shape (n_features,), dtype bool
        Selection mask.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``"modeling"`` / ``"discriminating"`` /
        ``"balanced"``.
    """
    if mode not in ("modeling", "discriminating", "balanced"):
        raise ValueError(
            f"Unknown mode={mode!r}; expected 'modeling', 'discriminating', "
            f"or 'balanced'."
        )
    if n_select is not None and int(n_select) <= 0:
        raise ValueError(
            f"n_select must be a positive int or None, got {n_select!r}."
        )
    powers = wold_variable_powers(X, y, n_components=n_components, scaling=scaling)
    if mode == "modeling":
        score = np.asarray(powers["modeling_power"], dtype=np.float64)
    elif mode == "discriminating":
        score = np.asarray(powers["discriminating_power"], dtype=np.float64)
    else:  # balanced
        # Min-max normalize MPOW and (unbounded) DPOW to [0, 1] BEFORE multiplying
        # (Phase-B gate MiniMax M2, user-approved), so the DPOW scale cannot
        # dominate the product; a constant array normalizes to zeros.
        score = _minmax_unit(
            np.asarray(powers["modeling_power"], dtype=np.float64)
        ) * _minmax_unit(
            np.asarray(powers["discriminating_power"], dtype=np.float64)
        )
    n_features = score.shape[0]
    if n_select is not None:
        mask = np.zeros(n_features, dtype=bool)
        # Stable argsort on -score -> descending score, ties keep lower index.
        order = np.argsort(-score, kind="stable")
        keep = order[: int(n_select)]
        mask[keep] = True
        return mask
    return score >= np.mean(score)


def wold_diagnostic_plot_data(
    X, y, n_components: int = 5, scaling: str = "none", wavelengths=None
) -> dict:
    """Package Wold MPOW/DPOW arrays for the Phase-D diagnostic plots (B2).

    Thin presentation layer over :func:`wold_variable_powers`: stacks the
    per-class modeling / discriminating power into ``(K, n_features)`` arrays
    whose row ``k`` corresponds to ``classes[k]`` (``classes = np.unique(y)``),
    plus the macro-averaged aggregates and an x-axis for the GUI.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Class labels.
    n_components : int, default=5
        Forwarded to :func:`wold_variable_powers`.
    scaling : {"none", "per_class", "global"}, default="none"
        Forwarded to :func:`wold_variable_powers`.
    wavelengths : array-like of shape (n_features,), optional
        x-axis values for the plot. Defaults to ``np.arange(n_features)``.

    Returns
    -------
    dict
        Keys:

        - ``"classes"``: ``(K,)`` class labels (column/row order).
        - ``"variables"``: ``(n_features,)`` plot x-axis (wavelengths or indices).
        - ``"modeling_power"`` / ``"discriminating_power"``: ``(K, n_features)``
          arrays; row ``k`` is class ``classes[k]``.
        - ``"modeling_power_agg"`` / ``"discriminating_power_agg"``:
          ``(n_features,)`` macro-averaged aggregates.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_features = X.shape[1]
    classes = np.unique(y)

    powers = wold_variable_powers(
        X, y, n_components=n_components, scaling=scaling
    )
    modeling = np.vstack(
        [powers["modeling_power_per_class"][c] for c in classes]
    )
    discriminating = np.vstack(
        [powers["discriminating_power_per_class"][c] for c in classes]
    )

    if wavelengths is None:
        variables = np.arange(n_features)
    else:
        variables = np.asarray(wavelengths)

    return {
        "classes": classes,
        "variables": variables,
        "modeling_power": modeling,
        "discriminating_power": discriminating,
        "modeling_power_agg": powers["modeling_power"],
        "discriminating_power_agg": powers["discriminating_power"],
    }
