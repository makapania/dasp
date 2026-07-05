"""Multi-class class modeling (T-31).

Implements :class:`MultiClassClassModel` — K independent per-class membership
models (SIMCA flagship) evaluated at a *global* level ``alpha``, producing a
per-sample x per-class decision matrix and a single / multiple / novel summary
label.

This is task **A2 + A3** of the T-31 build: the core orchestrator with the
SIMCA (``"pca-simca"``) engine and three column-scaling modes
(``scaling="per_class"`` is the textbook default). Non-SIMCA engines (A4),
per-class CV ``n_components`` tuning (A5), novelty evaluation (A6), dedicated
metrics (A7), and ``.dasp`` persistence (A8) arrive in subsequent tasks.

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
from sklearn.preprocessing import StandardScaler

from spectral_predict.contamination import PCASIMCA


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
        Per-class membership engine. A2 implements only ``"pca-simca"``
        (DD-SIMCA via :class:`~spectral_predict.contamination.PCASIMCA`);
        OCSVM / IF / LOF / EE engines arrive in A4.
    alpha : float, default=0.05
        Global significance level shared by every per-class model, so the
        ">= 2 of K" multiple-membership rule is coherent across classes
        (spec decision #6).
    n_components : int or dict, default=5
        PCA components for the ``"pca-simca"`` engine — either one int applied
        to every class or a ``{class_label: int}`` mapping. The string
        ``"per_class_cv"`` (per-class CV tuning) lands in A5.
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
        n_components: int | dict | str = 5,
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

        if self.n_components == "per_class_cv":
            raise NotImplementedError(
                'n_components="per_class_cv" (per-class CV tuning) lands in task A5 '
                "of T-31; pass an int or a {class_label: int} dict for now."
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

        for c in self.classes_:
            X_c = X[y == c]
            if X_c.shape[0] < self.min_class_samples:
                # Too few rows for a reliable MoM chi^2 fit: mark unmodelable,
                # keep the decision-matrix column, fit no scaler/engine (spec §8).
                self.unmodelable_.add(c)
                continue

            # Apply this class's scaling mode (train-only fit) before PCA.
            if self.scaling == "per_class":
                self.scalers_[c] = StandardScaler().fit(X_c)
                X_c_scaled = self.scalers_[c].transform(X_c)
            elif self.scaling == "global":
                X_c_scaled = self.global_scaler_.transform(X_c)
            else:  # "none" — X passes through unchanged
                X_c_scaled = X_c

            n_components_c = self._n_components_for(c)
            if self.engine == "pca-simca":
                self.models_[c] = PCASIMCA(
                    n_components=n_components_c, alpha=self.alpha
                ).fit(X_c_scaled)
            else:
                raise NotImplementedError(
                    f"engine={self.engine!r} is implemented in task A4 of T-31; "
                    'only engine="pca-simca" is available in A2.'
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
            P[:, k] = self.models_[c].p_joint(X_eval)
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
