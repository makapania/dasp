"""
Regression test for PLS-DA wavelength importance (VIP) extraction.

Verifies that get_feature_importances() correctly unwraps a PLS-DA pipeline
[pls, scaler, lr] and computes VIP scores from the PLS component, not from
the LogisticRegression step.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict.models import PLSTransformer, get_feature_importances


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=80,
        n_features=30,
        n_informative=10,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def fitted_plsda_pipeline(classification_data):
    X, y = classification_data
    pipe = Pipeline(
        [
            ("pls", PLSTransformer(n_components=5, scale=False)),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe, X, y


class TestPLSDAFeatureImportances:
    def test_importance_from_pipeline(self, fitted_plsda_pipeline):
        pipe, X, y = fitted_plsda_pipeline
        importances = get_feature_importances(pipe, "PLS-DA", X, y)
        assert importances is not None
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)

    def test_importance_matches_pls_component(self, fitted_plsda_pipeline):
        pipe, X, y = fitted_plsda_pipeline
        importances_pipeline = get_feature_importances(pipe, "PLS-DA", X, y)

        from spectral_predict.models import compute_vip

        pls_step = pipe.named_steps["pls"]
        importances_direct = compute_vip(pls_step, X, y)

        np.testing.assert_array_almost_equal(
            importances_pipeline,
            importances_direct,
            decimal=10,
        )

    def test_importance_not_from_lr(self, fitted_plsda_pipeline):
        pipe, X, y = fitted_plsda_pipeline
        importances = get_feature_importances(pipe, "PLS-DA", X, y)
        assert (
            len(importances) == X.shape[1]
        ), "VIP importances should be in original feature space, not PLS-score space"

    def test_importance_from_bare_pls_transformer(self, classification_data):
        X, y = classification_data
        pls = PLSTransformer(n_components=5, scale=False)
        pls.fit(X, y)
        importances = get_feature_importances(pls, "PLS-DA", X, y)
        assert importances is not None
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)

    def test_pipeline_with_scaler_model_steps(self, classification_data):
        X, y = classification_data
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", PLSTransformer(n_components=5, scale=False)),
            ]
        )
        pipe.fit(X, y)
        importances = get_feature_importances(pipe, "PLS-DA", X, y)
        assert importances is not None
        assert len(importances) == X.shape[1]
