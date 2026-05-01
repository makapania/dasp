"""T-36: save -> load roundtrip tests for autoscale-trained models.

Verifies that .dasp files written from autoscale=True training produce
identical predictions when reloaded — i.e. the pickled preprocessor.pkl
correctly carries the fitted StandardScaler step, and the metadata.json
correctly carries the autoscale flag.

These are the regressions Codex round-2 review (commit 2351c3c) called
out: prediction roundtrip is safe via the pickled preprocessor, but the
metadata-only path needs the explicit flag for downstream visibility.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

from spectral_predict.model_io import save_model, load_model
from spectral_predict.preprocess import build_preprocessing_pipeline


@pytest.fixture
def synthetic_regression_data():
    """Tiny synthetic spectral regression dataset (deterministic)."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 50, 30
    latent = rng.normal(size=n_samples)
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    X = np.outer(latent, base) + rng.normal(scale=0.1, size=(n_samples, n_features))
    y = 2.0 * latent + rng.normal(scale=0.1, size=n_samples)
    return X, y


def _train_autoscaled_pls(X, y, autoscale: bool):
    """Fit a snv preprocessing pipeline (with or without autoscale) + PLS regressor.
    Returns (preprocessor_pipeline, model, X_preprocessed)."""
    steps = build_preprocessing_pipeline('snv', autoscale=autoscale)
    preprocessor = Pipeline(steps)
    X_pre = preprocessor.fit_transform(X)
    model = PLSRegression(n_components=3)
    model.fit(X_pre, y)
    return preprocessor, model, X_pre


def _make_metadata(autoscale: bool, n_features: int) -> dict:
    return {
        'model_name': 'PLS',
        'task_type': 'regression',
        'preprocessing': 'snv',
        'wavelengths': [str(i) for i in range(n_features)],
        'n_vars': n_features,
        'n_samples': 50,
        'performance': {'R2': 0.9, 'RMSE': 0.1},
        'window': 17,
        'autoscale': autoscale,
    }


class TestAutoscaleSaveLoadRoundtrip:
    def test_autoscale_true_metadata_roundtrips(self, synthetic_regression_data, tmp_path):
        """metadata['autoscale'] == True must survive save -> load."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=True)
        filepath = tmp_path / 'autoscale_true.dasp'
        save_model(
            model=model,
            preprocessor=preprocessor,
            metadata=_make_metadata(autoscale=True, n_features=X.shape[1]),
            filepath=filepath,
        )
        loaded = load_model(filepath)
        assert loaded['metadata']['autoscale'] is True

    def test_autoscale_false_metadata_roundtrips(self, synthetic_regression_data, tmp_path):
        """metadata['autoscale'] == False must survive save -> load."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=False)
        filepath = tmp_path / 'autoscale_false.dasp'
        save_model(
            model=model,
            preprocessor=preprocessor,
            metadata=_make_metadata(autoscale=False, n_features=X.shape[1]),
            filepath=filepath,
        )
        loaded = load_model(filepath)
        assert loaded['metadata']['autoscale'] is False

    def test_autoscale_true_preprocessor_carries_standardscaler_step(
        self, synthetic_regression_data, tmp_path
    ):
        """The pickled preprocessor.pkl must include a StandardScaler step
        (named 'autoscale' per build_preprocessing_pipeline) when autoscale=True
        was used at training time."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=True)
        filepath = tmp_path / 'autoscale_pipe.dasp'
        save_model(
            model=model, preprocessor=preprocessor,
            metadata=_make_metadata(autoscale=True, n_features=X.shape[1]),
            filepath=filepath,
        )
        loaded = load_model(filepath)
        loaded_pre = loaded['preprocessor']
        assert loaded_pre is not None
        step_names = [name for name, _ in loaded_pre.steps]
        assert 'autoscale' in step_names, (
            f"Loaded preprocessor missing 'autoscale' step, got {step_names}"
        )

    def test_autoscale_false_preprocessor_omits_standardscaler_step(
        self, synthetic_regression_data, tmp_path
    ):
        """When autoscale=False at training, the saved preprocessor must NOT
        have an 'autoscale' step (otherwise predictions diverge from training)."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=False)
        filepath = tmp_path / 'no_autoscale_pipe.dasp'
        save_model(
            model=model, preprocessor=preprocessor,
            metadata=_make_metadata(autoscale=False, n_features=X.shape[1]),
            filepath=filepath,
        )
        loaded = load_model(filepath)
        loaded_pre = loaded['preprocessor']
        assert loaded_pre is not None
        step_names = [name for name, _ in loaded_pre.steps]
        assert 'autoscale' not in step_names, (
            f"Loaded preprocessor unexpectedly has 'autoscale' step: {step_names}"
        )

    def test_predictions_bit_identical_after_roundtrip_with_autoscale(
        self, synthetic_regression_data, tmp_path
    ):
        """The full predict roundtrip — preprocessor.transform then model.predict
        — must produce IDENTICAL output before and after save -> load when
        autoscale=True. This is the prediction-correctness guarantee."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=True)
        # Original prediction path
        X_pre_orig = preprocessor.transform(X)
        y_pred_orig = model.predict(X_pre_orig).ravel()

        filepath = tmp_path / 'predict_roundtrip.dasp'
        save_model(
            model=model, preprocessor=preprocessor,
            metadata=_make_metadata(autoscale=True, n_features=X.shape[1]),
            filepath=filepath,
        )
        loaded = load_model(filepath)

        # Loaded prediction path
        X_pre_loaded = loaded['preprocessor'].transform(X)
        y_pred_loaded = loaded['model'].predict(X_pre_loaded).ravel()

        np.testing.assert_array_equal(X_pre_orig, X_pre_loaded), (
            "Loaded preprocessor.transform must produce identical output to original"
        )
        np.testing.assert_array_equal(y_pred_orig, y_pred_loaded), (
            "Loaded predictions must be identical to original"
        )

    def test_autoscale_default_when_metadata_missing_field(
        self, synthetic_regression_data, tmp_path
    ):
        """Old .dasp files (saved before T-36) lack the 'autoscale' metadata
        field. Loading them must NOT crash; downstream readers should treat
        the absent field as False."""
        X, y = synthetic_regression_data
        preprocessor, model, _ = _train_autoscaled_pls(X, y, autoscale=False)
        # Strip the autoscale field to simulate a pre-T-36 .dasp file.
        metadata = _make_metadata(autoscale=False, n_features=X.shape[1])
        del metadata['autoscale']
        filepath = tmp_path / 'legacy_no_autoscale_field.dasp'
        save_model(model=model, preprocessor=preprocessor, metadata=metadata, filepath=filepath)
        loaded = load_model(filepath)
        # Field is genuinely absent — readers must handle this gracefully.
        assert 'autoscale' not in loaded['metadata']
        # The pickled preprocessor still works for prediction.
        X_pre = loaded['preprocessor'].transform(X)
        y_pred = loaded['model'].predict(X_pre).ravel()
        assert y_pred.shape == y.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
