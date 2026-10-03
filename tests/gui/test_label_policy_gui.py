"""GUI side of the classification label policy (review round 4).

Integer-valued numeric labels are fitted as given in every search, so the GUI
must (a) hand the validation rebuild the labels the models were fitted on and
(b) never decode raw numeric per-class keys through an encoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui_module


def _data(labels, n_per_class=(20, 18, 16), seed=5, p=30):
    rng = np.random.default_rng(seed)
    codes = np.concatenate([np.full(k, c) for c, k in enumerate(n_per_class[: len(labels)])])
    X = rng.normal(size=(len(codes), p))
    X[:, :4] += 0.8 * codes[:, None]
    X[:, 4] -= 0.7 * (codes == 1)
    return X, np.array(
        [labels[c] for c in codes], dtype=object if isinstance(labels[0], str) else None
    )


def _wavelengths(X):
    return np.arange(1000.0, 1000.0 + 2 * X.shape[1], 2.0)


def _legend_values_text(app):
    texts = []

    def walk(widget):
        for child in widget.winfo_children():
            try:
                texts.append(str(child.cget("text")))
            except Exception:  # noqa: BLE001 - frames/canvases have no text
                pass
            walk(child)

    walk(app.region_legend_frame)
    return next(t for t in texts if t.startswith("Values:"))


@pytest.fixture
def legend_state(gui_app):
    saved = (getattr(gui_app, "_class_rankings", None), gui_app.label_encoder)
    yield gui_app
    gui_app._class_rankings, gui_app.label_encoder = saved
    for widget in gui_app.region_legend_frame.winfo_children():
        widget.destroy()


def test_numeric_class_legend_shows_raw_labels(legend_state):
    app = legend_state
    app._class_rankings = {"class_labels": ["1", "2", "100"]}
    app.label_encoder = gui_module._display_label_encoder(np.array([1, 2, 100, 2]))
    assert app.label_encoder is None
    app._update_class_legend()
    assert _legend_values_text(app) == "Values: C0=1, C1=2, C2=100"


def test_numeric_legend_ignores_a_stale_encoder(legend_state):
    from sklearn.preprocessing import LabelEncoder

    app = legend_state
    app._class_rankings = {"class_labels": ["1", "2", "100"]}
    app.label_encoder = LabelEncoder().fit([1, 2, 100])  # pre-policy Bayesian encoder
    app._update_class_legend()
    # was "Values: C1=2, C1=100, C2=100"
    assert _legend_values_text(app) == "Values: C0=1, C1=2, C2=100"


def test_text_class_legend_decodes_codes(legend_state):
    app = legend_state
    app._class_rankings = {"class_labels": ["0", "1"]}
    app.label_encoder = gui_module._display_label_encoder(np.array(["clean", "dirty", "clean"]))
    app._update_class_legend()
    assert _legend_values_text(app) == "Values: C0=clean, C1=dirty"


def test_fractional_labels_keep_a_display_encoder():
    enc = gui_module._display_label_encoder(np.array([0.1, 0.2, 0.1]))
    assert enc is not None and list(enc.classes_) == [0.1, 0.2]


def test_bayesian_holdout_rebuild_matches_the_search_row():
    """The Bayesian holdout block's label handling reproduces the search fit."""
    from sklearn.preprocessing import LabelEncoder

    from spectral_predict.search import compute_validation_metrics_for_top_models
    from spectral_predict.unified_bayesian import run_unified_bayesian

    X, y = _data((1, 2, 100))
    df, _ = run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=_wavelengths(X),
        model_name="PLS-DA",
        task_type="classification",
        n_trials=3,
        cv_folds=3,
        n_top_regions=2,
        enable_sqlite_persistence="never",
        verbose=False,
    )
    # what the GUI block does: map onto training classes with a temporary
    # encoder, then restore the labels the models were fitted on
    temp = LabelEncoder().fit(y)
    y_train, y_val = gui_module._holdout_labels_as_fitted(
        temp, temp.transform(y), temp.transform(y), y
    )
    np.testing.assert_array_equal(y_train, y)
    out = compute_validation_metrics_for_top_models(
        df.copy(), X, y_train, X, y_val, "classification", _wavelengths(X), top_n=len(df)
    )
    rows = out.dropna(subset=["val_Accuracy"])
    assert len(rows) > 0
    for _, r in rows.iterrows():
        assert r["val_Accuracy"] == pytest.approx(r["Accuracy"])


def test_nsga2_text_holdout_encodes_training_labels_too():
    from spectral_predict.nsga2_search import convert_nsga2_to_v1_format, run_nsga2_search
    from spectral_predict.search import compute_validation_metrics_for_top_models

    X, y = _data(("clean", "dirty"))
    result = run_nsga2_search(
        X=X,
        y=y,
        task_type="classification",
        population_size=6,
        n_generations=2,
        cv_folds=3,
        min_wavelengths=10,
        verbose=0,
        models=["PLS-DA"],
    )
    enc = result["label_encoder"]
    assert enc is not None
    df = convert_nsga2_to_v1_format(
        result=result,
        n_wavelengths=X.shape[1],
        task_type="classification",
        folds=3,
        wavelengths=_wavelengths(X),
        X=X,
        y=y,
        compute_r2=False,
    )
    y_train, y_val = gui_module._encode_holdout_pair(enc, y, y)
    np.testing.assert_array_equal(y_train, enc.transform(y))
    out = compute_validation_metrics_for_top_models(
        df.copy(), X, y_train, X, y_val, "classification", _wavelengths(X), top_n=len(df)
    )
    rows = out.dropna(subset=["val_Accuracy"])
    assert len(rows) > 0
    # was 0 / NaN when y_train stayed raw text against coded y_val
    assert (rows["val_Accuracy"] > 0.6).all()
    assert np.isfinite(pd.to_numeric(rows["val_ROC_AUC"])).all()


def test_nsga2_holdout_drops_unknown_validation_labels():
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder().fit(["clean", "dirty"])
    y_train = np.array(["clean", "dirty", "clean"], dtype=object)
    y_val = np.array(["clean", "zzz", "dirty", "zzz"], dtype=object)
    tr, va, keep = gui_module._encode_holdout_pair(enc, y_train, y_val, drop_unknown=True)
    np.testing.assert_array_equal(keep, [True, False, True, False])
    np.testing.assert_array_equal(tr, [0, 1, 0])
    np.testing.assert_array_equal(va, [0, 1])


# ---------------------------------------------------------------------------
# Real Bayesian classification through the GUI worker (review round 5)
# ---------------------------------------------------------------------------

_WORKER_VARS = (
    "optimization_method",
    "task_type",
    "n_unified_trials",
    "folds",
    "output_dir",
    "bayesian_persistence_mode",
)


@pytest.fixture
def bayes_worker(gui_app, tmp_path, monkeypatch, reimport_modules):
    import sys

    from spectral_predict.search_controller import SearchController

    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()

    def run_now(_ms, func=None, *args):
        if func is not None:
            func(*args)

    monkeypatch.setattr(gui_app.root, "after", run_now)
    saved = {name: getattr(gui_app, name).get() for name in _WORKER_VARS}
    saved_state = (
        gui_app.active_indices,
        gui_app.excluded_spectra,
        gui_app.X,
        gui_app.y,
        gui_app.label_encoder,
        getattr(gui_app, "_class_rankings", None),
    )
    saved_results_df = getattr(gui_app, "results_df", None)
    gui_app.optimization_method.set("unified")
    gui_app.task_type.set("classification")
    gui_app.n_unified_trials.set(2)
    gui_app.folds.set(3)
    gui_app.output_dir.set(str(tmp_path / "out"))
    gui_app.bayesian_persistence_mode.set("never")
    gui_app.active_indices = None
    gui_app.search_controller = SearchController()
    monkeypatch.chdir(tmp_path)  # reports/ is written relative to cwd
    yield gui_app
    for name, value in saved.items():
        getattr(gui_app, name).set(value)
    (
        gui_app.active_indices,
        gui_app.excluded_spectra,
        gui_app.X,
        gui_app.y,
        gui_app.label_encoder,
        gui_app._class_rankings,
    ) = saved_state
    for widget in gui_app.region_legend_frame.winfo_children():
        widget.destroy()
    # The run populated the Results tree; later tests insert their own rows
    # with the same item ids ("Item 0 already exists"), so leave it empty.
    children = gui_app.results_tree.get_children()
    if children:
        gui_app.results_tree.delete(*children)
    gui_app.results_df = saved_results_df
    rs._reset_for_tests()


@pytest.mark.parametrize(
    "labels,expected_values",
    [(("clean", "dirty"), "Values: C0=clean, C1=dirty"), ((0.1, 0.2), "Values: C0=0.1, C1=0.2")],
    ids=["text", "fractional"],
)
def test_bayesian_run_keeps_its_display_encoder(bayes_worker, tmp_path, labels, expected_values):
    from sklearn.linear_model import LogisticRegression

    from spectral_predict.model_io import load_model, predict_with_model, save_model

    app = bayes_worker
    X, y = _data(labels, n_per_class=(16, 14))
    cols = [str(int(w)) for w in _wavelengths(X)]
    app.X = pd.DataFrame(X, columns=cols)
    app.y = pd.Series(y)
    app._run_analysis_thread(["PLS-DA"], "quick")

    # the shared tail of the worker must not wipe the Bayesian branch's encoder
    enc = app.label_encoder
    assert enc is not None
    assert list(enc.classes_) == list(labels)

    # the legend decodes the coded per-class keys back to the user's labels
    app._class_rankings = {"class_labels": ["0", "1"]}
    app._update_class_legend()
    assert _legend_values_text(app) == expected_values

    # a model fitted on the codes and saved with that encoder predicts user labels
    model = LogisticRegression(max_iter=1000).fit(X, enc.transform(y))
    wl = [float(c) for c in cols]
    meta = {
        "model_name": "LogisticRegression",
        "task_type": "classification",
        "wavelengths": wl,
        "n_vars": len(wl),
        "preprocessing": "raw",
    }
    path = tmp_path / "m.dasp"
    save_model(model, None, meta, path, label_encoder=enc)
    pred = predict_with_model(load_model(path), pd.DataFrame(X, columns=wl))
    assert set(np.unique(pred)) <= set(labels)
    np.testing.assert_array_equal(pred, enc.inverse_transform(model.predict(X)))
