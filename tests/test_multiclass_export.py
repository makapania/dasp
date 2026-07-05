"""T-31 Phase D3: exported reproduction script/notebook == in-app decision matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_predict.code_generator import (
    generate_multiclass_reproduction_notebook,
    generate_multiclass_reproduction_script,
)
from spectral_predict.search import build_multiclass_decision_view


def _synthetic(K=3, n=25, p=24, seed=3):
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for k in range(K):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(n, p)))
        labels += [f"C{k}"] * n
    X = pd.DataFrame(np.vstack(blocks), columns=[float(j) for j in range(p)])
    y = pd.Series(labels)
    return X, y


_CFG = {
    "method": "raw", "name": "raw", "deriv": None, "window": None,
    "polyorder": None, "baseline_method": None, "baseline_params": None,
    "smoothing": False, "smoothing_window": 17, "smoothing_polyorder": 2,
}


def _inapp_decision_matrix(view):
    classes = list(view["classes"])
    P, A = view["p_values"], view["accept"]
    rows = []
    for i, sid in enumerate(view["sample_ids"]):
        accepted = [classes[j] for j in range(len(classes)) if bool(A[i, j])]
        row = {"Sample": sid, "TrueClass": view["true_labels"][i],
               "Decision": view["labels"][i]}
        for j, c in enumerate(classes):
            row[f"p({c})"] = float(P[i, j])
        row["Accepted"] = ", ".join(str(a) for a in accepted)
        rows.append(row)
    return pd.DataFrame(rows)


def test_generated_script_reproduces_decision_matrix(tmp_path):
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, alpha=0.05, n_components=0.99,
        wavelengths=list(X.columns),
    )
    inapp = _inapp_decision_matrix(view)

    script = generate_multiclass_reproduction_script(
        view["config"], data_X=X, data_y=y, wavelengths=list(X.columns),
    )
    script_path = tmp_path / "repro.py"
    script_path.write_text(script, encoding="utf-8")

    # Run it; it writes decision_matrix.csv in cwd=tmp_path.
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"

    produced = pd.read_csv(tmp_path / "decision_matrix.csv")
    # Align dtypes: Accepted NaN -> "" for empty accept sets.
    produced["Accepted"] = produced["Accepted"].fillna("")
    inapp_cmp = inapp.copy()
    inapp_cmp["Accepted"] = inapp_cmp["Accepted"].fillna("")

    assert list(produced.columns) == list(inapp_cmp.columns)
    assert list(produced["Decision"]) == list(inapp_cmp["Decision"])
    pcols = [c for c in inapp_cmp.columns if c.startswith("p(")]
    np.testing.assert_allclose(
        produced[pcols].to_numpy(), inapp_cmp[pcols].to_numpy(), rtol=1e-6, atol=1e-9,
    )


def test_generated_script_reproduces_deriv_config(tmp_path):
    """Edge-mask (deriv+window) config must reproduce identically — the export
    embeds raw X and re-runs the same preprocessing, so the mask is re-derived
    inside the same function both times."""
    X, y = _synthetic(K=3, n=25, p=40, seed=4)
    # real float wavelengths so the edge mask has meaningful columns to drop
    X.columns = [1000.0 + j for j in range(X.shape[1])]
    cfg = {"method": "deriv", "name": "deriv1_w7", "deriv": 1, "window": 7,
           "polyorder": None, "baseline_method": None, "baseline_params": None,
           "smoothing": False, "smoothing_window": 17, "smoothing_polyorder": 2}
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=cfg, alpha=0.05, n_components=0.99,
        wavelengths=list(X.columns),
    )
    inapp = _inapp_decision_matrix(view)

    script = generate_multiclass_reproduction_script(
        view["config"], data_X=X, data_y=y, wavelengths=list(X.columns))
    (tmp_path / "repro.py").write_text(script, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(tmp_path / "repro.py")],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"
    produced = pd.read_csv(tmp_path / "decision_matrix.csv")
    produced["Accepted"] = produced["Accepted"].fillna("")
    inapp["Accepted"] = inapp["Accepted"].fillna("")
    assert list(produced["Decision"]) == list(inapp["Decision"])
    pcols = [c for c in inapp.columns if c.startswith("p(")]
    np.testing.assert_allclose(
        produced[pcols].to_numpy(), inapp[pcols].to_numpy(), rtol=1e-6, atol=1e-9)


def test_generated_notebook_is_valid_nbformat():
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, n_components=0.99,
    )
    nb = generate_multiclass_reproduction_notebook(view["config"], data_X=X, data_y=y)
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in nb["cells"])
    # code cell contains the backend call
    code = "".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "build_multiclass_decision_view" in code


def test_script_without_data_has_placeholder():
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, n_components=0.99,
    )
    script = generate_multiclass_reproduction_script(view["config"])
    assert "NotImplementedError" in script
    assert "build_multiclass_decision_view" in script
