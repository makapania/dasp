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


def test_generated_script_reproduces_mask_varsel_config(tmp_path):
    """Discrimination varsel (importance/cars/...) resolves variable_selection to
    a boolean ndarray on the view config. The exported script must embed it as a
    valid literal (not ``array([...])``, an undefined name) AND coerce it back to
    a bool mask so the reproduction runs and reproduces the in-app matrix."""
    from spectral_predict.search import (
        _multiclass_preprocess_matrix,
        _multiclass_varsel_mask,
    )

    X, y = _synthetic(K=3, n=25, p=40, seed=7)
    X.columns = [1000.0 + j for j in range(X.shape[1])]
    X_pp, wl_tr, _ = _multiclass_preprocess_matrix(
        X.to_numpy(dtype=float), _CFG, np.asarray(list(X.columns))
    )
    mask = _multiclass_varsel_mask(X_pp, y.to_numpy(), wl_tr, "importance", 20)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool

    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, alpha=0.05, n_components=0.99,
        variable_selection=mask, n_select=20, wavelengths=list(X.columns),
    )
    inapp = _inapp_decision_matrix(view)

    script = generate_multiclass_reproduction_script(
        view["config"], data_X=X, data_y=y, wavelengths=list(X.columns))
    # The bug signature: a bare numpy repr leaks an undefined ``array([...])``
    # literal into CONFIG (the coercion line uses ``np.array(CONFIG[...``, which
    # is distinct from the ``array([`` numpy-repr form).
    assert "array([" not in script
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

    # The notebook shares the same BODY template + reprsafe helper: its executed
    # code cells must also carry no bare ``array([`` literal and run to a matrix.
    nb = generate_multiclass_reproduction_notebook(
        view["config"], data_X=X, data_y=y, wavelengths=list(X.columns))
    nb = json.loads(json.dumps(nb))
    nb_code = "".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "array([" not in nb_code
    nbdir = tmp_path / "nb"
    nbdir.mkdir()
    (nbdir / "cells.py").write_text(nb_code, encoding="utf-8")
    proc_nb = subprocess.run([sys.executable, str(nbdir / "cells.py")],
                             cwd=nbdir, capture_output=True, text=True)
    assert proc_nb.returncode == 0, f"notebook code failed:\n{proc_nb.stderr}"
    assert (nbdir / "decision_matrix.csv").exists()


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


def test_generated_notebook_executes_and_reproduces(tmp_path):
    """The exported notebook must be JSON-serializable (as the GUI writes it),
    structurally valid nbformat-v4, AND runnable: concatenating its code cells
    (what Jupyter/Colab executes) reproduces the decision matrix on disk. The
    looser sibling test only checks the backend call is present, not that the
    notebook runs — this pins the runnable claim the T-31 consumer pass owes."""
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, n_components=0.99,
        wavelengths=list(X.columns),
    )
    nb = generate_multiclass_reproduction_notebook(
        view["config"], data_X=X, data_y=y, wavelengths=list(X.columns),
    )
    # Round-trips as JSON exactly as the GUI export writes it.
    nb = json.loads(json.dumps(nb, indent=1))
    # nbformat v4 top-level structure.
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert "metadata" in cell and "source" in cell
        if cell["cell_type"] == "code":
            assert "outputs" in cell and "execution_count" in cell

    code = "".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb["cells"] if c["cell_type"] == "code"
    )
    (tmp_path / "nb_cells.py").write_text(code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "nb_cells.py")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"notebook code failed:\n{proc.stderr}"
    assert (tmp_path / "decision_matrix.csv").exists()


def test_script_without_data_has_placeholder():
    X, y = _synthetic()
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=_CFG, n_components=0.99,
    )
    script = generate_multiclass_reproduction_script(view["config"])
    assert "NotImplementedError" in script
    assert "build_multiclass_decision_view" in script
