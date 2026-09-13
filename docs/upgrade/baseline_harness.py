"""Numerical baseline harness for the Python / dependency upgrade.

Run identically before and after each upgrade phase, then diff the outputs. Because
one phase moves exactly one variable, any diff is attributable to that variable.

    <python> docs/upgrade/baseline_harness.py <label> [--outdir DIR]

Writes, per label:
    <label>_search_regression.csv      run_search ranked table (aggregate)
    <label>_search_classification.csv  run_search ranked table (aggregate)
    <label>_predictions.csv            PER-SAMPLE predictions, one row per sample
    <label>_env.json                   interpreter + full dependency manifest

Why per-sample predictions as well as the ranked tables: aggregate CV metrics can
round two different prediction vectors to the same RMSE/R2, so a table-only
comparison can report "identical" while individual predictions moved. The
per-sample file is the sharper instrument; the ranked tables catch ranking and
selection changes that the per-sample fits do not exercise.

Reproducibility controls (all deliberate, do not relax without re-baselining):
  * BLAS/OpenMP thread counts are pinned to 1 before numpy is imported. Thread
    count changes reduction order inside the native libraries and is a real
    source of last-bit drift unrelated to the upgrade under test.
  * No hash() seeding remains (MultiGroupEPO now uses a blake2b digest), so
    PYTHONHASHSEED is not controlled here. It is read only at interpreter start;
    if a historical comparison needs it, set it before launching Python. The
    manifest records whatever value was inherited.
  * Models are built with n_jobs=1; get_model's own docstring names this the
    setting for reproducibility.
  * Every model family here is seeded (42, internally). The "stochastic" models
    are included on purpose: they carry the threading and fitting risk that the
    deterministic linear models never exercise.
"""

from __future__ import annotations

import os

# MUST precede the numpy import: the native libraries read these at load time, so
# setting them afterwards has no effect.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

import spectral_predict  # noqa: E402
from spectral_predict.io import align_xy, read_asd_dir, read_reference_csv  # noqa: E402
from spectral_predict.models import get_model  # noqa: E402
from spectral_predict.search import run_search  # noqa: E402

# Wall-clock columns differ run to run by design; everything else is compared.
VOLATILE_COLS = {"Time", "Elapsed", "Duration", "Timestamp", "Runtime"}

REG_MODELS = ["PLS", "Ridge", "RandomForest", "XGBoost", "LightGBM", "MLP"]
CLF_MODELS = ["PLS-DA", "RandomForest", "XGBoost", "LightGBM"]

SEED = 42


def _manifest() -> dict:
    from importlib.metadata import distributions

    return {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "spectral_predict": spectral_predict.__version__,
        "spectral_predict_file": spectral_predict.__file__,
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "threads": {v: os.environ[v] for v in sorted(os.environ) if v.endswith("NUM_THREADS")},
        "distributions": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in distributions()
                if d.metadata and d.metadata["Name"]
            )
        ),
    }


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop timing columns and impose a total order independent of row arrival.

    Sorting matters: run_search can permute tied rows, which would surface as a
    spurious diff. The ranking column is ``Preprocess``, NOT ``Preprocessing`` --
    sorting on a name that does not exist silently does nothing, so the presence
    check below is load-bearing rather than defensive.
    """
    out = df[[c for c in df.columns if c not in VOLATILE_COLS]].copy()
    missing = {"Model", "Preprocess", "SubsetTag"} - set(out.columns)
    if missing:
        raise RuntimeError(f"expected ranking columns absent: {sorted(missing)}")
    keys = [
        c
        for c in (
            "Model",
            "Preprocess",
            "Deriv",
            "Window",
            "Poly",
            "LVs",
            "SubsetTag",
            "n_vars",
            "Rank",
        )
        if c in out.columns
    ]
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _per_sample(X: pd.DataFrame, y, task: str, models: list[str]) -> pd.DataFrame:
    """Out-of-fold predictions, one value per sample per model, on a fixed split."""
    Xv = X.to_numpy(dtype=np.float64)
    if task == "classification":
        y_enc = pd.Series(y).astype("category").cat.codes.to_numpy()
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    else:
        y_enc = np.asarray(y, dtype=np.float64)
        splitter = KFold(n_splits=3, shuffle=True, random_state=SEED)

    rows = []
    for name in models:
        oof = np.full(len(y_enc), np.nan, dtype=np.float64)
        try:
            for tr, te in splitter.split(Xv, y_enc):
                model = get_model(name, task_type=task, n_jobs=1)
                model.fit(Xv[tr], y_enc[tr])
                oof[te] = np.asarray(model.predict(Xv[te]), dtype=np.float64).ravel()
        except Exception as exc:
            # One unavailable model must not suppress the rest of the baseline.
            print(f"    [skip] {task}/{name}: {type(exc).__name__}: {exc}")
            continue
        rows.extend(
            {"task": task, "model": name, "sample": str(sid), "prediction": val}
            for sid, val in zip(X.index, oof)
        )
        print(f"    {task}/{name}: ok")
    return pd.DataFrame(rows)


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    outdir = (
        Path(sys.argv[sys.argv.index("--outdir") + 1]) if "--outdir" in sys.argv else Path.cwd()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    assert str(REPO / "src") in spectral_predict.__file__, spectral_predict.__file__

    X, _meta = read_asd_dir(str(REPO / "example"))
    ref = read_reference_csv(str(REPO / "example" / "BoneCollagen.csv"), "File Number")
    X_reg, y_reg = align_xy(X, ref, "File Number", "%Collagen")
    X_clf, y_clf = align_xy(X, ref, "File Number", "CollagenCat")

    env = _manifest()
    # Hash the inputs so a baseline can never be silently compared against
    # different data.
    env["input_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(X_reg, index=True).values.tobytes()
    ).hexdigest()[:16]
    print(
        f"python {env['python']}  numpy {env['distributions'].get('numpy')}  "
        f"input {env['input_hash']}  spectra {X.shape}"
    )

    print("  per-sample predictions:")
    preds = (
        pd.concat(
            [
                _per_sample(X_reg, y_reg, "regression", REG_MODELS),
                _per_sample(X_clf, y_clf, "classification", CLF_MODELS),
            ],
            ignore_index=True,
        )
        .sort_values(["task", "model", "sample"], kind="mergesort")
        .reset_index(drop=True)
    )
    # Fixed-width formatting: the CSV text must not vary with float repr changes.
    preds["prediction"] = preds["prediction"].map(lambda v: f"{v:.12e}")
    preds.to_csv(outdir / f"{label}_predictions.csv", index=False)
    print(f"  predictions rows={len(preds)}")

    common = dict(
        folds=3,
        cv_strategy="kfold",
        preprocessing_methods={"raw": True, "snv": True},
        max_n_components=6,
    )
    for task, Xd, yd, models in (
        ("regression", X_reg, y_reg, ["PLS", "Ridge"]),
        ("classification", X_clf, y_clf, ["PLS-DA", "Ridge"]),
    ):
        df, _le = run_search(Xd, yd, task, models_to_test=models, **common)
        _tidy(df).to_csv(outdir / f"{label}_search_{task}.csv", index=False)
        print(f"  search/{task} rows={len(df)}")

    (outdir / f"{label}_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"wrote baseline '{label}' -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
