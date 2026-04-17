"""
Verify the LightGBM shared-model-state fix.

Runs run_search with GUI defaults on BoneCollagen (example/) for both
LightGBM and PLS. Captures calibration / CV metrics and counts
"parameter capture" warnings. Emits JSON to stdout.

Use via:
    .venv311/Scripts/python.exe scripts/verify_shared_model_fix.py > out.json
    .venv312/Scripts/python.exe scripts/verify_shared_model_fix.py > out.json

Exit 0 if clean. Exit 1 if any of: error raised, zero result rows,
feature-mismatch warnings, or NaN calibration/CV metrics.

GUI defaults sourced from spectral_predict_gui_optimized.py:
  preprocessing (2833-2839): raw=False, snv/sg1/sg2/deriv_snv=True, sg3/sg4=False
  window_sizes (2865-2869):  only window_17 checked -> [17]
  variable_counts (2856-2862): 10,20,50,100,250 checked (500,1000 off)
  subsets (2849-2850): enable_variable_subsets=True, enable_region_subsets=True
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXAMPLE_DIR = REPO_ROOT / "example"


def load_bone_collagen(target_col: str = "%Collagen") -> tuple[pd.DataFrame, pd.Series]:
    """Load BoneCollagen spectra + target.

    Args:
        target_col: '%Collagen' (regression) or 'CollagenCat' (classification).

    Returns:
        Tuple of (X, y) where X is the wide spectral DataFrame
        (rows=samples, columns=wavelengths) and y is the target series.
    """
    from spectral_predict.io import read_asd_dir

    spectra, _meta = read_asd_dir(EXAMPLE_DIR)
    ref = pd.read_csv(EXAMPLE_DIR / "BoneCollagen.csv", encoding="utf-8-sig")

    # Ref key: "Spectrum 00001" -> strip space -> "Spectrum00001"
    ref["__key__"] = ref["File Number"].astype(str).str.replace(" ", "", regex=False)
    # Spectra index: filename like "Spectrum00001.asd" -> strip ext
    spectra.index = (
        spectra.index.astype(str)
        .str.replace(".asd", "", regex=False)
        .str.replace(".spc", "", regex=False)
    )
    joined = spectra.join(
        ref.set_index("__key__")[[target_col]], how="inner"
    )
    joined = joined.dropna(subset=[target_col])

    y_raw = joined[target_col]
    y = y_raw.astype(float) if target_col == "%Collagen" else y_raw.astype(str)
    X = joined.drop(columns=[target_col]).astype(float)
    return X, y


def _is_finite(v: object) -> bool:
    """Return True if v is a finite numeric scalar (not NaN, not inf)."""
    if not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    f = float(v)
    return not (np.isnan(f) or np.isinf(f))


def _nan_count(series: pd.Series | None) -> int | None:
    """Return number of non-finite entries in series, or None if series is None."""
    if series is None:
        return None
    return int(len(series) - int(series.apply(_is_finite).sum()))


def _finite_stat(series: pd.Series | None, stat: str) -> float | None:
    """Apply a pandas reduction (e.g. 'min', 'max', 'median') to finite entries only."""
    if series is None or len(series) == 0:
        return None
    finite = series[series.apply(_is_finite)]
    if len(finite) == 0:
        return None
    return float(getattr(finite, stat)())


def run_one_model(model_name: str, X: pd.DataFrame, y: pd.Series, task_type: str = "regression") -> dict:
    """Run run_search for a single model and capture metrics + warning counts.

    Args:
        model_name: Identifier accepted by run_search (e.g. 'LightGBM', 'PLS', 'PLS-DA').
        X: Spectral feature matrix.
        y: Target (numeric for regression, string labels for classification).
        task_type: 'regression' or 'classification'.

    Returns:
        Dict summarizing the run: error string (or None), warning counts,
        n_rows, NaN counts, and best/median metrics for the appropriate task.
    """
    from spectral_predict.search import run_search

    kwargs = dict(
        folds=5,
        models_to_test=[model_name],
        # GUI defaults — spectral_predict_gui_optimized.py:2833-2839
        preprocessing_methods={
            "raw": False,
            "snv": True,
            "sg1": True,
            "sg2": True,
            "sg3": False,
            "sg4": False,
            "deriv_snv": True,
        },
        window_sizes=[17],  # gui:2867 default
        enable_variable_subsets=True,
        enable_region_subsets=True,
        variable_counts=[10, 20, 50, 100, 250],  # gui:2856-2860 default
        variable_selection_methods=["importance"],
    )

    captured = io.StringIO()
    error = None
    df_out = None
    try:
        with redirect_stdout(captured):
            df_out, _ = run_search(X, y, task_type=task_type, **kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    stdout_text = captured.getvalue()
    warning_lines = [L for L in stdout_text.splitlines() if "parameter capture" in L]

    record = {
        "model": model_name,
        "task_type": task_type,
        "error": error,
        "warning_feature_mismatch_count": len(warning_lines),
        "warning_sample": warning_lines[:5],
    }

    if df_out is not None and len(df_out):
        record["n_rows"] = int(len(df_out))
        if task_type == "regression":
            rmse = df_out.get("RMSE")
            rmsecv = df_out.get("RMSEcv")
            record["n_nan_cal_rmse"] = _nan_count(rmse)
            record["n_nan_cv_rmse"] = _nan_count(rmsecv)
            record["best_cal_rmse"] = _finite_stat(rmse, "min")
            record["median_cal_rmse"] = _finite_stat(rmse, "median")
            record["best_cv_rmse"] = _finite_stat(rmsecv, "min")
            record["median_cv_rmse"] = _finite_stat(rmsecv, "median")
        else:  # classification
            acc = df_out.get("Accuracy")
            acccv = df_out.get("Accuracycv")
            f1cv = df_out.get("F1cv")
            record["n_nan_cal_acc"] = _nan_count(acc)
            record["n_nan_cv_acc"] = _nan_count(acccv)
            record["n_nan_cv_f1"] = _nan_count(f1cv) if f1cv is not None else None
            record["best_cal_acc"] = _finite_stat(acc, "max")
            record["median_cal_acc"] = _finite_stat(acc, "median")
            record["best_cv_acc"] = _finite_stat(acccv, "max")
            record["median_cv_acc"] = _finite_stat(acccv, "median")
            record["best_cv_f1"] = _finite_stat(f1cv, "max") if f1cv is not None else None
    else:
        record["n_rows"] = 0

    return record


def git_sha() -> str | None:
    """Return the current HEAD commit SHA, or None if git lookup fails."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return None


def main() -> None:
    """CLI entry point. Runs regression and/or classification verification and prints JSON."""
    import argparse
    import platform
    import sklearn
    import lightgbm

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification", "both"], default="regression",
                        help="Which task types to run. Default 'regression' preserves legacy behavior.")
    args = parser.parse_args()

    result = {
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "git_sha": git_sha(),
        },
    }
    records = []

    if args.task in ("regression", "both"):
        load_silence = io.StringIO()
        with redirect_stdout(load_silence):
            X_reg, y_reg = load_bone_collagen("%Collagen")
        print(f"[verify] regression dataset X={X_reg.shape}, y={y_reg.shape}", file=sys.stderr)
        lgbm_rec = run_one_model("LightGBM", X_reg, y_reg, task_type="regression")
        pls_rec = run_one_model("PLS", X_reg, y_reg, task_type="regression")
        result["LightGBM"] = lgbm_rec
        result["PLS"] = pls_rec
        records.extend([lgbm_rec, pls_rec])

    if args.task in ("classification", "both"):
        load_silence = io.StringIO()
        with redirect_stdout(load_silence):
            X_cls, y_cls = load_bone_collagen("CollagenCat")
        print(f"[verify] classification dataset X={X_cls.shape}, y={y_cls.shape}, classes={sorted(y_cls.unique().tolist())}", file=sys.stderr)
        plsda_rec = run_one_model("PLS-DA", X_cls, y_cls, task_type="classification")
        lgbmcls_rec = run_one_model("LightGBM", X_cls, y_cls, task_type="classification")
        result["PLS-DA"] = plsda_rec
        result["LightGBM_classification"] = lgbmcls_rec
        records.extend([plsda_rec, lgbmcls_rec])

    bug_present = False
    for r in records:
        if r.get("error") is not None:
            bug_present = True
        elif (r.get("n_rows") or 0) == 0:
            bug_present = True
        elif (r.get("warning_feature_mismatch_count") or 0) > 0:
            bug_present = True
        elif (r.get("n_nan_cal_rmse") or 0) > 0 or (r.get("n_nan_cv_rmse") or 0) > 0:
            bug_present = True
        elif (r.get("n_nan_cal_acc") or 0) > 0 or (r.get("n_nan_cv_acc") or 0) > 0:
            bug_present = True
        elif (r.get("n_nan_cv_f1") or 0) > 0:
            bug_present = True
    result["bug_present"] = bool(bug_present)

    print(json.dumps(result, indent=2, default=str))
    sys.exit(1 if bug_present else 0)


if __name__ == "__main__":
    main()
