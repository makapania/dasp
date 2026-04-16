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


def load_bone_collagen():
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
        ref.set_index("__key__")[["%Collagen"]], how="inner"
    )
    joined = joined.dropna(subset=["%Collagen"])

    y = joined["%Collagen"].astype(float)
    X = joined.drop(columns=["%Collagen"]).astype(float)
    return X, y


def _is_finite(v):
    if not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    f = float(v)
    return not (np.isnan(f) or np.isinf(f))


def _nan_count(series):
    if series is None:
        return None
    return int(len(series) - int(series.apply(_is_finite).sum()))


def _finite_stat(series, stat):
    if series is None or len(series) == 0:
        return None
    finite = series[series.apply(_is_finite)]
    if len(finite) == 0:
        return None
    return float(getattr(finite, stat)())


def run_one_model(model_name, X, y):
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
            df_out, _ = run_search(X, y, task_type="regression", **kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    stdout_text = captured.getvalue()
    warning_lines = [L for L in stdout_text.splitlines() if "parameter capture" in L]

    record = {
        "model": model_name,
        "error": error,
        "warning_feature_mismatch_count": len(warning_lines),
        "warning_sample": warning_lines[:5],
    }

    if df_out is not None and len(df_out):
        rmse = df_out.get("RMSE")
        rmsecv = df_out.get("RMSEcv")
        record["n_rows"] = int(len(df_out))
        record["n_nan_cal_rmse"] = _nan_count(rmse)
        record["n_nan_cv_rmse"] = _nan_count(rmsecv)
        record["best_cal_rmse"] = _finite_stat(rmse, "min")
        record["median_cal_rmse"] = _finite_stat(rmse, "median")
        record["best_cv_rmse"] = _finite_stat(rmsecv, "min")
        record["median_cv_rmse"] = _finite_stat(rmsecv, "median")
    else:
        record["n_rows"] = 0

    return record


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return None


def main():
    import platform
    import sklearn
    import lightgbm

    # Redirect loader stdout (ASD reader prints) so it doesn't contaminate JSON output
    load_silence = io.StringIO()
    with redirect_stdout(load_silence):
        X, y = load_bone_collagen()
    print(f"[verify] dataset X={X.shape}, y={y.shape}", file=sys.stderr)

    lgbm_rec = run_one_model("LightGBM", X, y)
    pls_rec = run_one_model("PLS", X, y)

    result = {
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "git_sha": git_sha(),
        },
        "LightGBM": lgbm_rec,
        "PLS": pls_rec,
    }

    bug_present = any(
        (r.get("error") is not None)
        or ((r.get("n_rows") or 0) == 0)
        or ((r.get("warning_feature_mismatch_count") or 0) > 0)
        or ((r.get("n_nan_cal_rmse") or 0) > 0)
        or ((r.get("n_nan_cv_rmse") or 0) > 0)
        for r in (lgbm_rec, pls_rec)
    )
    result["bug_present"] = bool(bug_present)

    print(json.dumps(result, indent=2, default=str))
    sys.exit(1 if bug_present else 0)


if __name__ == "__main__":
    main()
