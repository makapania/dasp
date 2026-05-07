"""Compare Bayesian search with `enable_autoscale=True` vs `False` on BoneCollagen.

Question: when the Bayesian search adds `apply_autoscale ∈ {True, False}` as a
per-trial Optuna parameter (the default since T-44 / PR #54), does the best
trial it picks generalize better or worse on a held-out external set?

Design:
  * Hold out a fixed external partition (stratified for classification, random
    for regression). The same partition is used across all seeds — only the
    Bayesian search seed varies between replicates, so external numbers reflect
    search-trajectory variance, not data-split variance.
  * For each (task × autoscale_arm × seed) cell, run `run_unified_bayesian()`
    on the train set with the most autoscale-sensitive model (PLS for
    regression, PLS-DA for classification — neither scales features
    internally), pick the best trial by CV metric, then score that trial's
    rebuilt pipeline on the external set via
    `compute_validation_metrics_for_top_models()` (the same path the GUI
    Validation tab uses).
  * Also log how often TPE actually selected `apply_autoscale=True` in the
    `enable_autoscale=True` arm — distinguishes "TPE chose autoscale and
    external agreed it helped" from "TPE chose autoscale but external
    disagreed."

Usage:
  # Smoke test (1 seed, 30 trials, regression only):
  python tools/autoscale_bayesian_compare.py --smoke

  # Full sweep (3 seeds, both tasks, default 100 trials per arm):
  python tools/autoscale_bayesian_compare.py

  # Customize:
  python tools/autoscale_bayesian_compare.py --seeds 5 --trials 200 --tasks regression
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from spectral_predict.io import read_asd_dir
from spectral_predict.search import compute_validation_metrics_for_top_models
from spectral_predict.unified_bayesian import run_unified_bayesian


def _parse_params_loose(raw: Any) -> dict:
    """Parse a Params field that may be JSON, Python repr, or empty/NaN."""
    # Cover Python float NaN, numpy float scalars (any width), and pandas NA
    # uniformly. isinstance(x, float) misses np.float32 because float32 isn't
    # a Python-float subclass; isinstance(x, np.floating) catches all numpy
    # float widths. pd.isna accepts pd.NA without raising on truth-evaluation.
    if raw is None:
        return {}
    if isinstance(raw, (float, np.floating)) and np.isnan(raw):
        return {}
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except (ValueError, SyntaxError):
        return {}


EXAMPLE_DIR = REPO_ROOT / "example"
LABEL_CSV = EXAMPLE_DIR / "BoneCollagen.csv"

# Suppress chatter from sklearn / numpy during the sweep so the per-trial
# progress lines stay readable.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_paired_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load BoneCollagen labels + ASD spectra.

    Returns (X, y_reg, y_cls_int, wavelengths, class_names) where y_cls_int is
    LabelEncoder-encoded ints. PLS-DA's validation rebuild path expects integer
    class labels — passing raw 'Low'/'Medium'/'High' strings makes
    `PLS.fit(X, y)` crash trying to convert the labels to float.

    Joins on a normalized stem (strip whitespace) — CSV has 'Spectrum 00001',
    ASD files are 'Spectrum00001.asd'. Drops samples missing on either side.
    """
    from sklearn.preprocessing import LabelEncoder

    asd_df, _meta = read_asd_dir(EXAMPLE_DIR)
    asd_df.index = asd_df.index.astype(str).map(lambda s: re.sub(r"\s+", "", s))

    labels = pd.read_csv(LABEL_CSV)
    labels["join_key"] = labels["File Number"].astype(str).map(
        lambda s: re.sub(r"\s+", "", s)
    )

    common = asd_df.index.intersection(labels["join_key"])
    asd_df = asd_df.loc[common].sort_index()
    labels = labels.set_index("join_key").loc[common].sort_index()

    X = asd_df.to_numpy(dtype=float)
    wavelengths = asd_df.columns.to_numpy(dtype=float)
    y_reg = labels["%Collagen"].to_numpy(dtype=float)

    le = LabelEncoder()
    y_cls = le.fit_transform(labels["CollagenCat"].to_numpy())
    class_names = list(le.classes_)

    return X, y_reg, y_cls, wavelengths, class_names


def stratified_split_indices(
    y_cls: np.ndarray, test_frac: float = 0.30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/external split by class label, fixed across all runs."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y_cls))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_frac, stratify=y_cls, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def best_row_for_task(results_df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """Sort and slice top 1; return as a single-row df ready for validation."""
    if task_type == "regression":
        sorted_df = results_df.sort_values("RMSEcv", ascending=True, na_position="last")
    else:
        sorted_df = results_df.sort_values(
            "BalancedAcccv", ascending=False, na_position="last"
        )
    return sorted_df.head(1).copy().reset_index(drop=True)


def autoscale_chosen_fraction(results_df: pd.DataFrame) -> float | None:
    """Fraction of (non-duplicate) trials where TPE picked apply_autoscale=True.

    Returns None if the Autoscale column is absent (autoscale_off arm).
    """
    if "Autoscale" not in results_df.columns:
        return None
    col = results_df["Autoscale"].astype(bool)
    if len(col) == 0:
        return None
    return float(col.mean())


def run_one_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    wavelengths: np.ndarray,
    *,
    model_name: str,
    task_type: str,
    enable_autoscale: bool,
    seed: int,
    n_trials: int,
    cv_folds: int,
) -> dict[str, Any]:
    """Run one (arm × seed) Bayesian search cell, return summary dict."""
    t0 = time.time()
    results_df, _study = run_unified_bayesian(
        X=X_train,
        y=y_train,
        wavelengths=wavelengths,
        model_name=model_name,
        task_type=task_type,
        n_trials=n_trials,
        cv_folds=cv_folds,
        cv_strategy="kfold",
        cv_n_repeats=1,
        random_state=seed,
        verbose=False,
        enable_autoscale=enable_autoscale,
        enable_uve=False,
        enable_sqlite_persistence="never",  # in-memory only — don't pollute disk
    )
    elapsed = time.time() - t0

    if results_df is None or len(results_df) == 0:
        return {
            "ok": False,
            "error": "empty results_df",
            "elapsed_s": elapsed,
        }

    # The validation rebuild path needs Model populated (the GUI sets this
    # column after calling run_unified_bayesian).
    results_df = results_df.copy()
    if "Model" not in results_df.columns:
        results_df["Model"] = model_name

    auto_frac = autoscale_chosen_fraction(results_df)
    top1 = best_row_for_task(results_df, task_type)
    if "Model" not in top1.columns:
        top1["Model"] = model_name

    val_df = compute_validation_metrics_for_top_models(
        df_results=top1,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type=task_type,
        wavelengths=wavelengths,
        top_n=1,
    )

    row = val_df.iloc[0]
    summary: dict[str, Any] = {
        "ok": True,
        "elapsed_s": elapsed,
        "n_trials_requested": n_trials,
        "n_trials_completed": int(len(results_df)),
        "autoscale_chosen_fraction": auto_frac,
        "best_trial_autoscale": bool(row.get("Autoscale", False)),
        "best_preprocess": str(row.get("Preprocess", "")),
        "best_n_components": int(
            _parse_params_loose(row.get("Params", "{}")).get("n_components", -1)
        ),
    }
    if task_type == "regression":
        summary["best_cv_RMSEcv"] = float(row.get("RMSEcv", np.nan))
        summary["best_cv_R2cv"] = float(row.get("R2cv", np.nan))
        summary["external_RMSEP"] = float(row.get("RMSEP", np.nan))
        summary["external_R2pred"] = float(row.get("R2pred", np.nan))
    else:
        summary["best_cv_BalancedAcccv"] = float(row.get("BalancedAcccv", np.nan))
        summary["best_cv_Accuracy"] = float(row.get("Accuracy", np.nan))
        summary["external_val_Accuracy"] = float(row.get("val_Accuracy", np.nan))
        summary["external_val_F1"] = float(row.get("val_F1", np.nan))
    return summary


def task_config(task: str) -> tuple[str, str]:
    """Return (model_name, task_type_for_run_unified_bayesian)."""
    if task == "regression":
        return "PLS", "regression"
    if task == "classification":
        return "PLS-DA", "classification"
    raise ValueError(f"unknown task: {task}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny run: 1 seed, 30 trials, regression only. ~1-2 min total.",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Bayesian search seeds per arm (default 3).",
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Optuna trials per arm (default 100).",
    )
    parser.add_argument(
        "--tasks", default="regression,classification",
        help="Comma-separated subset of {regression,classification}.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Inner CV folds (default 5).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="Seed for the fixed train/external split (held constant across all "
             "arms × seeds so external numbers are comparable).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default tools/_autoscale_bayes_compare_<ts>.json).",
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.trials = 30
        args.tasks = "regression"

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = list(range(args.seeds))
    arms = [("autoscale_off", False), ("autoscale_on", True)]

    print("=" * 72)
    print("Bayesian autoscale on/off comparison — BoneCollagen")
    print("=" * 72)
    X, y_reg, y_cls, wavelengths, class_names = load_paired_data()
    print(f"Loaded n={len(y_reg)} samples × {X.shape[1]} wavelengths")
    print(f"  %Collagen range: [{y_reg.min():.2f}, {y_reg.max():.2f}], "
          f"mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")
    cls_counts = pd.Series(y_cls).value_counts().to_dict()
    cls_named = {class_names[k]: v for k, v in cls_counts.items()}
    print(f"  CollagenCat counts (encoded): {cls_counts} = {cls_named}")

    train_idx, val_idx = stratified_split_indices(
        y_cls, test_frac=0.30, seed=args.split_seed
    )
    print(f"Split (seed {args.split_seed}): train n={len(train_idx)}, "
          f"external n={len(val_idx)}")
    print(f"Sweep: tasks={tasks}, arms=on/off, seeds={seeds}, "
          f"trials/arm={args.trials}, cv_folds={args.cv_folds}")
    print()

    cells = []
    total_cells = len(tasks) * len(arms) * len(seeds)
    cell_i = 0

    for task in tasks:
        model_name, task_type = task_config(task)
        y_full = y_reg if task == "regression" else y_cls
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        for arm_label, enable_autoscale in arms:
            for seed in seeds:
                cell_i += 1
                tag = f"[{cell_i}/{total_cells}] task={task} arm={arm_label} seed={seed}"
                print(tag + " ...", flush=True)
                try:
                    summary = run_one_cell(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        wavelengths=wavelengths,
                        model_name=model_name,
                        task_type=task_type,
                        enable_autoscale=enable_autoscale,
                        seed=seed,
                        n_trials=args.trials,
                        cv_folds=args.cv_folds,
                    )
                except Exception as exc:
                    summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    import traceback
                    traceback.print_exc()
                summary.update(
                    {"task": task, "arm": arm_label, "seed": seed, "model": model_name}
                )
                cells.append(summary)
                if summary.get("ok"):
                    if task == "regression":
                        print(
                            f"    ok ({summary['elapsed_s']:.1f}s) "
                            f"best CV RMSEcv={summary['best_cv_RMSEcv']:.3f} "
                            f"R2cv={summary['best_cv_R2cv']:.3f} | "
                            f"ext RMSEP={summary['external_RMSEP']:.3f} "
                            f"R2pred={summary['external_R2pred']:.3f} | "
                            f"best_trial_auto={summary['best_trial_autoscale']} "
                            f"frac_auto={summary['autoscale_chosen_fraction']}"
                        )
                    else:
                        print(
                            f"    ok ({summary['elapsed_s']:.1f}s) "
                            f"best CV BAcc={summary['best_cv_BalancedAcccv']:.3f} | "
                            f"ext Acc={summary['external_val_Accuracy']:.3f} "
                            f"F1={summary['external_val_F1']:.3f} | "
                            f"best_trial_auto={summary['best_trial_autoscale']} "
                            f"frac_auto={summary['autoscale_chosen_fraction']}"
                        )
                else:
                    print(f"    FAILED: {summary.get('error')}")

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "tools" / f"_autoscale_bayes_compare_{int(time.time())}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "tasks": tasks,
                    "seeds": seeds,
                    "trials": args.trials,
                    "cv_folds": args.cv_folds,
                    "split_seed": args.split_seed,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                },
                "cells": cells,
            },
            indent=2,
            default=str,
        )
    )
    print()
    print(f"Wrote {out_path}")
    failed_cells = [c for c in cells if not c.get("ok")]
    if failed_cells:
        print(
            f"WARNING: {len(failed_cells)} of {len(cells)} cells failed; "
            f"returning nonzero exit so CI/automation surfaces the failure.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
