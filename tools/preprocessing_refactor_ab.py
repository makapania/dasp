"""Compare preprocessing-discovery refactor: does it produce great models the
prior path doesn't?

Tests TWO independent A/Bs from the b551421 refactor:
  * Phase 4 (TPE preprocessing): tpe_multistart=False vs True
  * Phase 2 (exhaustive preprocessing): ga_preprocess_phase2_n_seeds=0 vs 5

User's framing (clarified 2026-05-07): they pick from the FULL leaderboard,
not just top-N by CV. They filter by chemometrics criteria - high R2pred
with a small R2cv→R2pred gap. So the load-bearing question is "does the
new path produce great models (any rank) that the old path doesn't?", not
"is the top-1 stable."

Method:
  * Hold out a fixed external partition (stratified for classification).
  * For each (phase x arm x task) cell, run `run_search()` with the
    appropriate flags, varsel + region subsets ENABLED so the leaderboard
    has the same shape the user's GUI workflow produces.
  * Score the full leaderboard (capped at --score-cap by CV rank) on the
    external set via `compute_validation_metrics_for_top_models()`.
  * Per cell report:
      - max external anywhere (R2pred for regression; val_F1 / val_Accuracy
        for classification)
      - max external among models passing a chemometrics gap filter
        (|R2cv - R2pred| <= --gap-thresh, default 0.10)
      - top-K by external as a config fingerprint (preprocess + n_components)
  * Cross-arm: which top-K-by-external configs in arm-on are NOT in arm-off?
    Are the great models multistart-only, or do both paths find them?

CAVEAT: `run_search` does not plumb `random_state` to TPE / exhaustive
preprocessing discovery - both run at hardcoded internal seeds. So this is
a single-trajectory comparison. If a model is reported as "unique to
multistart," that's true for THIS trajectory, not necessarily across seeds.

Usage:
  python tools/preprocessing_refactor_ab.py --smoke
  python tools/preprocessing_refactor_ab.py
  python tools/preprocessing_refactor_ab.py --phases tpe --tasks regression
  python tools/preprocessing_refactor_ab.py --score-cap 200
"""

from __future__ import annotations

import argparse
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
from spectral_predict.search import compute_validation_metrics_for_top_models, run_search

EXAMPLE_DIR = REPO_ROOT / "example"
LABEL_CSV = EXAMPLE_DIR / "BoneCollagen.csv"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_paired_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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

    wavelengths = asd_df.columns.to_numpy(dtype=float)
    y_reg = labels["%Collagen"].to_numpy(dtype=float)

    le = LabelEncoder()
    y_cls = le.fit_transform(labels["CollagenCat"].to_numpy())
    class_names = list(le.classes_)
    return asd_df, y_reg, y_cls, wavelengths, class_names


def stratified_split_indices(y_cls, test_frac=0.30, seed=0):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y_cls))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_frac, stratify=y_cls, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def cv_sort_metric(task_type: str) -> tuple[str, bool]:
    """Return (column_name, ascending). Lower-better for regression."""
    if task_type == "regression":
        return "RMSEcv", True
    return "BalancedAcccv", False


def fingerprint_row(row: pd.Series) -> str:
    """Stable string ID for a leaderboard row - used for set diff across arms."""
    pre = str(row.get("Preprocess", ""))
    pre_base = str(row.get("PreprocessBase", pre))
    n_vars = int(row.get("n_vars", -1)) if pd.notna(row.get("n_vars", np.nan)) else -1
    # Params is a dict-like string. Capture n_components / LVs and the model
    # name to discriminate hyperparameter siblings from the same preprocess.
    params = str(row.get("Params", "{}"))[:200]
    model = str(row.get("Model", ""))
    return f"{model}|{pre or pre_base}|n_vars={n_vars}|{params}"


def score_full_leaderboard(
    results_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    wavelengths: np.ndarray,
    task_type: str,
    score_cap: int,
) -> pd.DataFrame:
    """Take top-cap by CV, rebuild + score each on external. Returns merged df with
    both CV columns and external columns."""
    sort_col, asc = cv_sort_metric(task_type)
    sorted_df = results_df.sort_values(sort_col, ascending=asc, na_position="last")
    capped = sorted_df.head(score_cap).copy().reset_index(drop=True)
    val_df = compute_validation_metrics_for_top_models(
        df_results=capped,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type=task_type,
        wavelengths=wavelengths,
        top_n=len(capped),
    )
    return val_df


def cell_summary(scored_df: pd.DataFrame, task_type: str, gap_thresh: float) -> dict:
    """Build the user-realistic passing set for an arm.

    The user picks from the FULL leaderboard with chemometrics filters, NOT
    from top-N-by-CV (which are typically overfit). The honest-pick candidate
    set is: rows where |R2cv - R2pred| (or |BAcc_cv - Acc_val|) <= gap_thresh.
    Within that passing set, the user evaluates by external metric + physical
    sense.

    Returns the full passing set (every row that clears the gap filter) so
    the cross-arm set-diff has the full picture, not a top-K projection."""
    out: dict[str, Any] = {"n_scored": int(len(scored_df))}

    if task_type == "regression":
        cv_col, ext_col = "R2cv", "R2pred"
        ext_vals = pd.to_numeric(scored_df[ext_col], errors="coerce")
        cv_vals = pd.to_numeric(scored_df[cv_col], errors="coerce")
        n_completed = int(ext_vals.notna().sum())
        out["n_completed"] = n_completed
        out["task_metric"] = "R2pred"

        if n_completed == 0:
            out["passing"] = []
            return out

        gap = (cv_vals - ext_vals).abs()
        mask = (gap <= gap_thresh) & cv_vals.notna() & ext_vals.notna()
        passing_rows: list[dict[str, Any]] = []
        for i in scored_df.index[mask]:
            passing_rows.append({
                "fp": fingerprint_row(scored_df.loc[i]),
                "preprocess": str(scored_df.loc[i].get("Preprocess", "")),
                "n_vars": int(scored_df.loc[i].get("n_vars", -1)) if pd.notna(scored_df.loc[i].get("n_vars", np.nan)) else -1,
                "params": str(scored_df.loc[i].get("Params", "{}"))[:120],
                "R2cv": float(cv_vals.loc[i]),
                "R2pred": float(ext_vals.loc[i]),
                "gap": float(gap.loc[i]),
            })
        # Sort by external descending - best generalizers first
        passing_rows.sort(key=lambda r: -r["R2pred"])
        out["passing"] = passing_rows
        out["n_passing"] = len(passing_rows)
    else:
        cv_col = "BalancedAcccv"
        ext_col = "val_F1"
        acc_col = "val_Accuracy"
        ext_vals = pd.to_numeric(scored_df[ext_col], errors="coerce")
        acc_vals = pd.to_numeric(scored_df[acc_col], errors="coerce")
        cv_vals = pd.to_numeric(scored_df[cv_col], errors="coerce")
        n_completed = int(ext_vals.notna().sum())
        out["n_completed"] = n_completed
        out["task_metric"] = "val_F1"

        if n_completed == 0:
            out["passing"] = []
            return out

        # For classification, gap is BAcc_cv vs val_Accuracy (probe-on-self vs
        # probe-on-external on the same metric type). val_F1 reported separately.
        gap = (cv_vals - acc_vals).abs()
        mask = (gap <= gap_thresh) & cv_vals.notna() & acc_vals.notna() & ext_vals.notna()
        passing_rows = []
        for i in scored_df.index[mask]:
            passing_rows.append({
                "fp": fingerprint_row(scored_df.loc[i]),
                "preprocess": str(scored_df.loc[i].get("Preprocess", "")),
                "n_vars": int(scored_df.loc[i].get("n_vars", -1)) if pd.notna(scored_df.loc[i].get("n_vars", np.nan)) else -1,
                "params": str(scored_df.loc[i].get("Params", "{}"))[:120],
                "BAcccv": float(cv_vals.loc[i]),
                "val_Accuracy": float(acc_vals.loc[i]),
                "val_F1": float(ext_vals.loc[i]),
                "gap": float(gap.loc[i]),
            })
        passing_rows.sort(key=lambda r: -r["val_F1"])
        out["passing"] = passing_rows
        out["n_passing"] = len(passing_rows)
    return out


def run_one_cell(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    wavelengths: np.ndarray,
    *,
    model_name: str,
    task_type: str,
    phase: str,           # "tpe" or "exhaustive"
    arm_on: bool,         # the under-test toggle (True = phase enabled, False = legacy)
    n_trials: int,
    n_starts: int,
    n_top: int,
    cv_folds: int,
    score_cap: int,
    gap_thresh: float,
) -> dict[str, Any]:
    y_train_s = pd.Series(y_train, name="target", index=X_train_df.index)

    # Phase-specific kwargs
    extra_kwargs: dict[str, Any] = {}
    if phase == "tpe":
        extra_kwargs.update(dict(
            tpe_preprocess=True,
            tpe_preprocess_n_trials=n_trials,
            tpe_preprocess_n_top=n_top,
            tpe_enable_autoscale=True,
            tpe_multistart=arm_on,
            tpe_n_starts=n_starts,
        ))
    elif phase == "exhaustive":
        extra_kwargs.update(dict(
            ga_preprocess=True,
            ga_preprocess_cv_folds=cv_folds,
            ga_preprocess_autoscale=True,
            ga_preprocess_phase2_n_seeds=(5 if arm_on else 0),
        ))
    else:
        raise ValueError(f"unknown phase: {phase}")

    t0 = time.time()
    results_df, _meta = run_search(
        X_train_df,
        y_train_s,
        task_type=task_type,
        folds=cv_folds,
        cv_strategy="kfold",
        cv_n_repeats=1,
        models_to_test=[model_name],
        # User-realistic: leave varsel + region subsets ON so the leaderboard
        # is its natural shape.
        enable_variable_subsets=True,
        enable_region_subsets=True,
        max_n_components=10,
        progress_callback=None,
        **extra_kwargs,
    )
    elapsed = time.time() - t0

    if results_df is None or len(results_df) == 0:
        return {"ok": False, "error": "empty results_df", "elapsed_s": elapsed}

    results_df = results_df.copy()
    if "Model" not in results_df.columns:
        results_df["Model"] = model_name

    t1 = time.time()
    scored_df = score_full_leaderboard(
        results_df,
        X_train=X_train_df.to_numpy(dtype=float),
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        wavelengths=wavelengths,
        task_type=task_type,
        score_cap=score_cap,
    )
    score_elapsed = time.time() - t1

    summary = cell_summary(scored_df, task_type, gap_thresh)
    summary.update({
        "ok": True,
        "elapsed_s_search": elapsed,
        "elapsed_s_score": score_elapsed,
        "n_results_total": int(len(results_df)),
    })
    return summary


def task_config(task: str) -> tuple[str, str]:
    if task == "regression":
        return "PLS", "regression"
    if task == "classification":
        return "PLS-DA", "classification"
    raise ValueError(task)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny: phase=tpe, regression only, score-cap=30, ~2 min.")
    parser.add_argument("--phases", default="tpe,exhaustive",
                        help="Comma-sep subset of {tpe,exhaustive}.")
    parser.add_argument("--tasks", default="regression,classification")
    parser.add_argument("--trials", type=int, default=75,
                        help="TPE trials per arm (only used by tpe phase).")
    parser.add_argument("--n-starts", type=int, default=5,
                        help="Multi-start count when tpe_multistart=True.")
    parser.add_argument("--n-top", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--score-cap", type=int, default=300,
                        help="Cap leaderboard scoring to top-N by CV (default 300 - large enough "
                             "to capture honest mid-rank models, bounded for wall time).")
    parser.add_argument("--gap-thresh", type=float, default=0.10,
                        help="|R2cv - R2pred| filter for balanced models (default 0.10) - "
                             "the user picks models that balance CV and external, so this is "
                             "the load-bearing threshold.")
    parser.add_argument("--print-top", type=int, default=15,
                        help="How many passing rows per arm to print in the report.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.phases = "tpe"
        args.tasks = "regression"
        args.trials = 20
        args.n_starts = 3
        args.score_cap = 50

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    arms = [("off", False), ("on", True)]

    print("=" * 72)
    print("Preprocessing-discovery refactor A/B - BoneCollagen, single trajectory")
    print("=" * 72)
    X_df, y_reg, y_cls, wavelengths, class_names = load_paired_data()
    print(f"Loaded n={len(y_reg)} samples x {X_df.shape[1]} wavelengths")

    train_idx, val_idx = stratified_split_indices(y_cls, seed=args.split_seed)
    X_train_df = X_df.iloc[train_idx]
    X_val = X_df.iloc[val_idx].to_numpy(dtype=float)
    print(f"Split: train n={len(train_idx)}, external n={len(val_idx)}")
    print(f"Sweep: phases={phases}, tasks={tasks}, score_cap={args.score_cap}, "
          f"gap_thresh={args.gap_thresh} (|R2cv - R2pred| balance filter)")
    print()

    cells: list[dict[str, Any]] = []
    total = len(phases) * len(tasks) * len(arms)
    cell_i = 0

    for phase in phases:
        for task in tasks:
            model_name, task_type = task_config(task)
            y_full = y_reg if task == "regression" else y_cls
            y_train = y_full[train_idx]
            y_val = y_full[val_idx]

            for arm_label, arm_on in arms:
                cell_i += 1
                tag = f"[{cell_i}/{total}] phase={phase} task={task} arm={arm_label}"
                print(tag + " ...", flush=True)
                try:
                    summary = run_one_cell(
                        X_train_df=X_train_df,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        wavelengths=wavelengths,
                        model_name=model_name,
                        task_type=task_type,
                        phase=phase,
                        arm_on=arm_on,
                        n_trials=args.trials,
                        n_starts=args.n_starts,
                        n_top=args.n_top,
                        cv_folds=args.cv_folds,
                        score_cap=args.score_cap,
                        gap_thresh=args.gap_thresh,
                    )
                except Exception as exc:
                    summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    import traceback
                    traceback.print_exc()
                summary.update({
                    "phase": phase, "task": task, "arm": arm_label,
                    "arm_on": arm_on, "model": model_name,
                })
                cells.append(summary)
                if summary.get("ok"):
                    n_pass = summary.get("n_passing", 0)
                    passing = summary.get("passing", [])
                    if task == "regression":
                        best_pass_str = (f"{passing[0]['R2pred']:.3f}@{passing[0]['R2cv']:.3f}cv"
                                         if passing else "-")
                        print(f"    ok (search {summary['elapsed_s_search']:.0f}s, "
                              f"score {summary['elapsed_s_score']:.0f}s) "
                              f"n_results={summary['n_results_total']} "
                              f"n_scored={summary['n_scored']} | "
                              f"n_passing={n_pass}  best passing R2pred={best_pass_str}")
                    else:
                        best_pass_str = (f"F1={passing[0]['val_F1']:.3f} Acc={passing[0]['val_Accuracy']:.3f}@BAcc{passing[0]['BAcccv']:.3f}"
                                         if passing else "-")
                        print(f"    ok (search {summary['elapsed_s_search']:.0f}s, "
                              f"score {summary['elapsed_s_score']:.0f}s) "
                              f"n_results={summary['n_results_total']} "
                              f"n_scored={summary['n_scored']} | "
                              f"n_passing={n_pass}  best passing {best_pass_str}")
                else:
                    print(f"    FAILED: {summary.get('error')}")

    # Detailed report per (phase x task): "similar and high" passing models
    print()
    print("=" * 72)
    print("PASSING-SET REPORT - models where R2cv and R2pred are similar AND high")
    print(f"(gap filter: |R2cv - R2pred| <= {args.gap_thresh:.2f})")
    print("Single trajectory per arm - no seed variance available.")
    print("=" * 72)

    aggregated = {}
    for phase in phases:
        for task in tasks:
            off = next((c for c in cells if c.get("ok") and c["phase"] == phase
                        and c["task"] == task and c["arm"] == "off"), None)
            on = next((c for c in cells if c.get("ok") and c["phase"] == phase
                       and c["task"] == task and c["arm"] == "on"), None)
            if off is None or on is None:
                continue

            print(f"\n========== phase={phase} task={task} ==========")
            n_off, n_on = off["n_passing"], on["n_passing"]
            print(f"  n passing gap filter: off (legacy)={n_off}  on (refactor)={n_on}")

            if task == "regression":
                # Print top-N passing for each arm
                for arm_label, cell in [("OFF (legacy)", off), ("ON (refactor)", on)]:
                    rows = cell["passing"][: args.print_top]
                    print(f"\n  arm={arm_label} - top {len(rows)} passing models by R2pred:")
                    print(f"    {'rank':>4}  {'R2pred':>7}  {'R2cv':>7}  {'gap':>6}  {'n_vars':>6}  preprocess")
                    for i, r in enumerate(rows):
                        print(f"    {i+1:>4}  {r['R2pred']:>7.4f}  {r['R2cv']:>7.4f}  "
                              f"{r['gap']:>6.4f}  {r['n_vars']:>6}  {r['preprocess']}")
                # Set diff at row fingerprint level - full passing sets
                off_set = {r["fp"] for r in off["passing"]}
                on_set = {r["fp"] for r in on["passing"]}
                shared = off_set & on_set
                only_off = off_set - on_set
                only_on = on_set - off_set
                print(f"\n  full passing-set diff (row fingerprint):")
                print(f"    shared (passing in BOTH arms):    {len(shared)}")
                print(f"    unique to OFF (legacy only):      {len(only_off)}")
                print(f"    unique to ON  (refactor only):    {len(only_on)}")
                # If the refactor produced great-by-our-criteria models the legacy
                # didn't, list the top of those by R2pred
                only_on_rows = [r for r in on["passing"] if r["fp"] in only_on][:5]
                only_off_rows = [r for r in off["passing"] if r["fp"] in only_off][:5]
                if only_on_rows:
                    print(f"    *** top passing models UNIQUE to refactor (would be lost without it):")
                    for r in only_on_rows:
                        print(f"        R2pred={r['R2pred']:.4f} R2cv={r['R2cv']:.4f} gap={r['gap']:.4f}  {r['preprocess']}")
                if only_off_rows:
                    print(f"    *** top passing models UNIQUE to legacy (would be lost WITH refactor):")
                    for r in only_off_rows:
                        print(f"        R2pred={r['R2pred']:.4f} R2cv={r['R2cv']:.4f} gap={r['gap']:.4f}  {r['preprocess']}")
                # Best passing in each arm
                off_best = off["passing"][0]["R2pred"] if off["passing"] else float("nan")
                on_best = on["passing"][0]["R2pred"] if on["passing"] else float("nan")
                print(f"\n  BEST passing R2pred:  off={off_best:.4f}  on={on_best:.4f}  delta=={on_best-off_best:+.4f}")
                aggregated[(phase, task)] = {
                    "off_n_passing": n_off, "on_n_passing": n_on,
                    "off_best_passing_R2pred": off_best,
                    "on_best_passing_R2pred": on_best,
                    "shared": len(shared),
                    "unique_to_off": len(only_off),
                    "unique_to_on": len(only_on),
                }
            else:  # classification
                for arm_label, cell in [("OFF (legacy)", off), ("ON (refactor)", on)]:
                    rows = cell["passing"][: args.print_top]
                    print(f"\n  arm={arm_label} - top {len(rows)} passing models by val_F1:")
                    print(f"    {'rank':>4}  {'F1':>6}  {'val_Acc':>7}  {'BAcccv':>7}  {'gap':>6}  {'n_vars':>6}  preprocess")
                    for i, r in enumerate(rows):
                        print(f"    {i+1:>4}  {r['val_F1']:>6.3f}  {r['val_Accuracy']:>7.3f}  "
                              f"{r['BAcccv']:>7.3f}  {r['gap']:>6.3f}  {r['n_vars']:>6}  {r['preprocess']}")
                off_set = {r["fp"] for r in off["passing"]}
                on_set = {r["fp"] for r in on["passing"]}
                shared = off_set & on_set
                only_off = off_set - on_set
                only_on = on_set - off_set
                print(f"\n  full passing-set diff (row fingerprint):")
                print(f"    shared (passing in BOTH arms):    {len(shared)}")
                print(f"    unique to OFF (legacy only):      {len(only_off)}")
                print(f"    unique to ON  (refactor only):    {len(only_on)}")
                only_on_rows = [r for r in on["passing"] if r["fp"] in only_on][:5]
                only_off_rows = [r for r in off["passing"] if r["fp"] in only_off][:5]
                if only_on_rows:
                    print(f"    *** top passing UNIQUE to refactor:")
                    for r in only_on_rows:
                        print(f"        F1={r['val_F1']:.3f} val_Acc={r['val_Accuracy']:.3f} BAcccv={r['BAcccv']:.3f} gap={r['gap']:.3f}  {r['preprocess']}")
                if only_off_rows:
                    print(f"    *** top passing UNIQUE to legacy:")
                    for r in only_off_rows:
                        print(f"        F1={r['val_F1']:.3f} val_Acc={r['val_Accuracy']:.3f} BAcccv={r['BAcccv']:.3f} gap={r['gap']:.3f}  {r['preprocess']}")
                off_best = off["passing"][0]["val_F1"] if off["passing"] else float("nan")
                on_best = on["passing"][0]["val_F1"] if on["passing"] else float("nan")
                print(f"\n  BEST passing val_F1:  off={off_best:.4f}  on={on_best:.4f}  delta=={on_best-off_best:+.4f}")
                aggregated[(phase, task)] = {
                    "off_n_passing": n_off, "on_n_passing": n_on,
                    "off_best_passing_val_F1": off_best,
                    "on_best_passing_val_F1": on_best,
                    "shared": len(shared),
                    "unique_to_off": len(only_off),
                    "unique_to_on": len(only_on),
                }

    out_path = (
        Path(args.out) if args.out
        else REPO_ROOT / "tools" / f"_preprocessing_refactor_ab_v3_{int(time.time())}.json"
    )
    out_path.write_text(json.dumps({
        "config": {
            "phases": phases, "tasks": tasks, "trials": args.trials,
            "n_starts": args.n_starts, "n_top": args.n_top,
            "cv_folds": args.cv_folds, "score_cap": args.score_cap,
            "gap_thresh": args.gap_thresh, "print_top": args.print_top,
            "split_seed": args.split_seed,
            "n_train": len(train_idx), "n_val": len(val_idx),
        },
        "cells": cells,
        "aggregated": {f"{p}|{t}": v for (p, t), v in aggregated.items()},
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")
    failed = [c for c in cells if not c.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
