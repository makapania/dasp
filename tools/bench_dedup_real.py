"""Production-realistic benchmark: pre-Option-A vs post-fix on example data.

Unlike ``tools/ab_dedup_compare.py`` (which uses RandomSampler for byte-
identical determinism), this tool uses the production TPE sampler with
multivariate=True — the actual sampler the user runs in the GUI. TPE
suggestion streams diverge between pre/post because PRUNED trials enter
its KDE history differently than COMPLETE-with-real-value trials, so
the explored configs are not identical. That divergence is the point:
we want to see whether post-fix delivers comparable wall-clock + best-
model quality on real data, not strictly identical exploration.

Pre/post emulation in one process:
- Pre = monkeypatch ``_register_or_prune_fingerprint`` to record-only
        (no TrialPruned). TPE sees every trial as COMPLETE.
- Post = real prune. TPE sees PRUNED for duplicates.

Both runs reuse the same TPE seed. First ~10 startup trials are
identical (random sampling); after that TPE adapts on diverging
history.

What we report per scenario:
- Wall-clock time (seconds)
- Total trial invocations
- COMPLETE trials (the useful fits)
- PRUNED trials (post only — the saved redundant work)
- Unique fingerprints (post should equal COMPLETE; pre should be lower)
- Best-row metric (RMSEcv for regression, Accuracycv for classification)
- Top-5 row identifiers
- Number of top-5 fingerprints common to both runs

Run from project root with .venv312 active:
    .venv312/Scripts/python.exe tools/bench_dedup_real.py [--n-trials 100]
"""
from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from spectral_predict.io import read_asd_dir
from spectral_predict import unified_bayesian as ub


REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"


@dataclass(frozen=True)
class Scenario:
    name: str
    model_name: str
    task_type: str
    target: str
    metric_col: str
    metric_higher_is_better: bool


SCENARIOS = {
    "pls_regression": Scenario(
        "pls_regression", "PLS", "regression", REGRESSION_TARGET,
        metric_col="RMSEcv", metric_higher_is_better=False,
    ),
    "lightgbm_classification": Scenario(
        "lightgbm_classification", "LightGBM", "classification", CLASSIFICATION_TARGET,
        metric_col="Accuracycv", metric_higher_is_better=True,
    ),
}


@contextlib.contextmanager
def disable_dedup_for_pre_emulation():
    """Monkeypatch ``_register_or_prune_fingerprint`` to record-only.

    Pre-Option-A emulation: every trial completes; TPE sees real
    COMPLETE values for all suggestions including duplicates.
    """
    original_register = ub._register_or_prune_fingerprint

    def record_without_pruning(trial, fingerprint, seen_fingerprints):
        if fingerprint not in seen_fingerprints:
            seen_fingerprints[fingerprint] = trial.number
        trial.set_user_attr("fingerprint", repr(fingerprint))

    ub._register_or_prune_fingerprint = record_without_pruning
    try:
        yield
    finally:
        ub._register_or_prune_fingerprint = original_register


def load_example_xy(max_features: int = 120):
    spectra, _meta = read_asd_dir("example")
    ref = pd.read_csv("example/BoneCollagen.csv")
    spectra = spectra.sort_index()
    ref = ref.copy()
    ref.index = (
        ref["File Number"].astype(str)
        .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )
    joined = spectra.join(ref, how="inner")
    feature_cols = list(spectra.columns)[:max_features]
    X = joined[feature_cols].to_numpy(dtype=float)
    wavelengths = np.asarray([float(c) for c in feature_cols], dtype=float)
    return X, wavelengths, joined


def run_phase(
    scenario: Scenario,
    n_trials: int,
    max_features: int,
    *,
    pre_fix_emulation: bool,
):
    """Run one phase (pre or post). Returns (df, study, elapsed_sec)."""
    X, wavelengths, joined = load_example_xy(max_features=max_features)
    y = joined[scenario.target].to_numpy(
        dtype=float if scenario.task_type == "regression" else None
    )
    cm = disable_dedup_for_pre_emulation() if pre_fix_emulation else contextlib.nullcontext()
    t0 = time.perf_counter()
    with cm:
        df, study = ub.run_unified_bayesian(
            X,
            y,
            wavelengths,
            model_name=scenario.model_name,
            task_type=scenario.task_type,
            n_trials=n_trials,
            cv_folds=3,
            cv_strategy="kfold",
            random_state=42,
            n_top_regions=3,
            enable_sqlite_persistence="never",
            early_stopping_rounds=None,
            enable_uve=False,
            verbose=False,
        )
    elapsed = time.perf_counter() - t0
    return df, study, elapsed


def fingerprint_map(study: optuna.Study) -> dict[int, str]:
    return {
        t.number: t.user_attrs["fingerprint"]
        for t in study.trials
        if "fingerprint" in t.user_attrs
        and t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < 1e9
    }


def trial_state_counts(study: optuna.Study):
    out = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0, "OTHER": 0}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            out["COMPLETE"] += 1
        elif t.state == optuna.trial.TrialState.PRUNED:
            out["PRUNED"] += 1
        elif t.state == optuna.trial.TrialState.FAIL:
            out["FAIL"] += 1
        else:
            out["OTHER"] += 1
    return out


def best_metric(df: pd.DataFrame, scenario: Scenario):
    if scenario.metric_col not in df.columns:
        return None
    if scenario.metric_higher_is_better:
        return float(df[scenario.metric_col].max())
    return float(df[scenario.metric_col].min())


def top_n_fingerprints(df: pd.DataFrame, study: optuna.Study, scenario: Scenario, n: int = 5) -> set[str]:
    if scenario.metric_col not in df.columns:
        return set()
    if "trial_number" not in df.columns:
        return set()
    fp_by_trial = fingerprint_map(study)
    asc = not scenario.metric_higher_is_better
    sorted_df = df.sort_values(scenario.metric_col, ascending=asc).head(n)
    return {
        fp_by_trial[int(t)]
        for t in sorted_df["trial_number"].astype(int).tolist()
        if int(t) in fp_by_trial
    }


def run_scenario(scenario: Scenario, n_trials: int, max_features: int):
    print(f"\n=== Scenario: {scenario.name} ===")
    print(f"  Model: {scenario.model_name}, task: {scenario.task_type}, n_trials: {n_trials}")
    print(f"  TPE sampler (production), seed=42, cv=3-fold KFold")
    print()

    print(f"  [PRE-fix emulation: dedup off, every duplicate runs]")
    pre_df, pre_study, pre_time = run_phase(scenario, n_trials, max_features, pre_fix_emulation=True)
    pre_states = trial_state_counts(pre_study)
    pre_unique_fps = len(set(fingerprint_map(pre_study).values()))
    pre_best = best_metric(pre_df, scenario)
    pre_top5 = top_n_fingerprints(pre_df, pre_study, scenario, n=5)
    print(f"    Wall-clock: {pre_time:.1f} s")
    print(f"    Trials: COMPLETE={pre_states['COMPLETE']} PRUNED={pre_states['PRUNED']} FAIL={pre_states['FAIL']}")
    print(f"    Unique fingerprints: {pre_unique_fps}")
    print(f"    Best {scenario.metric_col}: {pre_best:.4f}")
    print(f"    CSV rows: {len(pre_df)}")

    print(f"\n  [POST-fix: dedup on, duplicates raise TrialPruned]")
    post_df, post_study, post_time = run_phase(scenario, n_trials, max_features, pre_fix_emulation=False)
    post_states = trial_state_counts(post_study)
    post_unique_fps = len(set(fingerprint_map(post_study).values()))
    post_best = best_metric(post_df, scenario)
    post_top5 = top_n_fingerprints(post_df, post_study, scenario, n=5)
    print(f"    Wall-clock: {post_time:.1f} s")
    print(f"    Trials: COMPLETE={post_states['COMPLETE']} PRUNED={post_states['PRUNED']} FAIL={post_states['FAIL']}")
    print(f"    Unique fingerprints: {post_unique_fps}")
    print(f"    Best {scenario.metric_col}: {post_best:.4f}")
    print(f"    CSV rows: {len(post_df)}")

    overlap = pre_top5 & post_top5
    pct_dup_in_pre = (len(pre_df) - pre_unique_fps) / max(1, len(pre_df)) * 100

    print(f"\n  --- Comparison ---")
    print(f"    Wall-clock delta: {post_time - pre_time:+.1f} s ({(post_time/pre_time - 1)*100:+.1f}%)")
    print(f"    Pre had {pct_dup_in_pre:.1f}% duplicate rows in CSV; post has 0%")
    print(f"    Post unique fits: {post_unique_fps} vs pre {pre_unique_fps} (+{post_unique_fps - pre_unique_fps} more search coverage)")
    print(f"    Best metric delta ({scenario.metric_col}): post={post_best:.4f} pre={pre_best:.4f} ({post_best - pre_best:+.4f})")
    print(f"    Top-5 fingerprint overlap: {len(overlap)}/5")

    return {
        "scenario": scenario.name,
        "pre_time": pre_time,
        "post_time": post_time,
        "pre_unique": pre_unique_fps,
        "post_unique": post_unique_fps,
        "pre_best": pre_best,
        "post_best": post_best,
        "top5_overlap": len(overlap),
        "pre_dup_pct": pct_dup_in_pre,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        action="append",
        help="Run specific scenario(s); default runs all.",
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=120)
    args = parser.parse_args()

    names = args.scenario or list(SCENARIOS)
    summary = []
    for name in names:
        summary.append(run_scenario(SCENARIOS[name], args.n_trials, args.max_features))

    print("\n\n=== Final Summary ===")
    for row in summary:
        print(
            f"  {row['scenario']}: "
            f"pre_time={row['pre_time']:.1f}s post_time={row['post_time']:.1f}s "
            f"({(row['post_time']/row['pre_time'] - 1)*100:+.1f}%); "
            f"pre_unique={row['pre_unique']} post_unique={row['post_unique']}; "
            f"top5_overlap={row['top5_overlap']}/5; "
            f"pre_dup_pct={row['pre_dup_pct']:.1f}%; "
            f"best_metric_delta={row['post_best'] - row['pre_best']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
