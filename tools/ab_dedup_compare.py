"""A/B harness for Bayesian pre-fit deduplication.

Runs dasp's unified Bayesian search twice in one process:
- pre-fix emulation: every duplicate runs a fresh fit (the
  ``_register_or_replay_fingerprint`` monkeypatch returns None and
  ``_record_fingerprint_value`` is a no-op)
- post-fix behavior: duplicate fingerprints short-circuit via the
  cached prior trial's value (TPE history bit-identical to pre-dedup)

Both runs use RandomSampler(seed=42) by monkeypatching the module-level
TPESampler symbol — RandomSampler gives byte-identical determinism for
the row-by-row comparison. Production code remains TPE-based.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from spectral_predict.io import read_asd_dir
from spectral_predict import unified_bayesian as ub


REGRESSION_TARGET = "%Collagen"
CLASSIFICATION_TARGET = "CollagenCat"
DEDUP_COLS = [
    "Params",
    "PreprocessBase",
    "Deriv",
    "Window",
    "Poly",
    "Autoscale",
    "n_vars",
    "SubsetTag",
]
REGRESSION_METRICS = ["RMSE", "R2", "RMSEcv", "R2cv", "MAEcv"]
CLASSIFICATION_METRICS = ["Accuracy", "Accuracycv", "ROC_AUC", "ROC_AUCcv", "F1", "F1cv"]


@dataclass(frozen=True)
class Scenario:
    name: str
    model_name: str
    task_type: str
    target: str


SCENARIOS = {
    "pls_regression": Scenario("pls_regression", "PLS", "regression", REGRESSION_TARGET),
    "lightgbm_regression": Scenario("lightgbm_regression", "LightGBM", "regression", REGRESSION_TARGET),
    "lightgbm_classification": Scenario("lightgbm_classification", "LightGBM", "classification", CLASSIFICATION_TARGET),
}


@contextlib.contextmanager
def random_sampler_and_optional_dedup(disable_dedup: bool):
    from spectral_predict import search as search_mod

    original_sampler = ub.TPESampler
    original_register = ub._register_or_replay_fingerprint
    original_record = ub._record_fingerprint_value
    original_threading_fallback = search_mod._frozen_needs_threading_fallback

    def random_sampler_factory(*args, **kwargs):
        return optuna.samplers.RandomSampler(seed=kwargs.get("seed", 42))

    def record_without_replay(trial, fingerprint, seen_fingerprints):
        trial.set_user_attr("fingerprint", repr(fingerprint))
        return None  # always novel; caller proceeds with full fit

    def noop_record(fingerprint, trial, value, seen_fingerprints):
        pass

    ub.TPESampler = random_sampler_factory
    search_mod._frozen_needs_threading_fallback = lambda: True
    if disable_dedup:
        ub._register_or_replay_fingerprint = record_without_replay
        ub._record_fingerprint_value = noop_record
    try:
        yield
    finally:
        ub.TPESampler = original_sampler
        ub._register_or_replay_fingerprint = original_register
        ub._record_fingerprint_value = original_record
        search_mod._frozen_needs_threading_fallback = original_threading_fallback


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
    y_reg = joined[REGRESSION_TARGET].to_numpy(dtype=float)
    y_cls = joined[CLASSIFICATION_TARGET].to_numpy()
    return X, wavelengths, {REGRESSION_TARGET: y_reg, CLASSIFICATION_TARGET: y_cls}


def run_case(scenario: Scenario, n_trials: int, disable_dedup: bool, max_features: int):
    X, wavelengths, targets = load_example_xy(max_features=max_features)
    with random_sampler_and_optional_dedup(disable_dedup=disable_dedup):
        df, study = ub.run_unified_bayesian(
            X,
            targets[scenario.target],
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
    return df, study


def trial_fingerprint_map(study: optuna.Study) -> dict[int, str]:
    return {
        trial.number: trial.user_attrs["fingerprint"]
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and "fingerprint" in trial.user_attrs
        and trial.value is not None
        and trial.value < 1e9
    }


def comparable_value(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        return value.hex()
    return repr(value)


def validate_match(pre_df: pd.DataFrame, pre_study: optuna.Study, post_df: pd.DataFrame, post_study: optuna.Study, task_type: str):
    pre_fp_by_trial = trial_fingerprint_map(pre_study)
    post_fp_by_trial = trial_fingerprint_map(post_study)
    pre_rows = {}
    for _, row in pre_df.iterrows():
        fp = pre_fp_by_trial.get(int(row["trial_number"]))
        if fp and fp not in pre_rows:
            pre_rows[fp] = row

    metric_cols = REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
    compare_cols = [col for col in DEDUP_COLS + metric_cols if col in pre_df.columns and col in post_df.columns]
    matched = 0
    mismatches = []
    post_seen = set()
    for _, row in post_df.iterrows():
        fp = post_fp_by_trial.get(int(row["trial_number"]))
        post_seen.add(fp)
        pre_row = pre_rows.get(fp)
        if pre_row is None:
            mismatches.append((fp, "missing in pre"))
            continue
        bad_cols = [
            col for col in compare_cols
            if comparable_value(pre_row[col]) != comparable_value(row[col])
        ]
        if bad_cols:
            mismatches.append((fp, bad_cols))
        else:
            matched += 1

    if mismatches:
        raise AssertionError(f"{len(mismatches)} post rows did not match pre: {mismatches[:3]}")

    dedup_subset = [col for col in DEDUP_COLS if col in post_df.columns]
    dup_count = int(post_df.duplicated(subset=dedup_subset).sum()) if dedup_subset else 0
    if dup_count:
        dup_mask = post_df.duplicated(subset=dedup_subset, keep=False)
        dup_rows = post_df[dup_mask].sort_values(dedup_subset)
        diag = []
        for _, row in dup_rows.iterrows():
            fp = post_fp_by_trial.get(int(row.get("trial_number", -1)), "<no-fp>")
            diag.append(
                f"  trial={int(row.get('trial_number', -1))} "
                f"SubsetTag={row.get('SubsetTag')} n_vars={row.get('n_vars')} "
                f"Params={row.get('Params')} "
                f"PreprocessBase={row.get('PreprocessBase')} "
                f"Deriv={row.get('Deriv')} Window={row.get('Window')} "
                f"Poly={row.get('Poly')} Autoscale={row.get('Autoscale')} "
                f"fp_full={fp}"
            )
        raise AssertionError(
            f"post-fix dataframe has {dup_count} duplicate rows under {dedup_subset}:\n"
            + "\n".join(diag)
        )
    if len(pre_df) < len(post_df):
        raise AssertionError(f"pre row count {len(pre_df)} < post row count {len(post_df)}")

    return {
        "pre_unique_count": len(set(pre_fp_by_trial.values())),
        "post_row_count": len(post_df),
        "match_percent": 100.0 * matched / max(1, len(post_df)),
    }


def run_scenario(scenario: Scenario, n_trials: int, max_features: int, require_dedup: bool = False):
    pre_df, pre_study = run_case(scenario, n_trials=n_trials, disable_dedup=True, max_features=max_features)
    pre_total = sum(
        1 for t in pre_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < 1e9
    )
    pre_unique = len(set(trial_fingerprint_map(pre_study).values()))
    dedup_avoided = pre_total - pre_unique
    post_df, post_study = run_case(scenario, n_trials=pre_unique, disable_dedup=False, max_features=max_features)
    result = validate_match(pre_df, pre_study, post_df, post_study, scenario.task_type)
    print(
        f"{scenario.name}: pre_total={pre_total} pre_unique={pre_unique} "
        f"dedup_avoided={dedup_avoided} post_row_count={result['post_row_count']} "
        f"match_percent={result['match_percent']:.1f}"
    )
    if dedup_avoided == 0:
        msg = (
            f"  No collisions at n_trials={n_trials}; dedup path not exercised. "
            f"Bump --n-trials to verify the prune codepath end-to-end."
        )
        if require_dedup:
            raise AssertionError(msg)
        print(msg)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Trials per scenario. Bumped from 12 to 50 so dedup is more likely to fire.",
    )
    parser.add_argument("--max-features", type=int, default=120)
    parser.add_argument(
        "--require-dedup", action="store_true",
        help="Fail if no collisions occur — proves dedup codepath was exercised.",
    )
    args = parser.parse_args()

    names = args.scenario or list(SCENARIOS)
    for name in names:
        run_scenario(
            SCENARIOS[name],
            n_trials=args.n_trials,
            max_features=args.max_features,
            require_dedup=args.require_dedup,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
