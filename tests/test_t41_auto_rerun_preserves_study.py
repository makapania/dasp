"""T-41 follow-up: re-running an 'auto' analysis must never lose the earlier study.

'auto' persistence runs the first warmup trials in memory, then, for slow models,
copies the study into SQLite under a configuration-derived name. Reported by the
T-51 reviews (DeepSeek, Codex): on a second 'auto' run of the same configuration,
that migration targets a name that already exists; if it fails, the cleanup calls
``optuna.delete_study`` on that name — the EARLIER run's study.

These tests drive the real auto-decision: trials are made slow (> the 1 s threshold)
by wrapping the CV helper, not by patching the decision itself.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest

from spectral_predict import unified_bayesian as ub

SLOW_TRIAL_S = 1.05
WARMUP = 10


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    n_samples, n_features = 40, 60
    y = rng.uniform(0.0, 10.0, n_samples)
    X = np.outer(y, np.sin(np.linspace(0.0, 3.0 * np.pi, n_features)))
    X += rng.normal(0.0, 0.3, (n_samples, n_features))
    return X, y, np.linspace(1000.0, 2500.0, n_features)


@pytest.fixture()
def slow_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db = tmp_path / "t41_rerun.sqlite3"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False&timeout=30"
    monkeypatch.setattr(
        importlib.import_module("spectral_predict.run_state"), "get_storage_url", lambda: url
    )
    real = ub.cross_val_predict_pooled

    def slow(*args: Any, **kwargs: Any):
        time.sleep(SLOW_TRIAL_S)
        return real(*args, **kwargs)

    monkeypatch.setattr(ub, "cross_val_predict_pooled", slow)
    return url


def _run(n_trials: int, **kwargs: Any) -> optuna.Study:
    X, y, wl = _data()
    _, study = ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression",
        n_trials=n_trials, cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="auto", **kwargs,
    )
    return study


def _completed(url: str, name: str) -> int:
    study = optuna.load_study(study_name=name, storage=url)
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_second_auto_run_resumes_instead_of_deleting(slow_sqlite: str) -> None:
    first = _run(WARMUP + 1)
    name = first.study_name
    assert _completed(slow_sqlite, name) == WARMUP + 1, "first run should have migrated"
    first_trials = [(t.number, t.params, t.value) for t in
                    optuna.load_study(study_name=name, storage=slow_sqlite).trials]
    messages: list[dict] = []

    second = _run(WARMUP + 3, progress_callback=messages.append)

    assert name in optuna.study.get_all_study_names(storage=slow_sqlite), (
        "the second 'auto' run deleted the first run's persisted study"
    )
    stored = optuna.load_study(study_name=name, storage=slow_sqlite)
    assert [(t.number, t.params, t.value) for t in stored.trials[: WARMUP + 1]] == first_trials
    assert _completed(slow_sqlite, name) == WARMUP + 3, "the target is a total, resumed"
    assert second.study_name == name
    assert any(m.get("t41_decision") == "auto_resumed_existing_study" for m in messages)
    assert not any(m.get("t41_decision") == "migration_failed_inmemory" for m in messages)


def test_fresh_auto_run_still_stays_in_memory_and_creates_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume check must not touch storage when there is nothing to resume."""
    db = tmp_path / "never_created.sqlite3"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False&timeout=30"
    monkeypatch.setattr(
        importlib.import_module("spectral_predict.run_state"), "get_storage_url", lambda: url
    )
    study = _run(3)
    assert isinstance(study._storage, optuna.storages.InMemoryStorage)
    assert not db.exists()


def test_auto_run_with_other_studies_in_file_does_not_resume_them(slow_sqlite: str) -> None:
    other = optuna.create_study(study_name="some_other_model_study", storage=slow_sqlite)
    other.add_trial(optuna.trial.create_trial(value=1.0))
    messages: list[dict] = []
    _run(3, progress_callback=messages.append)  # fewer trials than warmup: no migration
    assert not any(m.get("t41_decision") == "auto_resumed_existing_study" for m in messages)
    assert _completed(slow_sqlite, "some_other_model_study") == 1


def test_failed_migration_never_deletes_a_preexisting_study(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Race: the resume check misses a study that exists by migration time."""
    first = _run(WARMUP + 1)
    name = first.study_name
    real_exists = ub._study_exists
    calls = {"n": 0}

    def miss_once(url: str, study_name: str) -> bool:
        calls["n"] += 1
        return False if calls["n"] == 1 else real_exists(url, study_name)

    monkeypatch.setattr(ub, "_study_exists", miss_once)
    messages: list[dict] = []
    _run(WARMUP + 2, progress_callback=messages.append)

    assert any(m.get("t41_decision") == "migration_failed_inmemory" for m in messages), (
        "setup did not reach the failed-migration branch"
    )
    assert name in optuna.study.get_all_study_names(storage=slow_sqlite)
    assert _completed(slow_sqlite, name) == WARMUP + 1


def test_sqlite_file_exists_parses_urls(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite3"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False&timeout=30"
    assert not ub._sqlite_file_exists(url)
    db.write_bytes(b"")
    assert ub._sqlite_file_exists(url)
    assert not ub._sqlite_file_exists("postgresql://host/db")
