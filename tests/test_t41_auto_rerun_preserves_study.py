"""T-41 follow-up: re-running an 'auto' analysis must never lose the earlier study.

'auto' persistence runs the first warmup trials in memory, then, for slow models,
copies the study into SQLite under a configuration-derived name. Reported by the
T-51 reviews (DeepSeek, Codex): on a second 'auto' run of the same configuration,
that migration targets a name that already exists; if it fails, the cleanup calls
``optuna.delete_study`` on that name — the EARLIER run's study. Post-merge reviews of
#78 showed no guard could make that delete safe, so it was removed: a failed migration
now only warns, and these tests assert earlier studies survive every failure mode.

These tests drive the real auto-decision: trials are made slow (> the 1 s threshold)
by wrapping the CV helper, not by patching the decision itself.
"""
from __future__ import annotations

import importlib
import os
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


def _run(n_trials: int, data: tuple | None = None, **kwargs: Any) -> optuna.Study:
    X, y, wl = data if data is not None else _data()
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
    assert not ub._sqlite_file_exists(url), "zero-byte file: listing studies would write a schema"
    db.write_bytes(b"x")
    assert ub._sqlite_file_exists(url)
    assert not ub._sqlite_file_exists("postgresql://host/db")


def test_rerun_on_different_data_is_not_resumed_and_keeps_first_study(slow_sqlite: str) -> None:
    """The study name has no data identity; resuming would replay the wrong scores."""
    first = _run(WARMUP + 1)
    name = first.study_name
    assert first.user_attrs[ub.DATA_FINGERPRINT_ATTR]
    X, y, wl = _data()
    other = (X, y[::-1].copy(), wl)  # same shapes and config, different targets
    messages: list[dict] = []

    second = _run(WARMUP + 2, data=other, progress_callback=messages.append)

    decisions = {m.get("t41_decision") for m in messages}
    assert "auto_existing_study_data_mismatch" in decisions
    assert "auto_resumed_existing_study" not in decisions
    assert "migration_failed_inmemory" not in decisions, "doomed migration was attempted"
    assert isinstance(second._storage, optuna.storages.InMemoryStorage)
    assert second.user_attrs[ub.DATA_FINGERPRINT_ATTR] != first.user_attrs[ub.DATA_FINGERPRINT_ATTR]
    assert name in optuna.study.get_all_study_names(storage=slow_sqlite)
    assert _completed(slow_sqlite, name) == WARMUP + 1


def test_check_then_copy_race_never_deletes(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both existence checks miss; copy_study's DuplicatedStudyError must block deletion."""
    first = _run(WARMUP + 1)
    name = first.study_name
    monkeypatch.setattr(ub, "_study_exists", lambda url, study_name: False)
    messages: list[dict] = []
    _run(WARMUP + 2, progress_callback=messages.append)
    assert any(m.get("t41_decision") == "migration_failed_inmemory" for m in messages)
    assert _completed(slow_sqlite, name) == WARMUP + 1


def test_unanswerable_check_never_authorises_deletion(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing listing (lock, AV, share) plus a non-duplicate migration error."""
    first = _run(WARMUP + 1)
    name = first.study_name
    monkeypatch.setattr(ub, "_study_exists", lambda url, study_name: None)

    def failing_migration(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated WAL/permission failure")

    monkeypatch.setattr(ub, "_migrate_study_to_sqlite", failing_migration)
    _run(WARMUP + 2)
    assert _completed(slow_sqlite, name) == WARMUP + 1


def _deny_stat(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Make ``Path.stat`` raise PermissionError for ``target`` only, as a locked,
    permission-denied or AV-scanned file can. Real pathlib predicates are left intact:
    ``Path.is_file`` swallows that error and returns False, which is the trap."""
    real_stat = Path.stat
    denied = os.path.normcase(os.path.abspath(target))

    def stat(self: Path, *args: Any, **kwargs: Any):
        if os.path.normcase(os.path.abspath(self)) == denied:
            raise PermissionError(13, "simulated permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)


def test_file_check_stat_error_reports_not_found_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "x.sqlite3"
    db.write_bytes(b"x")
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False"
    assert ub._sqlite_file_exists(url)
    _deny_stat(monkeypatch, db)
    assert ub._sqlite_file_exists(url) is False


def test_file_check_absent_directory_and_uri_forms(tmp_path: Path) -> None:
    assert not ub._sqlite_file_exists(f"sqlite:///{(tmp_path / 'no_dir' / 'x.db').as_posix()}")
    assert not ub._sqlite_file_exists(f"sqlite:///{tmp_path.as_posix()}")
    # SQLite URI filenames name a file only via SQLite's own parsing: never "exists".
    real = tmp_path / "studies.db"
    real.write_bytes(b"x")
    assert not ub._sqlite_file_exists(f"sqlite:///file:{real.as_posix()}?uri=true")
    assert not ub._sqlite_file_exists(f"sqlite:///{real.as_posix()}?mode=ro&uri=true")


def test_uri_storage_form_never_loses_earlier_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex P2: ``sqlite:///file:x.db?uri=true`` read as a missing path. With no cleanup
    delete, a re-run on that URL stays in memory and leaves the earlier study intact."""
    db = tmp_path / "studies.db"
    url = f"sqlite:///file:{db.as_posix()}?uri=true"
    monkeypatch.setattr(
        importlib.import_module("spectral_predict.run_state"), "get_storage_url", lambda: url
    )
    X, y, wl = _data()
    _, template = ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=0,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="never",
    )
    earlier = optuna.create_study(study_name=template.study_name, storage=url)
    earlier.add_trial(optuna.trial.create_trial(value=1.0))
    real_migrate = ub._migrate_study_to_sqlite
    monkeypatch.setattr(ub, "_AUTO_THRESHOLD_S", 0.0)

    def flaky(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated transient lock during copy")

    monkeypatch.setattr(ub, "_migrate_study_to_sqlite", flaky)
    _run(WARMUP + 1)
    monkeypatch.setattr(ub, "_migrate_study_to_sqlite", real_migrate)
    assert _completed(url, template.study_name) == 1


def test_file_check_oserror_never_authorises_deletion(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex post-merge HIGH: an OSError on the file check was read as "file absent",
    so a non-duplicate migration failure deleted the earlier run's study."""
    first = _run(WARMUP + 1)
    name = first.study_name
    _deny_stat(monkeypatch, Path(slow_sqlite[len("sqlite:///"):].split("?", 1)[0]))

    def failing_migration(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated WAL/permission failure")

    monkeypatch.setattr(ub, "_migrate_study_to_sqlite", failing_migration)
    messages: list[dict] = []
    _run(WARMUP + 2, progress_callback=messages.append)
    assert any(m.get("t41_decision") == "migration_failed_inmemory" for m in messages), (
        "setup did not reach the failed-migration branch"
    )
    assert name in optuna.study.get_all_study_names(storage=slow_sqlite)
    assert _completed(slow_sqlite, name) == WARMUP + 1


def test_existing_unfingerprinted_study_is_never_stamped(slow_sqlite: str) -> None:
    """A legacy study resumed via 'always' must not acquire the current data's identity."""
    X, y, wl = _data()
    _, template = ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=0,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="never",
    )
    name = template.study_name
    legacy = optuna.create_study(study_name=name, storage=slow_sqlite, direction="minimize")
    legacy.add_trial(optuna.trial.create_trial(value=5.0))  # trials, but no fingerprint attr

    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=2,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="always",
    )
    stored = optuna.load_study(study_name=name, storage=slow_sqlite)
    assert ub.DATA_FINGERPRINT_ATTR not in stored.user_attrs
    assert len(stored.trials) == 2

    # And a later 'auto' run on the same data must therefore still refuse to resume it.
    messages: list[dict] = []
    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=3,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="auto",
        progress_callback=messages.append,
    )
    assert any(m.get("t41_decision") == "auto_existing_study_data_mismatch" for m in messages)


def test_always_resume_of_unfingerprinted_study_warns_unverified(
    slow_sqlite: str, caplog: pytest.LogCaptureFixture
) -> None:
    X, y, wl = _data()
    _, template = ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=0,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="never",
    )
    legacy = optuna.create_study(
        study_name=template.study_name, storage=slow_sqlite, direction="minimize"
    )
    legacy.add_trial(optuna.trial.create_trial(value=5.0))  # no fingerprint attr
    messages: list[dict] = []
    with caplog.at_level("WARNING", logger="spectral_predict.unified_bayesian"):
        _, resumed = ub.run_unified_bayesian(
            X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=2,
            cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="always",
            progress_callback=messages.append,
        )
    assert resumed.study_name == template.study_name
    assert len(resumed.trials) == 2, "still resumed"
    assert any(m.get("data_unverified_resume") for m in messages)
    assert not any(m.get("data_mismatch_resume") for m in messages)
    assert "can't be verified" in caplog.text


def test_always_resume_with_matching_fingerprint_does_not_warn(slow_sqlite: str) -> None:
    _run(WARMUP + 1)
    X, y, wl = _data()
    messages: list[dict] = []
    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression",
        n_trials=WARMUP + 1, cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="always", progress_callback=messages.append,
    )
    assert not any(
        m.get("data_unverified_resume") or m.get("data_mismatch_resume") for m in messages
    )


def test_always_resume_on_different_data_warns(slow_sqlite: str) -> None:
    first = _run(WARMUP + 1)
    X, y, wl = _data()
    messages: list[dict] = []
    _, resumed = ub.run_unified_bayesian(
        X=X, y=y[::-1].copy(), wavelengths=wl, model_name="PLS", task_type="regression",
        n_trials=WARMUP + 1, cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="always", progress_callback=messages.append,
    )
    assert resumed.study_name == first.study_name
    assert any(m.get("data_mismatch_resume") for m in messages)


def test_unreadable_stored_fingerprint_blocks_resume(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(WARMUP + 1)

    def unreadable(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated locked database")

    monkeypatch.setattr(ub.optuna, "load_study", unreadable)
    assert ub._stored_data_fingerprint(slow_sqlite, first.study_name) is None


def test_fingerprint_handles_empty_and_datetime_arrays() -> None:
    empty = ub._data_fingerprint(np.empty((0, 5)), np.empty(0), np.empty(0))
    assert empty != ub._data_fingerprint(np.empty((0, 6)), np.empty(0), np.empty(0))
    stamps = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
    assert ub._data_fingerprint(np.ones((2, 3)), stamps, np.arange(3.0))
    # The buffer view hashes the same bytes as a copy would (backward compatible).
    X = np.asfortranarray(np.arange(12.0).reshape(3, 4))[:, ::2]
    manual = __import__("hashlib").sha256()
    for arr in (X, np.arange(3), np.arange(2.0)):
        a = np.asarray(arr)
        manual.update(f"{a.dtype.str}|{a.shape}|".encode("utf-8"))
        manual.update(np.ascontiguousarray(a).tobytes())
    assert ub._data_fingerprint(X, np.arange(3), np.arange(2.0)) == manual.hexdigest()[:16]


def test_never_mode_does_not_fingerprint(slow_sqlite: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*args: Any, **kwargs: Any):
        raise AssertionError("fingerprint computed for a run that cannot persist")

    monkeypatch.setattr(ub, "_data_fingerprint", must_not_run)
    X, y, wl = _data()
    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression", n_trials=1,
        cv_folds=3, random_state=7, verbose=False, enable_sqlite_persistence="never",
    )


def test_failed_migration_never_deletes_a_study_created_during_the_copy(
    slow_sqlite: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Codex P1: after the absence check, another process creates the study, then this
    attempt's copy fails with a non-duplicate (transient lock) error. Nothing can prove
    who created the study, so failed migrations never delete; they warn instead."""
    created: list[str] = []

    def other_process_creates_then_copy_fails(
        study, storage_url, study_name, *args: Any, **kwargs: Any
    ):
        other = optuna.create_study(study_name=study_name, storage=storage_url)
        other.add_trial(optuna.trial.create_trial(value=3.0))
        created.append(study_name)
        raise RuntimeError("simulated database is locked during copy_study")

    monkeypatch.setattr(ub, "_migrate_study_to_sqlite", other_process_creates_then_copy_fails)
    messages: list[dict] = []
    with caplog.at_level("WARNING", logger="spectral_predict.unified_bayesian"):
        _run(WARMUP + 1, progress_callback=messages.append)
    assert created, "setup did not reach migration"
    assert any(m.get("t41_decision") == "migration_failed_inmemory" for m in messages)
    assert _completed(slow_sqlite, created[0]) == 1
    warning = " ".join(r.getMessage() for r in caplog.records)
    assert created[0] in warning and "Nothing was deleted" in warning


def test_no_code_path_deletes_studies() -> None:
    import inspect

    assert "delete_study" not in inspect.getsource(ub)
