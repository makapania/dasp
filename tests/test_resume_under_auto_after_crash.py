"""Crash recovery works under the user's 'auto' persistence setting.

The GUI used to force the persistence radio to 'always' when the user accepted
"Resume previous run?", because 'auto' once ignored the saved study. Since the T-41
follow-up, 'auto' resumes a stored study whose name and data fingerprint match, so
the flip was removed. These tests prove the premise with a real crash: a child
process runs an 'auto' search through ``run_state``, migrates to SQLite, and is
killed with ``os._exit`` mid-run (no clean shutdown, no WAL checkpoint). This
process then resumes through ``find_incomplete_run`` / ``resume_run`` and re-runs
under 'auto'.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import optuna
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = 14
CRASH_AFTER = 12  # completed trials when the child dies; > the 10-trial warmup

_CHILD = textwrap.dedent(
    """
    import os, sys, time
    import numpy as np
    from spectral_predict import run_state as rs
    from spectral_predict import unified_bayesian as ub

    data = np.load(sys.argv[1])
    X, y, wl = data["X"], data["y"], data["wl"]
    real = ub.cross_val_predict_pooled

    def slow(*args, **kwargs):  # > the 1 s auto threshold, so the study migrates
        time.sleep(1.05)
        return real(*args, **kwargs)

    ub.cross_val_predict_pooled = slow
    rs.start_run(
        label="crash", dataset_fingerprint=rs.fingerprint_dataset(X, y),
        model_names=["PLS", "Ridge"], n_trials_per_model=int(sys.argv[2]),
        bayesian_persistence_mode="auto",
    )

    def crash(msg):
        if msg.get("current") == int(sys.argv[3]) and "Trial" in msg.get("message", ""):
            os._exit(17)

    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression",
        n_trials=int(sys.argv[2]), cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="auto", progress_callback=crash,
    )
    os._exit(99)  # the crash must happen before the run finishes
    """
)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    n_samples, n_features = 40, 60
    y = rng.uniform(0.0, 10.0, n_samples)
    X = np.outer(y, np.sin(np.linspace(0.0, 3.0 * np.pi, n_features)))
    X += rng.normal(0.0, 0.3, (n_samples, n_features))
    return X, y, np.linspace(1000.0, 2500.0, n_features)


@pytest.fixture
def crashed_auto_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reimport_modules):
    """Run an 'auto' PLS search in a child process and kill it after migration."""
    env_key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_DATA_HOME"
    monkeypatch.setenv(env_key, str(tmp_path))
    X, y, wl = _data()
    npz = tmp_path / "data.npz"
    np.savez(npz, X=X, y=y, wl=wl)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(npz), str(TARGET), str(CRASH_AFTER)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 17, f"child did not crash as planned:\n{proc.stderr[-3000:]}"

    _, rs = reimport_modules("spectral_predict.resource_paths", "spectral_predict.run_state")
    rs._reset_for_tests()
    yield rs, (X, y, wl)
    rs._reset_for_tests()


def _run(model: str, data: tuple, messages: list[dict]) -> optuna.Study:
    from spectral_predict import unified_bayesian as ub

    X, y, wl = data
    _, study = ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name=model, task_type="regression",
        n_trials=TARGET, cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="auto", progress_callback=messages.append,
    )
    return study


def _snapshot(url: str, name: str) -> list[tuple]:
    study = optuna.load_study(study_name=name, storage=url)
    return [(t.number, t.state, t.params, t.value) for t in study.trials]


def test_auto_resumes_a_crashed_migrated_run(crashed_auto_run) -> None:
    rs, data = crashed_auto_run

    meta = rs.find_incomplete_run()
    assert meta is not None, "the crash should leave the resume sidecar behind"
    assert meta.bayesian_persistence_mode == "auto"
    assert rs.resume_run(meta.run_id) is not None
    url = rs.get_storage_url()
    names = optuna.study.get_all_study_names(storage=url)
    assert len(names) == 1, names
    name = names[0]
    before = _snapshot(url, name)
    completed = optuna.trial.TrialState.COMPLETE
    assert sum(s == completed for _, s, _, _ in before) == CRASH_AFTER

    messages: list[dict] = []
    study = _run("PLS", data, messages)

    assert any(m.get("t41_decision") == "auto_resumed_existing_study" for m in messages)
    assert not isinstance(study._storage, optuna.storages.InMemoryStorage)
    assert study.study_name == name
    assert optuna.study.get_all_study_names(storage=url) == [name], "no fresh study"
    after = _snapshot(url, name)
    assert after[: len(before)] == before, "saved trials must be kept unchanged"
    assert sum(s == completed for _, s, _, _ in after) == TARGET, "continues to the target"


def test_accepted_resume_after_environment_change_is_announced(
    crashed_auto_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex review of #79: after e.g. a NumPy update the study name changes, so 'auto'
    cannot resume. It must say so (and flag the decline) rather than silently restart."""
    from spectral_predict import unified_bayesian as ub

    rs, data = crashed_auto_run
    meta = rs.find_incomplete_run()
    assert rs.resume_run(meta.run_id) is not None
    url = rs.get_storage_url()
    (old_name,) = optuna.study.get_all_study_names(storage=url)
    before = _snapshot(url, old_name)

    real_env = ub._numerical_environment

    def updated_numpy() -> dict:
        env = real_env()
        env["packages"] = dict(env["packages"], numpy="99.0.0")
        return env

    monkeypatch.setattr(ub, "_numerical_environment", updated_numpy)
    messages: list[dict] = []
    study = _run("PLS", data, messages)

    notices = [m for m in messages if m.get("environment_changed")]
    assert notices, "the environment-change diagnostic must run for an accepted 'auto' resume"
    assert all(m.get("resume_declined") for m in notices)
    assert old_name in notices[0]["message"]
    assert not any(m.get("t41_decision") == "auto_resumed_existing_study" for m in messages)
    assert study.study_name != old_name
    assert _snapshot(url, old_name) == before, "the old study is preserved untouched"


def test_resumed_run_on_different_data_flags_the_decline(crashed_auto_run) -> None:
    rs, (X, y, wl) = crashed_auto_run
    meta = rs.find_incomplete_run()
    assert rs.resume_run(meta.run_id) is not None
    url = rs.get_storage_url()
    (name,) = optuna.study.get_all_study_names(storage=url)
    before = _snapshot(url, name)

    y_changed = y.copy()
    y_changed[0] += 1.0  # same shape and configuration, so the same study name
    messages: list[dict] = []
    _run("PLS", (X, y_changed, wl), messages)

    declined = [m for m in messages if m.get("resume_declined")]
    assert [m.get("t41_decision") for m in declined] == ["auto_existing_study_data_mismatch"]
    assert _snapshot(url, name) == before


def test_auto_without_a_store_never_lists_studies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new 'auto' diagnostics must not open or create storage when no file exists."""
    import importlib

    from spectral_predict import unified_bayesian as ub

    db = tmp_path / "absent.sqlite3"
    url = f"sqlite:///{db.as_posix()}?check_same_thread=False&timeout=30"
    monkeypatch.setattr(
        importlib.import_module("spectral_predict.run_state"), "get_storage_url", lambda: url
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("studies were enumerated without an existing store")

    monkeypatch.setattr(optuna.study, "get_all_study_names", forbidden)
    X, y, wl = _data()
    ub.run_unified_bayesian(
        X=X, y=y, wavelengths=wl, model_name="PLS", task_type="regression",
        n_trials=3, cv_folds=3, random_state=7, verbose=False,
        enable_sqlite_persistence="auto",
    )
    assert not db.exists()


def test_auto_resume_leaves_unmigrated_models_fresh(crashed_auto_run) -> None:
    """Multi-model run: only PLS reached SQLite; Ridge starts over and PLS is untouched."""
    rs, data = crashed_auto_run
    meta = rs.find_incomplete_run()
    assert rs.resume_run(meta.run_id) is not None
    url = rs.get_storage_url()
    (pls_name,) = optuna.study.get_all_study_names(storage=url)
    before = _snapshot(url, pls_name)

    messages: list[dict] = []
    study = _run("Ridge", data, messages)

    assert not any(m.get("t41_decision", "").startswith("auto_") for m in messages)
    assert study.study_name != pls_name
    assert _snapshot(url, pls_name) == before
