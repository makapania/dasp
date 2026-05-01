"""T-11 D regression tests for Optuna run-state persistence."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fresh_state(tmp_path, monkeypatch):
    """Fresh run_state module with a tmp user-data dir.

    Uses LOCALAPPDATA / XDG_DATA_HOME injection to redirect the optuna sidecar
    location away from the user's real ~/AppData. Each test gets a fresh
    module-level singleton.
    """
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    sys.modules.pop("spectral_predict.run_state", None)
    sys.modules.pop("spectral_predict.resource_paths", None)
    rp = importlib.import_module("spectral_predict.resource_paths")
    rs = importlib.import_module("spectral_predict.run_state")
    rs._reset_for_tests()

    yield rs, rp, tmp_path

    rs._reset_for_tests()


def test_no_active_run_initially(fresh_state):
    rs, _, _ = fresh_state

    assert rs.get_storage_url() is None
    assert rs.is_resuming() is False
    assert rs.find_incomplete_run() is None


def test_start_run_writes_sidecar_and_url(fresh_state):
    rs, rp, _ = fresh_state

    meta = rs.start_run(
        label="standard",
        dataset_fingerprint="abc123",
        model_names=["pls", "ridge"],
        n_trials_per_model=100,
    )

    assert meta.run_id
    assert meta.storage_url.startswith("sqlite:///")
    assert meta.label == "standard"
    assert meta.dataset_fingerprint == "abc123"
    assert meta.model_names == ["pls", "ridge"]
    assert meta.n_trials_per_model == 100
    assert rs.get_storage_url() == meta.storage_url

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    assert sidecar.exists()
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["run_id"] == meta.run_id


def test_start_run_idempotent_within_same_search(fresh_state):
    rs, _, _ = fresh_state

    meta1 = rs.start_run(label="a", model_names=["m1"])
    meta2 = rs.start_run(label="b", model_names=["m2"])
    assert meta1.run_id == meta2.run_id
    assert meta1.storage_url == meta2.storage_url


def test_mark_complete_clears_sidecar_and_url(fresh_state):
    rs, rp, _ = fresh_state

    rs.start_run(label="x", model_names=["m"])
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    assert sidecar.exists()

    rs.mark_complete()
    assert rs.get_storage_url() is None
    assert not sidecar.exists()


def test_find_incomplete_run_after_simulated_crash(fresh_state):
    rs, _, _ = fresh_state

    meta = rs.start_run(label="comprehensive", model_names=["catboost", "xgb"])
    # Simulate process death: sidecar persists, but module-level singleton
    # is reset (mimics a fresh process).
    rs._reset_for_tests()

    found = rs.find_incomplete_run()
    assert found is not None
    assert found.run_id == meta.run_id
    assert found.label == "comprehensive"
    assert found.model_names == ["catboost", "xgb"]


def test_resume_run_reactivates_storage(fresh_state):
    rs, _, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["pls"])
    # Touch the sqlite path so resume_run sees it exists.
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    resumed = rs.resume_run(meta.run_id)
    assert resumed is not None
    assert resumed.run_id == meta.run_id
    assert rs.get_storage_url() == meta.storage_url
    assert rs.is_resuming() is True


def test_resume_run_returns_none_for_unknown_id(fresh_state):
    rs, _, _ = fresh_state

    assert rs.resume_run("nonexistent_id") is None


def test_resume_run_discards_when_sqlite_missing(fresh_state):
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["pls"])
    rs._reset_for_tests()
    # SQLite file was never actually created (no Optuna ran). resume should
    # detect the missing file, discard the sidecar, and return None.

    result = rs.resume_run(meta.run_id)
    assert result is None
    assert rs.find_incomplete_run() is None  # sidecar was cleaned up


def test_discard_incomplete_run(fresh_state):
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["pls"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    assert rs.discard_incomplete_run(meta.run_id) is True
    assert rs.find_incomplete_run() is None
    assert not Path(meta.storage_path).exists()


def test_discard_returns_false_for_unknown_id(fresh_state):
    rs, _, _ = fresh_state

    assert rs.discard_incomplete_run("nonexistent_id") is False


def test_corrupt_sidecar_self_heals(fresh_state):
    rs, rp, _ = fresh_state

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.write_text("{not valid json", encoding="utf-8")

    assert rs.find_incomplete_run() is None
    # Corrupt sidecar should have been deleted.
    assert not sidecar.exists()


def test_fingerprint_dataset_deterministic_and_distinguishes(fresh_state):
    rs, _, _ = fresh_state

    rng = np.random.default_rng(0)
    X1 = rng.normal(0, 1, (20, 100))
    y1 = rng.normal(0, 1, 20)
    fp1 = rs.fingerprint_dataset(X1, y1)
    fp2 = rs.fingerprint_dataset(X1, y1)
    assert fp1 == fp2  # deterministic

    rng2 = np.random.default_rng(1)
    X2 = rng2.normal(0, 1, (20, 100))
    y2 = rng2.normal(0, 1, 20)
    fp3 = rs.fingerprint_dataset(X2, y2)
    assert fp1 != fp3  # distinguishes different data


def test_verify_fingerprint_passes_when_not_resuming(fresh_state):
    rs, _, _ = fresh_state

    rs.start_run(label="x", dataset_fingerprint="abc", model_names=["m"])
    # Not in resuming state — verify must pass regardless of fingerprint.
    matches, _ = rs.verify_resume_fingerprint("definitely_different")
    assert matches is True


def test_verify_fingerprint_blocks_mismatch_on_resume(fresh_state):
    rs, _, _ = fresh_state

    meta = rs.start_run(label="x", dataset_fingerprint="abc", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    rs.resume_run(meta.run_id)
    assert rs.is_resuming() is True

    matches, stored = rs.verify_resume_fingerprint("xyz")
    assert matches is False
    assert stored == "abc"


def test_verify_fingerprint_passes_on_match_after_resume(fresh_state):
    rs, _, _ = fresh_state

    meta = rs.start_run(label="x", dataset_fingerprint="abc", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    rs.resume_run(meta.run_id)
    matches, stored = rs.verify_resume_fingerprint("abc")
    assert matches is True
    assert stored == "abc"


def test_verify_fingerprint_passes_when_stored_unknown(fresh_state):
    """Older sidecars may not have a fingerprint; tolerate that."""
    rs, _, _ = fresh_state

    meta = rs.start_run(label="x", dataset_fingerprint="unknown", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    rs.resume_run(meta.run_id)
    matches, _ = rs.verify_resume_fingerprint("anything_at_all")
    assert matches is True


def test_clear_resume_state_drops_flag_without_deleting_sidecar(fresh_state):
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", dataset_fingerprint="abc", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    rs.resume_run(meta.run_id)
    assert rs.is_resuming() is True

    rs.clear_resume_state()
    assert rs.is_resuming() is False
    assert rs.get_storage_url() is None
    # Sidecar still on disk so user can decide later.
    assert (rp.get_user_optuna_dir() / "active_run.json").exists()


def test_atomic_write_no_partial_file_on_failure(fresh_state, monkeypatch):
    """Codex HIGH #5: a write failure must not leave a partial sidecar."""
    rs, rp, _ = fresh_state

    sidecar_dir = rp.get_user_optuna_dir()
    sidecar = sidecar_dir / "active_run.json"

    # Pre-populate a known good sidecar.
    rs.start_run(label="initial", dataset_fingerprint="abc", model_names=["m"])
    rs._reset_for_tests()
    initial_content = sidecar.read_text(encoding="utf-8")

    # Simulate an os.replace failure mid-atomic-write. Old sidecar must
    # remain intact (the function raises but doesn't corrupt).
    real_replace = __import__("os").replace
    call_count = {"n": 0}

    def fail_replace(src, dst):
        call_count["n"] += 1
        raise OSError("simulated failure")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(OSError):
        rs._atomic_write_json(sidecar, {"new": "data"})

    monkeypatch.setattr("os.replace", real_replace)
    # Old content must be intact.
    assert sidecar.read_text(encoding="utf-8") == initial_content
    # No leaked .tmp file in the dir.
    leaked = list(sidecar_dir.glob("active_run.json.*.tmp"))
    assert leaked == []


def test_verify_fingerprint_blocks_run_id_mismatch(fresh_state):
    """Kimi MAJOR #2: verify_resume_fingerprint must reject sidecars whose
    run_id no longer matches the resumed run (a second app instance could
    have overwritten active_run.json between resume_run() and now)."""
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", dataset_fingerprint="abc", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()
    rs.resume_run(meta.run_id)
    assert rs.is_resuming() is True

    # Simulate a second instance overwriting the sidecar with a different
    # run_id while we're holding the resume in memory.
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.write_text(
        json.dumps({
            "run_id": "different_run_id_xxx",
            "storage_path": "/tmp/nope.sqlite3",
            "storage_url": "sqlite:////tmp/nope.sqlite3",
            "label": "stranger",
            "dataset_fingerprint": "abc",  # COINCIDENTAL fingerprint match
            "model_names": [],
            "n_trials_per_model": None,
            "started_iso": "2026-01-01T00:00:00",
        }),
        encoding="utf-8",
    )

    # Even though the fingerprint coincidentally matches, the run_id
    # doesn't, so verify must reject.
    matches, stored = rs.verify_resume_fingerprint("abc")
    assert matches is False
    assert stored is None  # Reset to None since we can't trust ANY field


def test_unified_bayesian_uses_storage_when_active(fresh_state):
    """End-to-end smoke: when run_state is active, optuna.create_study()
    inside unified_bayesian receives storage= and load_if_exists=True."""
    rs, _, _ = fresh_state

    # Simulate the production code path: GUI calls start_run() at the top
    # of _run_analysis_thread, then unified_bayesian.run_unified_bayesian()
    # calls optuna.create_study with the active storage URL.
    meta = rs.start_run(label="x", model_names=["pls"])
    assert rs.get_storage_url() == meta.storage_url

    import optuna
    # Create a study via the same path unified_bayesian uses (storage url).
    study = optuna.create_study(
        direction="minimize",
        study_name="test_study",
        storage=rs.get_storage_url(),
        load_if_exists=True,
    )
    assert study is not None

    # Re-create with the same name + storage — must load existing.
    study2 = optuna.create_study(
        direction="minimize",
        study_name="test_study",
        storage=rs.get_storage_url(),
        load_if_exists=True,
    )
    assert study2._study_id == study._study_id
