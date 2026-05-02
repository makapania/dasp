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
        bayesian_persistence_mode="auto",  # T-41: opt into SQLite to test URL
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

    result = rs.discard_incomplete_run(meta.run_id)
    assert result.fully_succeeded is True
    assert rs.find_incomplete_run() is None
    assert not Path(meta.storage_path).exists()


def test_discard_returns_false_for_unknown_id(fresh_state):
    rs, _, _ = fresh_state

    # Unknown id: empty DiscardResult (nothing to delete, no errors).
    result = rs.discard_incomplete_run("nonexistent_id")
    assert result.sidecar_deleted is False
    assert result.storage_deleted is False
    assert result.errors == []
    assert result.fully_succeeded is False


def test_corrupt_sidecar_self_heals(fresh_state):
    """find_incomplete_run quarantines corrupt sidecars rather than deleting
    them outright (PR #6 review: a downgrade from a future schema looks
    identical to a corruption, so preserve the file for forensics)."""
    rs, rp, _ = fresh_state

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.write_text("{not valid json", encoding="utf-8")

    assert rs.find_incomplete_run() is None
    # Original sidecar moved aside, not deleted.
    assert not sidecar.exists()
    assert sidecar.with_suffix(".corrupt").exists()


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
    meta = rs.start_run(label="x", model_names=["pls"],
                        bayesian_persistence_mode="auto")  # T-41: need SQLite URL
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


# ----------------------------------------------------------------------
# Tests added in response to PR #6 multi-agent review (Codex meta-review)
# ----------------------------------------------------------------------


def test_start_run_idempotent_returns_first_call_args(fresh_state):
    """Cluster C regression: second `start_run` call within the same run
    must return the FIRST call's metadata, not a Frankenstein dataclass
    with the second caller's label/fingerprint/model_names mixed in.
    """
    rs, _, _ = fresh_state

    meta1 = rs.start_run(
        label="initial",
        dataset_fingerprint="fp_initial",
        model_names=["pls", "ridge"],
        n_trials_per_model=100,
    )
    meta2 = rs.start_run(
        label="DIFFERENT",
        dataset_fingerprint="fp_DIFFERENT",
        model_names=["xgboost"],
        n_trials_per_model=999,
    )

    # All fields must reflect meta1's args, NOT meta2's.
    assert meta2.run_id == meta1.run_id
    assert meta2.storage_url == meta1.storage_url
    assert meta2.storage_path == meta1.storage_path
    assert meta2.label == "initial"
    assert meta2.dataset_fingerprint == "fp_initial"
    assert meta2.model_names == ["pls", "ridge"]
    assert meta2.n_trials_per_model == 100
    assert meta2.started_iso == meta1.started_iso


def test_mark_complete_does_not_delete_unrelated_sidecar(fresh_state):
    """NEW BUG #1 regression: mark_complete must only delete sidecars
    whose run_id matches the active run. A user with a paused Bayesian
    run who completes a fresh grid search would otherwise lose the
    paused run's sidecar.
    """
    rs, rp, _ = fresh_state

    # Simulate: a prior run's sidecar is on disk (user clicked "Decide
    # later" on the resume prompt). The user starts a fresh analysis.
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({
            "run_id": "previous_unrelated_run_id",
            "storage_path": "/tmp/old.sqlite3",
            "storage_url": "sqlite:////tmp/old.sqlite3",
            "label": "old_run",
            "dataset_fingerprint": "old_fp",
            "model_names": ["pls"],
            "n_trials_per_model": 50,
            "started_iso": "2026-01-01T00:00:00",
        }),
        encoding="utf-8",
    )

    # User runs a fresh grid search with no run_state activation
    # (grid path doesn't call start_run). mark_complete is still called
    # at the end of analysis from the GUI's success path.
    rs.mark_complete()

    # Unrelated sidecar must survive.
    assert sidecar.exists()
    surviving = json.loads(sidecar.read_text(encoding="utf-8"))
    assert surviving["run_id"] == "previous_unrelated_run_id"


def test_mark_complete_raises_on_unlink_failure(fresh_state, monkeypatch):
    """A1 regression: when sidecar.unlink() fails, mark_complete re-raises
    so the GUI handler can surface the failure, AND in-memory state is
    preserved so the next launch can still find/resume the sidecar.
    """
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["m"],
                        bayesian_persistence_mode="auto")  # T-41: need SQLite URL
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    assert sidecar.exists()

    real_unlink = Path.unlink

    def fail_unlink(self, *a, **kw):
        if self == sidecar:
            raise OSError("simulated locked file")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="simulated locked file"):
        rs.mark_complete()

    # In-memory state must be PRESERVED — we still have an active run
    # on disk that the next launch needs to be able to find.
    assert rs.get_storage_url() == meta.storage_url
    assert rs._active_run_id == meta.run_id


def test_find_incomplete_run_quarantines_corrupt_sidecar(fresh_state):
    """A2 regression: corrupt sidecar gets quarantined to .corrupt rather
    than silently deleted, so a future-schema downgrade or an actual
    corruption can be inspected after the fact.
    """
    rs, rp, _ = fresh_state

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{ this is not valid json", encoding="utf-8")

    result = rs.find_incomplete_run()
    assert result is None
    assert not sidecar.exists()
    quarantined = sidecar.with_suffix(".corrupt")
    assert quarantined.exists()
    assert "not valid json" in quarantined.read_text(encoding="utf-8")


def test_find_incomplete_run_propagates_oserror(fresh_state, monkeypatch):
    """A2 regression: a locked / unreadable sidecar must raise OSError
    rather than silently returning None — the caller (GUI startup) can
    then surface a warning rather than treating "no resume offer" as
    indistinguishable from "no run on disk".
    """
    rs, rp, _ = fresh_state

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text('{"run_id": "x"}', encoding="utf-8")

    def fail_read(self, *a, **kw):
        raise OSError("simulated locked file")

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(OSError, match="simulated locked file"):
        rs.find_incomplete_run()


def test_discard_returns_discard_result_with_per_file_status(fresh_state):
    """A3 regression: discard returns DiscardResult with per-file booleans
    and an errors list, not a bare True.
    """
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["m"])
    Path(meta.storage_path).touch()
    rs._reset_for_tests()

    result = rs.discard_incomplete_run(meta.run_id)
    assert isinstance(result, rs.DiscardResult)
    assert result.sidecar_deleted is True
    assert result.storage_deleted is True
    assert result.errors == []
    assert result.fully_succeeded is True


def test_discard_reports_partial_failure(fresh_state, monkeypatch):
    """A3 regression: when one of the unlinks fails, discard reports
    partial success with an explanatory error string.
    """
    rs, rp, _ = fresh_state

    meta = rs.start_run(label="x", model_names=["m"])
    storage_path = Path(meta.storage_path)
    storage_path.touch()
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    rs._reset_for_tests()

    real_unlink = Path.unlink

    def selective_fail_unlink(self, *a, **kw):
        if self == storage_path:
            raise OSError("simulated SQLite lock")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", selective_fail_unlink)
    result = rs.discard_incomplete_run(meta.run_id)

    assert result.sidecar_deleted is True
    assert result.storage_deleted is False
    assert result.errors and "simulated SQLite lock" in result.errors[0]
    assert result.fully_succeeded is False


def test_resume_rejects_path_traversal(fresh_state):
    """Defense-in-depth: resume_run must refuse storage_paths that
    resolve outside the project's user-optuna directory (tampered
    sidecar via Dropbox/OneDrive sync conflict, etc.).
    """
    rs, rp, tmp_path = fresh_state

    # Tampered sidecar pointing at an arbitrary file outside optuna dir.
    fake_target = tmp_path / "elsewhere" / "evil.sqlite3"
    fake_target.parent.mkdir(parents=True, exist_ok=True)
    fake_target.touch()

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({
            "run_id": "tampered",
            "storage_path": str(fake_target),
            "storage_url": f"sqlite:///{fake_target.as_posix()}",
            "label": "x",
            "dataset_fingerprint": None,
            "model_names": [],
            "n_trials_per_model": None,
            "started_iso": "2026-01-01T00:00:00",
        }),
        encoding="utf-8",
    )

    assert rs.resume_run("tampered") is None
    # Sidecar should remain — auto-discard would help an attacker
    # by silently cleaning up evidence of tampering.
    assert sidecar.exists()


def test_discard_rejects_storage_path_traversal(fresh_state):
    """Defense-in-depth: discard must refuse to unlink a storage_path
    that resolves outside the project's user-optuna directory.
    """
    rs, rp, tmp_path = fresh_state

    fake_target = tmp_path / "elsewhere" / "evil.sqlite3"
    fake_target.parent.mkdir(parents=True, exist_ok=True)
    fake_target.touch()

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({
            "run_id": "tampered",
            "storage_path": str(fake_target),
            "storage_url": f"sqlite:///{fake_target.as_posix()}",
            "label": "x",
            "dataset_fingerprint": None,
            "model_names": [],
            "n_trials_per_model": None,
            "started_iso": "2026-01-01T00:00:00",
        }),
        encoding="utf-8",
    )

    result = rs.discard_incomplete_run("tampered")
    # Sidecar deletion is fine (it's in our dir). Storage deletion is refused.
    assert result.sidecar_deleted is True
    assert result.storage_deleted is False
    assert any("outside optuna dir" in e for e in result.errors)
    # Off-path file must NOT have been deleted.
    assert fake_target.exists()


def test_t44_no_phantom_hasattr_typos_in_gui():
    """T-44: hasattr-guarded reads of phantom attribute names silently
    short-circuit to the fallback path, with no AttributeError to alert
    a developer. Two known phantoms surfaced during T-44 + DeepSeek
    sibling-survey:

    - n_trials_var (actual: n_unified_trials) — silently zeroes
      RunMetadata.n_trials_per_model in every sidecar.
    - task_type_var (actual: task_type) — silently falls through to
      inference-from-y in _save_selected_ensemble; 'auto' radio value
      can't be reproduced by inference, and y=None falls all the way
      to the 'regression' default regardless of the actual radio.

    Source-text regression catches both. Class-of-bug pin (any rename
    that re-introduces the wrong name in any future call site fails)."""
    gui_src = (
        Path(__file__).parent.parent / "spectral_predict_gui_optimized.py"
    ).read_text(encoding="utf-8")
    phantoms = ["n_trials_var", "task_type_var"]
    found = [name for name in phantoms if name in gui_src]
    assert not found, (
        f"T-44 regression: phantom-hasattr typo(s) re-introduced: {found}. "
        f"These attribute names don't exist on SpectralPredictApp; the "
        f"hasattr guard silently skips them. Use the actual Tk var names: "
        f"n_unified_trials (NOT n_trials_var), task_type (NOT task_type_var)."
    )
