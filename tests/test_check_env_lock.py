"""Tests for scripts/check_env_lock.py, which lets launchers apply a changed lockfile."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_env_lock.py"

_spec = importlib.util.spec_from_file_location("check_env_lock", SCRIPT)
check_env_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_env_lock)


def _fake_installed(versions: dict[str, str]):
    def installed_version(name: str) -> str:
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    return installed_version


def test_parse_lock_skips_comments_and_normalizes_names():
    text = "# header\n\nJcamp==1.3.2\nspecio_py310==0.1  # trailing\n"

    assert check_env_lock.parse_lock(text) == {"jcamp": "1.3.2", "specio-py310": "0.1"}


def test_parse_lock_rejects_non_exact_pin():
    with pytest.raises(ValueError, match="line 2"):
        check_env_lock.parse_lock("numpy==2.0\njcamp>=1.3.2\n")


def test_find_drift_reports_stale_version():
    """An environment built from the old lock (jcamp 1.2.2) must be flagged."""
    drift = check_env_lock.find_drift(
        {"jcamp": "1.3.2", "numpy": "2.3.0"},
        installed_version=_fake_installed({"jcamp": "1.2.2", "numpy": "2.3.0"}),
    )

    assert drift == [("jcamp", "1.3.2", "1.2.2")]


def test_find_drift_reports_missing_package():
    drift = check_env_lock.find_drift({"jcamp": "1.3.2"}, installed_version=_fake_installed({}))

    assert drift == [("jcamp", "1.3.2", None)]


def test_find_drift_empty_when_in_sync():
    pins = {"jcamp": "1.3.2"}

    assert check_env_lock.find_drift(pins, installed_version=_fake_installed(pins)) == []


def test_cli_exits_nonzero_on_drift(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("definitely-not-installed-package-xyz==9.9.9\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--lock", str(lock)], capture_output=True, text=True
    )

    assert result.returncode == check_env_lock.EXIT_DRIFTED
    assert "definitely-not-installed-package-xyz" in result.stdout


def test_cli_accepts_lockfile_with_bom(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("# header\ndefinitely-not-installed-package-xyz==9.9.9\n", encoding="utf-8-sig")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--lock", str(lock)], capture_output=True, text=True
    )

    assert result.returncode == check_env_lock.EXIT_DRIFTED


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_read_lock_text_decodes_powershell_encodings(tmp_path, encoding):
    lock = tmp_path / "lock.txt"
    lock.write_text("jcamp==1.3.2\n", encoding=encoding)

    assert check_env_lock.parse_lock(check_env_lock.read_lock_text(lock)) == {"jcamp": "1.3.2"}


def test_repository_lockfile_parses():
    pins = check_env_lock.parse_lock((REPO / "requirements-lock.txt").read_text(encoding="utf-8"))

    assert "jcamp" in pins
