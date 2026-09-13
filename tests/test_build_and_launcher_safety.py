"""Build verification and launcher failures must stop unsafe startup."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("missing", [None, "source", "bundle"])
def test_build_repairs_pandas_in_paths_with_spaces(tmp_path, monkeypatch, capsys, missing):
    spec = importlib.util.spec_from_file_location("build_script", REPO / "build_installer_py312.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    root = tmp_path / "build checkout with spaces"
    assert root.resolve().is_relative_to(tmp_path.resolve())
    venv = root / ".venv314"
    python = venv / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (root / "spectral_predict_py312.spec").touch()
    (root / "spectral_predict_gui_optimized.py").touch()
    site_packages = venv / "Lib" / "site-packages"
    relative_pandas = Path("pandas") / "util" / "__init__.py"
    pandas_source = site_packages / relative_pandas
    if missing != "source":
        pandas_source.parent.mkdir(parents=True)
        pandas_source.write_bytes(b"# correct pandas module\n")

    bundle = root / "dist" / builder.APP_NAME
    pandas_bundle = bundle / "_internal" / relative_pandas
    monkeypatch.setattr(builder, "PROJECT_ROOT", root)
    monkeypatch.setattr(builder, "DIST_DIR", root / "dist")
    monkeypatch.setattr(builder, "BUILD_VENV", venv)
    monkeypatch.setattr(builder, "IS_WINDOWS", True)
    monkeypatch.setattr(builder, "IS_MACOS", False)

    def fake_build(command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, stdout=f"314\n{site_packages}\n")
        assert command[1:3] == ["-m", "PyInstaller"]
        internal = bundle / "_internal"
        internal.mkdir(parents=True)
        (bundle / f"{builder.APP_NAME}.exe").touch()
        (internal / "python314.dll").touch()
        (internal / "python3.dll").touch()
        if missing != "bundle":
            pandas_bundle.parent.mkdir(parents=True)
            pandas_bundle.write_bytes(b"# wrong module from TOC collision\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builder.subprocess, "run", fake_build)
    succeeded = builder.run_pyinstaller()
    output = capsys.readouterr().out
    if missing is None:
        assert succeeded
        assert pandas_bundle.read_bytes() == pandas_source.read_bytes()
        assert "[REPAIR]" in output
    else:
        assert not succeeded
        assert "ERROR" in output
        assert "match venv" not in output


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises cmd.exe batch behavior")
@pytest.mark.parametrize(
    "check_rc,lock_rc,editable_rc,expected,returncode",
    [
        (1, 1, 0, ["check", "lock"], 1),
        (1, 0, 1, ["check", "lock", "editable"], 1),
        (1, 0, 0, ["check", "lock", "editable", "gui"], 0),
        (0, 0, 0, ["check", "gui"], 0),
    ],
    ids=["lock-failed", "editable-failed", "installed", "already-present"],
)
def test_launcher_checks_each_install_result(
    tmp_path, check_rc, lock_rc, editable_rc, expected, returncode
):
    python = r".venv314\Scripts\python.exe"
    source = (REPO / "RUN_SPECTRAL_PREDICT.bat").read_text(encoding="utf-8")
    guard = f'if not exist "{python}"'
    assert source.count(guard) == 1
    source = source.replace(guard, 'if not exist "%~dp0stub.cmd"')
    assert source.count(python) == 4, "Every Python call must be stubbed; never run pip here"
    # Explicit paths: bare names fail under NoDefaultCurrentDirectoryInExePath=1.
    (tmp_path / "launcher.bat").write_text(source.replace(python, 'call "%~dp0stub.cmd"'))
    (tmp_path / "stub.cmd").write_text(
        "@echo off\n"
        'if "%~1"=="-c" (\n'
        " echo PROBE check\n"
        " exit /b %PR65_CHECK_RC%\n"
        ")\n"
        'if "%~5"=="-r" (\n'
        " echo PROBE lock\n"
        " exit /b %PR65_LOCK_RC%\n"
        ")\n"
        'if "%~5"=="-e" (\n'
        " echo PROBE editable\n"
        " exit /b %PR65_EDITABLE_RC%\n"
        ")\n"
        "echo PROBE gui\n"
        "exit /b 0\n"
    )
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(tmp_path / "launcher.bat")],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PR65_CHECK_RC=str(check_rc),
            PR65_LOCK_RC=str(lock_rc),
            PR65_EDITABLE_RC=str(editable_rc),
        ),
        input="\n",
        text=True,
        capture_output=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    trace = [
        line.removeprefix("PROBE ")
        for line in result.stdout.splitlines()
        if line.startswith("PROBE ")
    ]
    assert trace == expected, result.stdout + result.stderr
    assert result.returncode == returncode


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises cmd.exe batch behavior")
def test_launcher_without_venv_points_to_installer(tmp_path):
    source = (REPO / "RUN_SPECTRAL_PREDICT.bat").read_text(encoding="utf-8")
    (tmp_path / "launcher.bat").write_text(source)
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(tmp_path / "launcher.bat")],
        cwd=tmp_path,
        input="\n",
        text=True,
        capture_output=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 1
    assert "install.bat" in result.stdout
    assert "Failed to install" not in result.stdout
