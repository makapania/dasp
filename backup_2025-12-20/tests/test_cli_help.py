"""Test CLI help output."""

import subprocess
import sys


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectral_predict.cli", "-h"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "Spectral Predict" in result.stdout
    assert "--spectra" in result.stdout
    assert "--reference" in result.stdout
    assert "--target" in result.stdout


def test_cli_version():
    """Test that CLI version works."""
    result = subprocess.run(
        [sys.executable, "-m", "spectral_predict.cli", "--version"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "Spectral Predict" in result.stdout
