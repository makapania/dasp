import subprocess, sys

def test_cli_help_runs():
    # Ensure CLI help prints without error
    cmd = [sys.executable, "-m", "spectral_predict.cli", "-h"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0
    assert "Spectral Predict" in (cp.stdout + cp.stderr)
