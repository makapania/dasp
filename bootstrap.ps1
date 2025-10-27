python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest -q
python -m spectral_predict.cli -h
Write-Host "Bootstrap complete."
