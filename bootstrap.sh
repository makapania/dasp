#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q || true
spectral-predict -h
echo "Bootstrap complete."
