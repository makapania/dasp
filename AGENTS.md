# Repository Guidelines

## Project Structure & Module Organization
Source code lives under `src/` (Python packages) and the GUI entry points (`spectral_predict_gui_optimized.py`, `run_full_analysis.py`, etc.) at the repo root. Tests reside in `tests/` plus a handful of legacy regression scripts in the root (e.g., `test_tab7_automated_full.py`). Example datasets and quick-start assets live in `example/`, while generated artifacts belong in `outputs/` or `reports/`. Keep bulky diagnostics (CV logs, ASD exports) out of Git—drop them into `outputs/` and reference them from Markdown reports instead of committing binaries.

## Build, Test, and Development Commands
- Create a virtualenv, install editable deps, and include GUI extras:  
  `python3 -m venv .venv && source .venv/bin/activate && pip install -e .[dev] specdal`
- Smoke the CLI:  
  `spectral-predict --asd-dir example/BoneCollagen --reference example/BoneCollagen_reference.csv --id-column "File Number" --target "%Collagen" --gui`
- GUI runners (e.g., `spectral_predict_gui_optimized.py`, `run_full_analysis.py`) assume the repo root as CWD.
- Quality gates before any PR: `python3 -m pytest -q`, targeted regressions such as `python3 -m pytest tests/test_tab7_state.py`, plus `black src tests` and `flake8 src tests`.

## Coding Style & Naming Conventions
Code is formatted with Black (line length 100). Use `snake_case` for functions/modules, `CapWords` for classes, and explicit type hints in public APIs. GUI widgets should have intent-revealing names (`tab7_mode_label`, `refine_run_button`). CLI flags and config keys should match the existing naming in README examples (`--lambda-penalty`, `Deriv`, `SubsetTag`). Prefer short docstrings/comments that explain *why* unusual logic exists (e.g., wavelength order preservation) rather than restating the code.

## Testing Guidelines
Pytest discovery targets `tests/` via `pyproject.toml` (`python_files = "test_*.py"`, `python_functions = "test_*"`). Place new regression tests alongside the feature they guard (e.g., Tab 7 headless tests inside `tests/`). For GUI-related checks, mark them `@pytest.mark.gui` and provide fallbacks/headless shims. Keep fixtures lightweight—reuse `example/` assets or generate synthetic data on the fly. Always capture key metrics (R², RMSE, accuracy) in test failures so investigations can start from the console log.

## Commit & Pull Request Guidelines
Follow the Conventional Commit format (`fix:`, `feat:`, `docs:`) and include PROBLEM / ROOT CAUSE / FIX / VERIFICATION bullet blocks in commit messages or PR descriptions. Update any relevant guides (`CODEX_HANDOFF.md`, `HANDOFF_R2_MISMATCH_INVESTIGATION_2025_11_07.md`, `README.md`) whenever behavior changes. PRs should link to tracking issues, list the exact commands used for validation (`pytest …`, `TAB7_TRACE=1 …`), and include screenshots or trace snippets when touching the GUI. Avoid force-pushes after review unless you explicitly coordinate with the reviewer.*** End Patch
