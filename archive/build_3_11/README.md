# Archived: Python 3.11 build path

These files are the legacy Python 3.11 / PyInstaller build path for Spectral Predict. They were retired on 2026-04-21 in favor of the Python 3.12 build (`spectral_predict_py312.spec` + `build_installer_py312.py` at the repo root).

## Contents

| File | Role |
|---|---|
| `spectral_predict.spec` | PyInstaller spec for the Windows 3.11 bundle |
| `spectral_predict_mac.spec` | PyInstaller spec for the macOS 3.11 bundle |
| `build_installer.py` | Build orchestrator (PyInstaller + Inno Setup wrapper) |
| `installer/spectral_predict.iss` | Inno Setup script |
| `BUNDLED_APP_BUILD_GUIDE.md` | Build guide for the 3.11 path |
| `run_gui.bat` | One-click launcher pointing at `.venv311` |
| `run_v3.bat` | One-click launcher for the long-deprecated Dear PyGui V3 prototype |

## Why archived, not deleted

History — these files were the production build path through April 2026 and are referenced in old commits, build artifacts, and external documentation. Keeping them in-repo means `git log -p` can still show the full evolution and someone reproducing an old release can pull a tagged commit and build it. Restoring any single file is `git mv archive/build_3_11/<file> <original-path>`.

## What replaced them

- Build script: `build_installer_py312.py` at repo root
- Spec: `spectral_predict_py312.spec` at repo root
- Inno Setup: `installer/spectral_predict_py312.iss`
- Build guide: `docs/BUNDLED_APP_BUILD_GUIDE_PY312.md`
- Dev launcher: `RUN_SPECTRAL_PREDICT.bat` (uses `.venv312`)

The `_py312` suffix on the live files is now historical baggage from the parallel-development period — safe to drop in a future cleanup.
