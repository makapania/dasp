# Spectral Predict — Bundled App Build Guide (Python 3.12)

Build a standalone Spectral Predict bundle + Windows installer using Python 3.12 + PyInstaller 6.x. This is the **shipped** build path as of `0.5.0b1`. The Python 3.11 path was retired on 2026-04-21; its files are preserved at `archive/build_3_11/` (see that folder's README) for historical / reproducibility purposes only.

---

## Quick Start

```bash
.venv312\Scripts\python.exe build_installer_py312.py
```

This single command handles everything:
1. Icon creation (`asp_logo.ico`)
2. PyInstaller build using `spectral_predict_py312.spec`
3. Post-build torch cleanup (saves ~255 MB)
4. Pandas self-repair (fixes a known PyInstaller TOC corruption — see below)
5. Inno Setup compilation (if ISCC.exe is installed)

**Output:**
- `dist/SpectralPredict-py312/SpectralPredict-py312.exe` — standalone bundle (~1.4 GB folder)
- `dist/installer/SpectralPredict_Setup_py312_0.5.0b1.exe` — single-file installer (~299 MB, LZMA2-compressed)

---

## Prerequisites

### Python 3.12 venv (`.venv312/`)

The build script uses `.venv312/` which must exist with all project dependencies + PyInstaller:

```bash
# First-time setup (or after deleting .venv312):
install.bat                                          # creates .venv312, runs `pip install -e .`
.venv312\Scripts\pip install pyinstaller             # add the build tool
```

The `install.bat` script (in repo root) handles Python 3.12 detection and venv creation. See `INSTALL.md` for the user-facing setup walkthrough.

### Optional: Inno Setup 6 (Windows installer)

Download from <https://jrsoftware.org/isdl.php>. The build script auto-detects `ISCC.exe` at the standard install paths. If absent, the build still produces the `dist/SpectralPredict-py312/` folder; only the single-file installer is skipped.

---

## How It Works

`build_installer_py312.py` runs five steps:

1. **Icon creation** — Converts `asp_logo_final.png` to `asp_logo.ico` using Pillow.
2. **PyInstaller build** — Invokes PyInstaller from `.venv312` against `spectral_predict_py312.spec`.
3. **Post-build torch cleanup** — `shutil.rmtree` removes `_internal/torch/`, `torchvision/`, `torchaudio/`. The `learned_preprocessing` module imports torch but is dead code in the GUI; bundling torch would add ~800 MB for no benefit. Filtering inside the spec causes PyInstaller TOC corruption (see Troubleshooting), so cleanup is post-COLLECT.
4. **Pandas self-repair** — Byte-compares `_internal/pandas/util/__init__.py` against the venv version and restores when mismatched. PyInstaller 6.18 occasionally overwrites this file with `packaging/_structures.py` content (intermittent, ~1 in 3 rebuilds). See Troubleshooting.
5. **Inno Setup** — If `ISCC.exe` is found, runs `installer/spectral_predict_py312.iss` to produce the installer .exe.

(The 3.11 spec previously coexisted in parallel; it is now in `archive/build_3_11/` and no longer built.)

---

## Build Files

| File | Purpose |
|------|---------|
| `build_installer_py312.py` | Build orchestrator |
| `spectral_predict_py312.spec` | PyInstaller spec for Python 3.12 |
| `installer/spectral_predict_py312.iss` | Inno Setup script |
| `asp_logo_final.png` | Source icon (auto-converted) |
| `asp_logo.ico` | Generated Windows icon |

The 3.11 build path was archived on 2026-04-21 to `archive/build_3_11/` (specs, build script, .iss, and old build guide). Restore via `git mv archive/build_3_11/<file> <original-path>` if you ever need to reproduce an old release.

---

## Multiprocessing Behavior in the Bundle

**The bundle uses joblib's `threading` backend, not `loky`.** The reason:

- `loky`'s `spawn` method re-executes the frozen .exe to start child processes
- The child runs `multiprocessing.freeze_support()` in PyInstaller's runtime hook (`pyi_rth_multiprocessing.py:43`)
- That hook crashes on argv parsing: `ValueError: not enough values to unpack (expected 2, got 1)`
- The parent retries spawning → fork-bomb of GUI windows

The fallback is gated by `_frozen_needs_threading_fallback()` in `src/spectral_predict/search.py:9` which returns `True` for any frozen build. The GUI entry point also wraps `freeze_support()` in `try/except sys.exit(0)` as defense in depth.

**Practical impact:** CPU-bound grid search in the bundle uses threads (constrained by Python's GIL for pure-Python work, but native code in xgboost/lightgbm/catboost releases the GIL so it still parallelizes effectively). Pure sklearn fits don't see the speedup that loky multiprocessing would give.

---

## Troubleshooting

### `ImportError: cannot import name 'capitalize_first_letter' from 'pandas.util'`

**Cause:** PyInstaller 6.18 dist-info collection collision overwrites `_internal/pandas/util/__init__.py` with vendored `packaging/_structures.py` content. The bundled file starts with `# Vendored from https://github.com/pypa/packaging/...`.

**Fix:** The build script's pandas self-repair (step 4) handles this automatically — look for `[REPAIR] Restored pandas/util/__init__.py` in the build output. If you see this message, the workaround triggered. If you somehow get a corrupted bundle without the repair triggering, manually run:

```bash
cp .venv312/Lib/site-packages/pandas/util/__init__.py dist/SpectralPredict-py312/_internal/pandas/util/__init__.py
```

If other files start exhibiting the same corruption pattern, add them to the `repair_targets` list in `build_installer_py312.py` near line 225.

### `PermissionError: [WinError 5] Access is denied: '...SpectralPredict-py312.exe'`

**Cause:** A previous bundle is still running (locks files). Most often the GUI you launched earlier in the session.

**Fix:** Close the running GUI before rebuilding. Check with:
```bash
tasklist | findstr SpectralPredict
```

### `ModuleNotFoundError: No module named 'X'` in the bundled exe

**Fix:** Add `X` to `explicit_hiddenimports` in `spectral_predict_py312.spec`, or to `packages_to_collect` if it has data files / submodules. Then rebuild.

### Brief cmd.exe console window flashes when starting an analysis

**Cause:** joblib/loky probes the CPU count via subprocess on first use. In a windowed PyInstaller bundle, any subprocess flashes a console window briefly.

**Fix:** Already handled — the GUI sets `LOKY_MAX_CPU_COUNT` at startup (frozen-only) so loky skips the probe. If you still see a flash, add other `LOKY_*` env vars at the same location in `spectral_predict_gui_optimized.py`.

### catboost / xgboost / lightgbm DLL load failures

The Python 3.11 build had this class of bug. **The 3.12 build does not** — verified across rebuilds. If you somehow hit a DLL error, check that the bundle includes the relevant `.libs/` or `lib/` subdirectory under `_internal/`.

---

## Verification

```bash
# Quick smoke test (31 import checks, runs under the bundled Python)
dist\SpectralPredict-py312\SpectralPredict-py312.exe --test
```

Expect `[SUCCESS] ALL TESTS PASSED!` and `Imports: 31/31`. If anything fails, see the troubleshooting section above for the specific module.

For workflow testing (CV runs, model training, plot rendering), launch the GUI directly:
```bash
dist\SpectralPredict-py312\SpectralPredict-py312.exe
```

---

## Maintenance

### Adding a new dependency

1. Add to `pyproject.toml`
2. Install in `.venv312`: `.venv312\Scripts\pip install -e . --upgrade`
3. Add to `packages_to_collect` in `spectral_predict_py312.spec` if it has data files or compiled submodules
4. Rebuild and verify with `--test`

### Changing the GUI entry script

If renaming `spectral_predict_gui_optimized.py`, update both:
- `spectral_predict_py312.spec` line 234 (the `Analysis(...)` first argument)
- `build_installer_py312.py` `main_script` variable

---

*Version: 0.5.0b1 | Last updated: 2026-04-21*
