# Spectral Predict - Bundled App Build Guide

Build a standalone Spectral Predict executable for Windows or macOS.

---

## Quick Start

```bash
python build_installer.py
```

This single command handles everything: icon creation, PyInstaller build, and platform-specific packaging (Inno Setup on Windows, codesign on macOS).

**Output:**
- Windows: `dist/SpectralPredict/SpectralPredict.exe` + optional Inno Setup installer
- macOS: `dist/SpectralPredict.app`

---

## Prerequisites

### Python 3.11 Virtual Environment (Required)

The build script uses `.venv311/` which must contain Python 3.11 with all dependencies:

```bash
python3.11 -m venv .venv311
.venv311/Scripts/pip install -e ".[dev]"   # Windows
.venv311/bin/pip install -e ".[dev]"       # macOS
pip install pyinstaller pillow
```

### Platform-Specific Requirements

| Platform | Required | Optional |
|----------|----------|----------|
| Windows | Python 3.11, PyInstaller, Pillow | [Inno Setup 6](https://jrsoftware.org/isdl.php) (for installer .exe) |
| macOS | Python 3.11, PyInstaller, Pillow | Xcode CLI tools (for `sips`, `iconutil`, `codesign`) |

---

## How It Works

`build_installer.py` runs three steps:

1. **Icon creation** — Converts `asp_logo_final.png` to `.ico` (Windows) or `.icns` (macOS) using Pillow or native tools
2. **PyInstaller build** — Uses `spectral_predict.spec` (Windows) or `spectral_predict_mac.spec` (macOS) to bundle the app
3. **Platform packaging** — Windows: runs Inno Setup if available; macOS: ad-hoc codesigns the .app bundle

The build script always uses `.venv311/` Python to ensure correct binary compatibility with scientific libraries (catboost, xgboost, etc.).

---

## Build Files

| File | Purpose |
|------|---------|
| `build_installer.py` | Cross-platform build orchestrator |
| `spectral_predict.spec` | PyInstaller spec for Windows |
| `spectral_predict_mac.spec` | PyInstaller spec for macOS |
| `asp_logo_final.png` | Source icon (auto-converted to .ico/.icns) |
| `asp_logo.ico` | Generated Windows icon |
| `installer/spectral_predict.iss` | Inno Setup installer script |

---

## Troubleshooting

### DLL Load Failed (catboost, xgboost, etc.)

**Cause:** Python version mismatch. Scientific packages have compiled extensions tied to the Python version.

**Fix:** Ensure `.venv311/` contains Python 3.11:
```bash
.venv311/Scripts/python.exe --version   # Should say 3.11.x
```

### Pipeline Not Fitted (in bundled exe only)

**Cause:** sklearn's `check_is_fitted()` behaves differently in PyInstaller bundles.

**Fix:** Already handled automatically by `_ensure_pipeline_fitted()` in `model_io.py`. Rebuild the bundle.

### Module Not Found

**Fix:** Add the module to `explicit_hiddenimports` in the spec file and rebuild.

### scipy.stats Errors

**Note:** The spec file sets `optimize=0` and `noarchive=True` to prevent scipy compatibility issues.

---

## Maintenance

### Adding New Dependencies

1. Install in `.venv311`
2. Add to `packages_to_collect` or `explicit_hiddenimports` in the spec file
3. Rebuild and test

### Adding New Custom Transformers

Ensure `fit()` method sets:
```python
self.n_features_in_ = X.shape[1]
self._is_fitted = True
```

### Changing Module Structure

1. Update `datas` section in the spec file
2. Rebuild and test

---

*Version: 0.4.0 | Last updated: 2026-02-15*
