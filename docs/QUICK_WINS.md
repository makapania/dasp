# Spectral Predict — Quick Wins Audit

> Generated: 2026-04-29
> Scope: `spectral_predict_gui_optimized.py`, `search.py`, `unified_bayesian.py`, `report.py`, `models.py`, and supporting modules
> Goal: High-impact, low-effort fixes to move Commercial Readiness from 4/10 toward 8/10

---

## Priority Matrix

| Priority | Category | File(s) | Issue | Est. Fix | Commercial Impact |
|----------|----------|---------|-------|----------|-------------------|
| **P0** | Debug Hygiene | `search.py` | `[PLS-DA DEBUG]` and `[DEBUG]` prints scattered at lines 2313, 2789, 2834, 3091, 3093, 3098, 3100, 3991 | 5 min | **HIGH** — looks amateur in front of customers |
| **P0** | Debug Hygiene | `calibration_transfer.py` | `=== CTAI Debug Information ===` print flood starting line 755; ~66 total prints; no logger | 15 min | **HIGH** — console spam during transfers |
| **P0** | Debug Hygiene | `nsga2_search.py` | 42 prints and 29 bare `except Exception:` blocks | 20 min | **HIGH** — hides real errors, floods logs |
| **P1** | Version Consistency | `__init__.py`, `model_io.py`, GUI | Hardcoded `0.5.0b1` in 3 places; GUI title and BETA badge are stale | 10 min | **MED-HIGH** — customers see "beta" everywhere |
| **P1** | Code Correctness | `models.py` | VIP formula uses `np.var(y)` (line ~1738) instead of per-component explained variance; imported `logging` is unused | 15 min | **MED-HIGH** — VIP scores are wrong for variable selection |
| **P1** | Error Handling | `scoring.py`, `regions.py`, `search.py` | Bare `except:` blocks at lines 509, 517, 535 (scoring), line 69 (regions), line 4637 (search) | 10 min | **MED** — silent failures mask real bugs |
| **P2** | Dead Code | `interference.py` | Duplicate `EPO` class — placeholder at line 564 ("TO BE IMPLEMENTED IN PHASE 2") and real implementation at line 820 | 5 min | **MED** — confusing to maintainers |
| **P2** | Config Hygiene | `code_generator.py` | Hardcodes `'version': '3.9.0'` at line 373 | 5 min | **MED** — generated scripts claim wrong Python version |
| **P2** | Branding/Trust | `export_bundle.py` | Placeholder citation URL `https://github.com/yourusername/spectral-predict` at line 244 | 5 min | **MED** — looks unprofessional in exported bundles |
| **P2** | Documentation Drift | `templates/header.py`, `baseline.py`, `preprocess.py`, `variable_selection.py` | Header claims "Spectral Predict v3"; ALS `p` default mismatch (0.001 vs 0.01); UVE docstring is self-contradictory | 15 min | **LOW-MED** — erodes trust in generated scripts |
| **P3** | API Polish | `models.py`, `search.py`, `ensemble.py`, `scoring.py`, etc. | Missing `__all__` in ~10 public modules | 20 min | **LOW-MED** — affects IDE autocomplete and `from x import *` |
| **P3** | Debug Hygiene | `ensemble.py` | Diagnostic env-var prints (`SPECTRAL_PREDICT_DEBUG`) at lines 1871-1893 | 5 min | **LOW** — harmless but unprofessional |
| **P3** | Logging | `unified_bayesian.py` | Verbose trial progress prints (lines 1748-1773, 1882, 1911-1943) bypass existing logger | 10 min | **LOW** — logger exists, prints don't respect log level |

---

## Detailed Breakdown

### P0 — Debug Print Cleanup (15 min total)

**`search.py`**
- Line 2313: `print(f"[DEBUG] Validation config: ...")`
- Line 2789: `print(f"[DEBUG] Running {name}...")`
- Line 2834: `print(f"[DEBUG] {name} completed...")`
- Lines 3091, 3093, 3098, 3100: Metrics debug prints
- Line 3991: `print("[PLS-DA DEBUG] classes:", ...)`
- Fix: Replace with `logger.debug(...)` or delete outright.

**`calibration_transfer.py`**
- Lines 755+: `print("=== CTAI Debug Information ===")` and ~65 other prints
- No logger is imported; add `import logging` and `logger = logging.getLogger(__name__)` at top
- Convert all prints to `logger.info/debug`

**`nsga2_search.py`**
- 42 `print()` statements
- 29 `except Exception:` blocks (some may need narrowing)
- Fix: Replace prints with logger; at minimum add `logger = logging.getLogger(__name__)`

### P1 — Version & Branding Consistency (10 min)

**Current state:**
- `src/spectral_predict/__init__.py:3`: `__version__ = '0.5.0b1'`
- `src/spectral_predict/model_io.py:48`: Same hardcoded string
- `spectral_predict_gui_optimized.py:2541`: Title bar says "Spectral Predict v3 (Beta)"
- `spectral_predict_gui_optimized.py:4625-4636`: BETA badge label

**Fix:**
1. Define `__version__` in ONE place (`__init__.py`)
2. Import it in `model_io.py` and `report.py`
3. Update GUI title to read from `__version__` dynamically
4. Remove or conditionalize BETA badge (e.g., only if version contains 'b' or 'beta')

### P1 — VIP Formula Bug (15 min)

**File:** `src/spectral_predict/models.py` (~line 1738)

**Issue:** The VIP implementation approximates the denominator with `np.var(y)` instead of the sum of explained variance per component. This is a known incorrect shortcut.

**Impact:** VIP scores drive variable selection decisions; wrong scores = wrong variables selected.

**Fix:** Implement the standard VIP formula:
```python
vip = np.sqrt(p * np.sum(ss * w**2, axis=1) / np.sum(ss))
```
where `ss = explained_variance_per_component`.

### P1 — Bare Except Blocks (10 min)

**Locations:**
- `scoring.py:509`, `scoring.py:517`, `scoring.py:535`
- `regions.py:69`
- `search.py:4637`

**Fix:** Change `except:` to `except Exception:` (minimum) or catch the specific expected exception. Add logging inside the block so failures aren't silent.

### P2 — Dead/Placeholder Code (5-15 min)

**`interference.py`**
- Lines 564-600: Placeholder `EPO` class with `raise NotImplementedError(...)` and comment "TO BE IMPLEMENTED IN PHASE 2"
- Lines 820+: Real `EPO` class
- Fix: Delete the placeholder.

**`code_generator.py:373`**
- `PYTHON_VERSION = '3.9.0'` hardcoded
- Fix: Use `f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"` or at least `3.x`.

**`export_bundle.py:244`**
- `https://github.com/yourusername/spectral-predict`
- Fix: Replace with real repo URL or company homepage.

### P2 — Documentation Drift (15 min)

**`templates/header.py`**
- Claims "Spectral Predict v3" but `__init__.py` says `0.5.0b1`
- Fix: Make header template interpolate actual version string.

**`baseline.py:139` vs `preprocess.py:482`**
- `BaselineALS` default `p=0.001` in `baseline.py`
- `apply_baseline_als` default `p=0.01` in `preprocess.py`
- Fix: Unify to one value (0.001 is more common in chemometrics).

**`variable_selection.py:44`**
- Docstring says cutoff_multiplier is "typically 0.5-1.0" but also says "cutoff = mean(abs) + multiplier * std" — if multiplier is < 1, the cutoff is close to the mean, which is contradictory for "noise threshold"
- Fix: Clarify docstring; standard UVE cutoff is mean + 0.5-1.0 * std, but the text should explain that higher multiplier = fewer variables kept.

### P3 — Missing `__all__` Exports (20 min)

**Modules needing `__all__`:**
- `models.py` (exports ~20 public classes)
- `search.py` (main API)
- `ensemble.py`
- `scoring.py`
- `baseline.py`
- `preprocess.py`
- `variable_selection.py`
- `contamination.py`
- `io.py`
- `calibration_transfer.py`

**Fix:** Add `__all__ = ['Class1', 'Class2', 'function1', ...]` to each.

### P3 — Logger Bypass in Bayesian (10 min)

**`unified_bayesian.py`**
- Lines 1748-1773, 1882, 1911-1943: `print(f"Trial {trial.number}...")`, `print(f"Best so far: ...")`
- Logger already imported at top of file
- Fix: Replace prints with `logger.info(...)` or add a `verbose` flag so prints only happen when requested.

---

## Recommended Execution Order (for maximum impact per minute)

1. **Delete debug prints in `search.py`** (5 min) — highest visibility
2. **Replace prints with logger in `calibration_transfer.py`** (15 min) — stops console spam
3. **Fix `__version__` in one place and wire to GUI** (10 min) — professional polish
4. **Fix VIP formula in `models.py`** (15 min) — correctness
5. **Delete placeholder EPO in `interference.py`** (2 min) — dead code
6. **Fix bare except blocks** (10 min) — reliability
7. **Update `code_generator.py` and `export_bundle.py` placeholders** (10 min) — generated artifacts look professional
8. **Add `__all__` to top 3-4 most imported modules** (15 min) — API polish

**Total estimated time: ~80 minutes** for all P0-P2 items.

---

## Files Modified During This Audit

- `docs/QUICK_WINS.md` (this file)
