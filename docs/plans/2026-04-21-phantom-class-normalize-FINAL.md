# Phantom-class-drop + 91-vs-94 Refine mismatch — Final plan

**Status:** Approved for implementation 2026-04-21
**Root cause:** `astype(str)` in the mixed-type coercion added earlier today splits numerically-equivalent labels (`250` int vs `250.0` float vs `"250"` str vs `"250 "` whitespace) into distinct string classes. The rare-class auto-drop at `gui:26042-26070` then drops the phantom minority class. Refine tab coerces but does not replicate the drop, so `_validate_training_configuration` compares saved 91 vs current 94 and warns.

**User's observation:** all "250" cells look identical in Excel. Most likely explanation: 3 cells were Text-formatted (or had apostrophe prefix, or trailing whitespace) before values were entered — invisible in Excel but stored as `str` while others are `int`. Our `astype(str)` preserves those differences rather than normalizing.

## Fix: Option A — numeric-aware label normalization

Replace naive `astype(str)` with a helper that collapses numerically-equivalent values into a canonical string.

### New helper

```python
def _normalize_mixed_type_labels(series: pd.Series) -> pd.Series:
    """Normalize mixed-type class labels so numeric-equivalent values collapse.

    Examples:
        pd.Series([250, 250.0, "250", "250.0", "250 ", np.nan]) → ["250","250","250","250","250",NaN]
    """
    def _norm(v):
        if pd.isna(v):
            return v
        if isinstance(v, str):
            v = v.strip()
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except (ValueError, TypeError):
            return str(v)
    return series.apply(_norm)
```

### Sites to patch

**Primary (bug trigger):**
- `spectral_predict_gui_optimized.py:26033` — `_run_analysis_thread` coercion (auto-drop reads this result)
- `spectral_predict_gui_optimized.py:35285` — `_run_refined_model_thread` coercion

**Validation-wide sweep (all sites from commit `382b7e6`):**
| # | Location | Context |
|---|---|---|
| 1 | `gui:19531` | `_validation_spxy` |
| 2 | `gui:19633` | `_validation_stratified` stratify |
| 3 | `gui:19767-19771` | `_create_validation_set` class distribution |
| 4 | `gui:26883-26887` | Bayesian validation metrics |
| 5 | `gui:27115` | NSGA-II validation metrics |
| 6 | `gui:37162` | Refine validation metrics |
| 7 | `search.py:454-458` | `compute_validation_metrics_for_top_models` |
| 8 | `gui:40306, 40310` | Prediction confusion matrix |
| 9 | `gui:40521` | Prediction scatter plot coloring |
| 10 | `gui:40691, 40709` | Prediction metrics display |

### Diagnostic log enhancement

At `_run_analysis_thread` coercion site (primary user-facing entry), extend the log line to print per-type distribution with a few sample values, AND the collapsed-labels list if normalization merges any. This helps the user understand what the 3 "different" cells actually are:

```python
from collections import Counter
by_type = Counter(type(v).__name__ for v in y_filtered.dropna())
self._log_progress(f"  [i] Target column has mixed Python types: {dict(by_type)}")

_before = set(y_filtered.dropna().astype(str).unique())
y_filtered = _normalize_mixed_type_labels(y_filtered)
_after = set(y_filtered.dropna().unique())
if _after != _before:
    _collapsed = sorted(_before - _after)
    self._log_progress(
        f"  [i] Normalized numeric-equivalent labels into canonical strings. "
        f"Collapsed labels: {_collapsed}."
    )
```

## Explicitly NOT doing

- **Option B** (replicate auto-drop in Refine): would silence the mismatch but still loses 3 real specimens for a spurious reason. Once Option A lands, Refine mismatch disappears automatically.
- **Migrate backend `search.py:975, 3232`**: pre-existing deferred debt; out of scope for this fix.
- **Touch regression / one-class numeric-target paths**: helper only runs where the existing `astype(str)` coercion runs (classification / one-class object-dtype with mixed types).

## Test plan

### Unit tests (`tests/test_mixed_type_target_coercion.py` — extend existing file)

| Test | Input | Expected |
|---|---|---|
| `test_int_float_string_collapse` | `[250, 250.0, "250", "250.0", np.nan]` | `["250","250","250","250",NaN]` |
| `test_non_integer_float_preserved` | `[250.5, "250.5", 250]` | `["250.5","250.5","250"]` |
| `test_scientific_notation_collapse` | `["2.5e2", 250, "250.0"]` | `["250","250","250"]` |
| `test_genuine_strings_unchanged` | `["grass", "tree", "grass"]` | same |
| `test_whitespace_stripped` | `[" 250 ", 250, "250\xa0"]` | all `"250"` (strip handles NBSP via pre-strip) |
| `test_bool_collapses_with_int` | `[True, 1, "1", "True"]` | `["1","1","1","True"]` |
| `test_empty_string_preserved` | `["", 0]` | `["","0"]` |
| `test_nan_preserved_not_stringified` | `[np.nan, 250]` | `[NaN, "250"]` |

### Integration tests

- `test_phantom_class_not_dropped`: synthetic y = 91 × int(250) + 3 × str("250"). After analysis kickoff: len == 94, no rare-class popup.
- `test_refine_mismatch_not_triggered`: same data; run analysis → refine round-trip → no mismatch warning from `_validate_training_configuration`.

## Risks / edge cases

- **Integers > 2⁵³** lose precision in `float()` (documented limitation; spectroscopy rarely has class labels this large).
- **`True` collapses with `1`**: acceptable for classification; document it.
- **NaN preservation**: helper explicitly guards `pd.isna(v)` before any string conversion; test asserts.
- **Non-breaking space `\xa0`**: `str.strip()` does handle NBSP. Verified.
- **Performance**: `Series.apply` is O(n); fine for spectral datasets (hundreds-thousands of rows).

## Implementation sequencing

Single commit:
1. Add helper (module-level in GUI, near other pure helpers, or in `src/spectral_predict/analysis_subset.py` for testability).
2. Replace `astype(str)` at 12 sites with `_normalize_mixed_type_labels(...)`.
3. Enhance diagnostic log at the primary Analysis-tab site.
4. 8 unit tests + 2 integration tests.
