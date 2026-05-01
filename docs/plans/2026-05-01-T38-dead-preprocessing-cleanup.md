# T-38: Dead Preprocessing Module Cleanup (Sketch)

> **Status:** ROUGH PLAN — bundle with T-37 merge or do as a follow-up cleanup PR after T-37 ships.

**Goal:** Remove three preprocessing modules that are zero-call-site dead code, and retire `ga_preprocessing.py` after T-37 absorbs its useful insights.

**Source:** Conversation 2026-04-30. Confirmed by independent audits (Explore agent + DeepSeek V4 Pro): three modules are imported nowhere or imported but never called functionally.

---

## Modules to delete

### 1. `src/spectral_predict/learned_preprocessing.py` (775 lines)

- **Purpose:** PyTorch CNN-based learnable preprocessing (InstanceNorm + Conv1d + ReLU) that replaces fixed SNV/Savitzky-Golay with end-to-end trained layers.
- **Usage:** Zero. Not imported anywhere in `spectral_predict_gui_optimized.py` or `src/spectral_predict/search.py`. CLAUDE.md explicitly notes "not wired into the GUI." Import block was removed 2026-04-17.
- **Tests:** None.
- **Action:** Delete file. Document deletion in `docs/SESSION_LOG.md`. If PyTorch preprocessing is later prioritized, design fresh.

### 2. `src/spectral_predict/ensemble_preprocessing.py` (701 lines)

- **Purpose:** Stacked-preprocessing ensemble — `StackedPreprocessingRegressor` / `StackedPreprocessingClassifier` train same base model on multiple preprocessed views, combine via Ridge meta-model.
- **Usage:** Imported at `gui:237` to set `HAS_ENSEMBLE_PREPROCESSING = True`, but the flag is never read in any functional code path. Not in `search.py`.
- **Tests:** Has `tests/test_ensemble_preprocessing.py` — also delete.
- **Action:** Delete `ensemble_preprocessing.py`, delete `tests/test_ensemble_preprocessing.py`, remove the dead `HAS_ENSEMBLE_PREPROCESSING` import block from GUI.

### 3. `src/spectral_predict/preprocessing_wrapper.py` (220 lines)

- **Purpose:** Thin stateless transformer (`PreprocessorConfig`) to reconstruct preprocessing from stored config dict instead of fitted objects.
- **Usage:** Only imported by `ensemble_preprocessing.py` (itself dead). Doubly orphaned.
- **Tests:** None.
- **Action:** Delete file (after ensemble_preprocessing is deleted).

## Module to retire after T-37

### 4. `src/spectral_predict/ga_preprocessing.py` (1814 lines)

- **Purpose:** GA + exhaustive + smart-2-stage preprocessing search.
- **Usage:** Active — called from `search.py:1362` when `smart_preprocess=False`. Has GUI toggle.
- **Replacement:** T-37 TPE quick-discovery subsumes the search functionality. Two valuable pieces port to T-37:
  - `DERIVATIVE_WINDOW_RANGES` (sensible window-per-derivative-order priors) at `:75-80`
  - Multi-seed robustness logic (smart-mode 2-stage validation pattern)
- **Action after T-37 ships:**
  - Delete `ga_preprocessing.py`.
  - Delete `tests/test_ga_preprocessing.py` (or migrate any preprocessing-agnostic tests to the T-37 test suite).
  - Remove the `ga_preprocess` toggle from the GUI (Tab 4A).
  - Remove the `optimize_preprocessing` import + call from `search.py:88, :1362-1489`.

---

## Tasks

1. Verify zero-call-site claims one more time via fresh `grep` before deletion (modules may have been re-wired since this audit).
2. Delete the three dead modules + their tests in a single commit.
3. Remove the dead `HAS_ENSEMBLE_PREPROCESSING` flag from GUI.
4. Run full test sweep to confirm no regressions.
5. After T-37 ships: separate commit retiring `ga_preprocessing.py`.
6. Note in `docs/SESSION_LOG.md` and PROJECT_STATUS.md.

---

## Risk assessment

- **Reversibility:** Full — deleted modules live in git history; restore via `git checkout` if needed.
- **Coupling:** Verified zero functional couplings (just one dead-flag import).
- **User impact:** Zero — these features were never reachable from the GUI.
- **Test coverage:** The dead modules' tests are also removed; this reduces test suite size by ~1500 lines but covers nothing reachable.

---

## Sequencing

- **Blocked on:** Nothing for the three dead-module deletions (could ship today).
- **Blocked on:** T-37 ship for the `ga_preprocessing.py` retirement.
- **Recommended:** Do the three dead-module deletions as a small standalone PR; retire `ga_preprocessing.py` as part of T-37's PR or immediately after.
