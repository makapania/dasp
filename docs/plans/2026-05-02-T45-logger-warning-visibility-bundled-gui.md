# T-45: `logger.warning` calls invisible in bundled GUI

**Status:** PLANNED — small architectural fix, ~30 LOC.
**Filed:** 2026-05-02.
**Source:** pr-review-toolkit silent-failure-hunter MEDIUM finding during the consolidated T-43+T-42+T-38 review.
**Priority:** MEDIUM — affects production users' ability to diagnose corrupted sidecars, WAL rejection on network shares, capture-time Tk failures, and other warning-level events.

## Problem

The codebase calls `logger.warning(...)` in several user-facing failure paths:

- `src/spectral_predict/run_state.py:153` — sidecar with invalid `bayesian_persistence_mode` coerced to `'never'`.
- `src/spectral_predict/run_state.py:174` — sidecar `gui_settings` malformed (string/list instead of dict), coerced to None.
- `src/spectral_predict/run_state.py:233-256` — empty-SQLite cleanup failures, lock-failures, etc.
- `src/spectral_predict/run_gui_settings.py:233` — Tk var `.get()` failure during capture; setting silently omitted from the snapshot.
- `src/spectral_predict/unified_bayesian.py:1658, 1662` — WAL pragma rejection (network share, AV-shimmed paths).
- `src/spectral_predict/unified_bayesian.py` (T-42 hoist guard) — value mismatch between current call args and stored study.user_attrs on resume.

No call to `logging.basicConfig()` happens anywhere in the codebase (verified via grep). Python defaults to WARNING+ on stderr. The PyInstaller-bundled GUI on Windows uses the GUI subsystem with no console — stderr is `nul` or a detached file.

**Net effect: every `logger.warning` written to these paths is invisible to production users.** The "log a warning so the issue isn't invisible" comments throughout the run_state module are technically wrong for the bundled-app deployment target.

## Failure modes hidden today

- Capture-time Tk `.get()` failure silently omits a setting from the snapshot. User's next resume restores fewer settings than expected; no banner indication.
- Sidecar with corrupt `bayesian_persistence_mode` coerced to `'never'`. User expected `'always'` for crash-recovery, gets a silent in-memory run.
- WAL rejection on a network-synced user-data dir (Dropbox, OneDrive). Performance silently halves; the run is more lock-prone.
- T-42 hoist mismatch on resume. User passes different `early_stopping_rounds` or `cv_strategy` than the prior session; the warning would alert a developer/API caller, but production users never see it.

## Fix

Two complementary approaches; either is sufficient on its own:

### Option A: file-based default handler at app startup

In `spectral_predict_gui_optimized.py`'s startup path (or in `run_logging.setup_run_logger`), add a default handler that writes to `<user_data_dir>/dasp/dasp.log` with rotation:

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    user_data_dir / "dasp.log",
    maxBytes=1_000_000, backupCount=3, encoding="utf-8",
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
))
root = logging.getLogger("spectral_predict")
root.addHandler(handler)
root.setLevel(logging.WARNING)
```

Cost: ~15 LOC at app startup. User can find the log file in the `Help > Open data directory` menu (already exists per existing run_logging code).

### Option B: route module warnings through `_log_progress`

Plumb a `progress_callback` through the modules so warnings can land in the GUI's progress text widget AND the disk log. More invasive (touches every callsite); higher value because warnings appear in the running session's UI, not just on disk.

Recommended: do Option A first (cheap, fixes the worst case). Option B is a longer-term refactor that's out of scope for this ticket.

## Out of scope

- Rewriting `run_state.py` / `run_gui_settings.py` to take a callback; that's a separate larger refactor.
- Removing the `logger.warning` calls and replacing with raises — they're correctly best-effort in those paths.

## Verification

1. Build the bundled installer (per `docs/BUNDLED_APP_BUILD_GUIDE_PY312.md`).
2. Hand-craft a corrupted sidecar with `bayesian_persistence_mode: "garbage"`.
3. Launch the app; verify the warning appears in `<user_data_dir>/dasp/dasp.log`.
