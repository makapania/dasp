# Codex Review: Imbalance Handling Fix Commits

**Date:** 2026-04-21  
**Branch/worktree:** `worktree-agent-a4805dd0`  
**Commits reviewed:** `0d6b9f0`, `182469b`, `797cbe4`  
**Verdict:** ship-with-fixes

## Overall Verdict

The executor mostly followed the plan and the changes are scoped to the intended files. The dropdown values are now task-type-aware, the one-class imbalance UI is hidden, the initial dropdown is cleaned up, and backend Bayesian/NSGA-II substitutions are surfaced through `logger.warning()` plus the existing GUI `progress_callback` path.

However, the substitution banner does not actually appear when `_refresh_imbalance_methods()` auto-migrates an invalid selection. The banner text is set and then immediately cleared by `_update_imbalance_method_description()`. This is the main user-facing regression in the fix set and should be corrected before shipping the banner behavior.

`python -m py_compile spectral_predict_gui_optimized.py src/spectral_predict/unified_bayesian.py src/spectral_predict/nsga2_search.py` passed.

## Blocking Issues

### 1. Substitution banner is immediately cleared

Location: `spectral_predict_gui_optimized.py:22416-22420`, `spectral_predict_gui_optimized.py:22323-22325`

When the current imbalance method is invalid for the new task type, `_refresh_imbalance_methods()` does this:

1. Sets the replacement method.
2. Calls `_set_imbalance_banner(...)`.
3. Calls `_update_imbalance_method_description(None)`.

But `_update_imbalance_method_description()` starts by calling `_clear_imbalance_banner()`, so the banner text is removed in the same call stack. This means the intended warning does not appear for task switching, including the explicit scenario from the plan: classification `adasyn` -> regression `smogn`.

Suggested fix: update the method description before setting the banner, or add an optional flag to `_update_imbalance_method_description(..., clear_banner=True)` and call it with `False` from `_refresh_imbalance_methods()` when the change is an auto-substitution.

## Suggestions

### 1. Dropdown migration is behaviorally correct, aside from the banner

The classification/regression method lists match the plan. Switching from classification to regression after selecting `adasyn` migrates to `smogn`; switching back migrates to `smote`. One-class gets an empty method list and an empty selected value, while the section is hidden.

After fixing the banner ordering, this should satisfy the task-type-aware dropdown requirement.

### 2. Backend progress callback wiring looks correct

Locations: `src/spectral_predict/unified_bayesian.py:1718-1723`, `src/spectral_predict/nsga2_search.py:1817-1822`, `spectral_predict_gui_optimized.py:27525-27528`

Both backend paths now call `progress_callback({'message': ...})`, and the GUI callback reads `info.get('message', '')` and forwards it to `_log_progress()`. The GUI passes callbacks into Bayesian and NSGA-II runs, including the Bayesian wrapper path, so these warnings should reach the progress text area.

Minor caveat: `_progress_callback()` also updates progress labels using default `current=0`, `total=1` for message-only callbacks. That is not a correctness issue, but a future cleanup could skip progress-label updates when the callback dict only contains `message`.

### 3. One-class visibility is implemented as planned

Promoting `imbalance_frame` and `imbalance_section_heading` to instance attributes was done, and `rg` found no remaining local `imbalance_frame` references. `_update_one_class_controls_visibility()` hides and restores both widgets using the same `grid_remove()` / `grid()` pattern used elsewhere.

### 4. Plan adherence is strong, with one functional miss

Tasks 4.1, 4.2, 4.4, and 4.5 were implemented as planned. Task 4.3 was structurally implemented, but the banner does not display because of the clear-after-set ordering bug above.

The plan explicitly left one-class imbalance detection display behavior out of scope. The current `_detect_and_display_imbalance()` still treats non-classification, including `one_class`, as regression for the distribution summary path. That matches the plan's out-of-scope note, but it remains a possible future cleanup.

## Nits

1. `unified_bayesian.py` now has `import logging`, a blank line, `logger = ...`, another blank line, then other imports. It is valid, but most of the file would read cleaner if all imports stayed together followed by module constants/loggers.

2. The banner message uses ASCII `->`, while the manual smoke test in the plan uses a Unicode arrow. ASCII is consistent with the repo-editing constraints and is fine.

3. `_refresh_imbalance_methods()` clears the banner whenever the current method is valid. That is reasonable for task-type changes, but after the ordering bug is fixed, be careful not to clear a substitution message during an unrelated valid refresh unless that is intentional.

## Final Recommendation

Fix the banner clear ordering, then ship. I did not find missed `self.imbalance_frame` conversions, backend callback wiring problems, Python syntax issues, or obvious threading regressions in these commits.
