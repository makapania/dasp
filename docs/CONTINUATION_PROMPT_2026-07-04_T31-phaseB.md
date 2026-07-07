# T-31 Multi-Class SIMCA — Continuation Prompt (Phase B onward)

You are continuing the **T-31 multi-class SIMCA / class-modeling** build. Phase A (backend core) is complete, reviewed, hardened, and pushed. Your job is **Phase B**, then C, then D — one phase at a time, each ending in a multi-family review gate.

## 0. READ FIRST (mandatory, in this order)
1. `docs/PROJECT_STATUS.md` — top block is the T-31 active direction (state, commits, decisions).
2. `docs/superpowers/specs/2026-07-04-T31-multiclass-class-modeling.md` — the design spec (the source of truth for method decisions).
3. `docs/plans/2026-07-04-T31-multiclass-simca-implementation.md` — the TDD task plan (Phases A→D, tasks A1..D3).
4. `docs/SESSION_LOG.md` — the 2026-07-04 entry has the non-obvious findings (small-n calibration, empirical-p floor, leakage, NaN-metric bugs).
5. Skim `src/spectral_predict/simca.py` — the module you're extending (`MultiClassClassModel` + `multiclass_simca_metrics`/`wilson_ci`/`novelty_tradeoff_auc`).

## 1. Environment / branch
- Branch: **`feat/T31-multiclass-simca`** (off `origin/main`), pushed, tip is the docs commit `1f79eb5`. **`git branch --show-current` must equal `feat/T31-multiclass-simca` before ANY git op** (a parallel session shares this repo's HEAD).
- Python: **`.venv312/Scripts/python.exe`** only. Windows, Git Bash available.
- Run tests: `.venv312/Scripts/python.exe -m pytest tests/test_simca.py -q` (34 pass at Phase-A end). No new deps without `pyproject.toml`.

## 2. Execution model (proven across A1–A8 — keep it)
**Opus (you) orchestrate; GLM-5.2 write-mode workers implement; you review + commit per task.** Per task:
1. **Write the contract tests yourself** (Opus) in `tests/test_simca.py`, then run them and **confirm they FAIL** (red) — never a tautological test. Before setting a numeric threshold in a test, verify it's empirically achievable with a quick throwaway probe (this prevented an impossible-band halt in A2).
2. **Delegate the implementation** to a GLM-5.2 worker via the `opencode-call` agent: model alias `glm`, WRITE mode, **HALT-OR-BLOCK** instruction ("if a test looks wrong, STOP and report — do NOT edit tests or guess"), tell it to edit only the target file(s), run the suite, leave changes uncommitted. The wrapper often prints a false "(no changes detected)" — trust `git diff`, not that line.
3. **Review the diff yourself.** Verify correctness (esp. leakage: anything fit on data must be train-only inside folds), run the suite + a no-regression check on adjacent suites.
4. **Commit per task** with explicit `git add <paths>` (NEVER `git add -A` — the tree has many untracked `tools/` scratch files) + push. Commit message trailer:
   ```
   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
   Claude-Session: <your session url>
   ```
   (CRLF: `git diff --stat` after edits is fine; the repo normalizes LF→CRLF.)

## 3. Phase B tasks (spec §5.6; plan "Phase B")
Extend `src/spectral_predict/simca.py`. **All varsel is class-modeling-native or supervised-on-the-real-multi-class-label** — this is legitimate here (unlike one-class) because there IS a genuine multi-class label; state the reversal of the one-class UVE/iPLS exclusion in comments where relevant.

- **B1 — Wold modeling + discriminating power.**
  - Modeling power per class per variable: `MPOW_j = 1 − s_resid,j / s_total,j` (classical **Wold 1976**, distinct from the DD-SIMCA T²/Q framework — say so).
  - Discriminating power: pairwise cross-fit residual ratios (residual of class-i rows on class-j's model vs their own). **K>2 aggregation = macro-average the one-vs-rest per-variable DPOW** (pinned in the plan). "Balanced" mode selects variables maximizing `MPOW·DPOW`.
  - Selection modes: `modeling` / `discriminating` / `balanced`. Tests: MPOW hand-value on a fixed matrix; DPOW ranking stable across resamples (Spearman ρ ≥ 0.8, 20 resamples on K=3 synthetic); "balanced" retains ≥2/3 of known-relevant variables.
- **B2 — Wold diagnostic plot data.** Per-variable MPOW/DPOW arrays (correct shape/order for K classes) for the GUI to render in Phase D. Test: shapes/order.
- **B3 — supervised prefilter wiring.** Reuse the existing supervised methods (`importance`/`spa`/`cars`/`cars-tree`/`ga`) as a **global prefilter on the multi-class label**; tag `varsel_path=supervised|wold` on results. **Gate it with an external-novel-class regression test** proving the supervised path does NOT degrade the novelty rate below the full-spectra baseline (spec §5 guardrail — supervised selects for discrimination and could hurt novelty).

Design the varsel API to integrate cleanly (e.g. a `variable_selection` param on `MultiClassClassModel` that selects a wavelength mask before fitting per-class models, persisted in A8's format). Keep single-Y and existing paths untouched.

## 4. Deferred items to fold in during B/C/D (from the Phase-A gate)
- **`predict_with_uncertainty` has no `multiclass_simca` branch** — needed before Phase D GUI (it currently falls through and would return the dict where an ndarray is expected). Add a branch or explicit `NotImplementedError`.
- **`_cross_fit_null` uses `get_one_class_model` directly with no PCA wrapper for EllipticEnvelope** (unlike `run_one_class_cv`), so EE folds can fail when n_features>n_samples — add a warning on fold failure / PCA-reduce for EE.
- **Tuning-scaler leakage** (minor: the per_class scaler is fit on all rows before the inner tuning CV — affects only the discrete n_components choice). Fix if cheap; otherwise note it survives to merge review.
- **`novelty_tradeoff_auc` threshold count** is O(unique p-values) — downsample to ~500 before production scale (Phase C).

## 5. Multi-family review gate (MANDATORY at each phase boundary)
After Phase B is committed, run a **multi-family panel** on the Phase-B diff (spec + code): **Codex 5.5 (high) + ≥2 orthogonal families** from {DeepSeek V4 Pro, Kimi K2.7, MiniMax M3} — **rotate so GLM-5.2 (which wrote the code) does NOT review its own diff**. Route: Codex via the `codex-reviewer` agent; the others via `opencode-call` (aliases `deepseek` / `kimi27` / `minimax`), read-only. Consolidate findings, fold real ones with discriminating tests (revert-and-confirm-FAIL), surface any methodology decision to the user. At **merge-readiness** (after Phase D): full multi-family whole-diff pass + the complete **pr-review-toolkit** (code-reviewer, silent-failure-hunter, pr-test-analyzer, type-design-analyzer), then the **merge gate** = local diff-failure-set vs `origin/main` (main is red on cloud CI since 2025-10-27, so compare failure SETS; the PR must add zero new failures). **Do NOT auto-merge — await explicit user greenlight.**

## 6. Guardrails / decisions already locked (do not re-litigate)
- **min_class_samples = LAYERED** (user-approved): hard-block n<10; non-SIMCA n<20 unmodelable+warn (empirical-p floor `1/(m+1)` can't reach α=0.05 below m=20); SIMCA warns at n<max(20,5·n_comp) but still models; calibration surfaced via A7 metrics + Wilson CIs.
- **Chemometric standards, not ML standards, on leakage:** per-spectrum SNV/SG/baseline are NOT leakage; column-autoscale, calibration, varsel fitting are train-only inside folds. Both discriminant and class-modeling are legitimate paradigms — don't "correct" one toward the other.
- **α is global** (never per-class). **IsolationForest `score_samples` is NOT negated** (higher=more-normal). `scaling="per_class"` is the default. Never edit `search.py`'s single-Y path; keep existing regression/classification/one_class paths byte-identical.
- **Real-data validation set:** `C:\Users\mspon\Desktop\_DeskSync\contamination\Contaminated Samples Raw_ORAU Added.xlsx`, sheet `All Samples` (757×2151 FTIR; metadata cols `Specimen, Collagen, Site, contamination, Consolidant`; spectral cols are the integer-named 350–2500). `Site`=10 classes is the flagship novelty case; `contamination`=6, `Consolidant`=2. Use for the Phase-C real-data e2e (LOCO on held-out sites). Aggregate metrics only — don't dump raw spectra. Phase-A smoke: SIMCA flagged 53–86% of held-out-site samples novel vs 100%-forced discriminant baseline.

## 7. Session protocol
Per `CLAUDE.md`: append non-obvious findings to `SESSION_LOG.md` as you go; update `PROJECT_STATUS.md` after each phase; commit + push docs so other machines see them. Do NOT ask the user to remind you.

Start by reading the files in §0, then begin **Phase B / task B1** (write its contract tests first).
