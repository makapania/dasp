# Continuation prompt v2 — DASP roadmap (2026-04-30 evening)

> **How to use this:** On the other computer, `cd` into your DASP repo,
> `git pull`, then open Claude Code (or another assistant) and paste the
> prompt below into the first message of a new session. Update or trim
> sections that have already happened by the time you start.

---

## Prompt to paste

I'm continuing the DASP roadmap from a different machine. State of the
work is fully captured in repo docs. Read these first, in order:

1. `CLAUDE.md` — project session protocol (mandatory).
2. `docs/PROJECT_STATUS.md` — top section is the most recent state. As of
   2026-04-30 evening, T-06 / T-06b / T-08 / T-11 / T-12 / T-15 are
   resolved (merged or dropped). T-16 reframed pending user decision.
   T-19 reframed and approved with smaller scope.
3. `docs/SESSION_LOG.md` — top three entries cover today's session
   (T-08/T-11/T-15/T-16/T-19 evening) and the morning's T-06 + T-06b
   work. Read both for the lessons-learned and the recurring overzealous-
   flag patterns.
4. `docs/bugfix_validation/README.md` — validation gate methodology + 5
   lessons. Re-read every time before drafting a verdict. Status table at
   the top shows current ticket dispositions.
5. `docs/bugfix_validation/T11_pause_resume_hardening.md` —
   exemplar verdict-note style with full Codex+Kimi review fold-in.

### Master rule (non-negotiable, ahead of any sklearn/ML instinct)

DASP implements chemometrics methods from the literature. The literature
is the specification. Validate every methodology decision against:

1. **Chemometrics literature** — Wold 2001, Centner 1996, Araújo 2001,
   Nørgaard 2000, Wang 1991, Lin 1989, Savitzky & Golay 1964,
   Pomerantsev/Kucheryavskiy/Rodionova 2025 LOVE, Westad & Marini 2015,
   Workman 2018, Brereton, Filzmoser 2009.
2. **Applied domain** — bone FTIR / paleoanthropology / collagen /
   diagenesis / isotopes.
3. **If sklearn / generic ML / genomics ML / AutoML conventions
   conflict, chemometrics wins** unless the user explicitly overrides.
4. **Commercial software is a critical sanity check** (Unscrambler,
   SIMCA, OPUS, GRAMS, PLS_Toolbox / Solo, mdatools).
5. **DASP ships as a bundled Inno Setup desktop app** to non-technical
   users — there's no Python REPL in their environment. A "fix"
   reachable only programmatically is dead code for the actual user
   base.
6. **A real finding can warrant zero action** (T-26 lesson — main
   behavior already matched PLS_Toolbox default; no fix needed). Same
   pattern for T-08 (specific bug claim was empirically false on the
   actual code).

### Validation gate methodology (apply to EVERY ticket before fixing)

For each ticket considered, do this in order:

1. **Verify reality in the codebase.** Read the flagged file/line
   directly. Check the surrounding control flow. Don't trust framings
   from earlier docs without verifying. **Multiple flags from prior
   re-evaluations turned out to be wrong on actual code reading: T-26,
   T-32, search.py:2855, bayesian random_state, T-08 (cited line numbers
   referenced PLS-mode branch instead of tree-mode).**
2. **Verify reachability in the GUI.** Trace the code path from a
   button click. Many "bugs" turn out to be defensive code in
   unreachable branches. Specifically check: does the user click
   anything that triggers it? Is there an upstream guard? **T-32 was
   defensive code in unreachable branch; T-08 was reachable but
   not actually buggy.**
3. **Verify field alignment with documentation lookup, not generic
   intuition.** Search the actual paper PDF for canonical formulas.
   Search Eigenvector / Sartorius / CAMO docs for commercial behavior.
   Don't claim "universal practice" without a citation. **T-15 was
   dropped because Westad & Marini 2015 + Workman 2018 do NOT mandate
   LOGO over external test sets, and only PLS_Toolbox among competitors
   exposes group-aware CV.**
4. **Empirical demonstration on a representative dataset.** Construct a
   small synthetic case where the bug should bite and verify the
   magnitude. **T-21 was scoped down by running an actual SG-on-
   converted-grid measurement; T-08 was DROPPED after empirical
   reproducer showed all three claims were false.**
5. **Test sweep.** Before committing any fix: run T-X-specific tests,
   then `tests/smoke/test_imports.py` + `tests/test_cv_strategy.py` +
   `tests/test_search_comprehensive.py` + relevant domain tests
   (`tests/test_calibration_transfer*` for PDS, `tests/test_plsda_importance.py`
   for VIP, `tests/test_run_state.py` + `tests/test_search_controller.py`
   + `tests/test_run_logging.py` for pause/resume + Optuna persistence,
   etc.). **260+ tests pass per ticket has been the bar.**
6. **Distribution-model check before claiming "fixed."** Backend-only
   knobs that aren't reachable from the GUI deliver zero value to
   bundled-app users.

### Cross-family review pattern (use it for non-trivial fixes)

After implementing any fix touching > 2 files or any infrastructure
ticket, dispatch parallel reviewers:

- **Codex** (US-trained, via codex-reviewer agent) — catches
  algorithmic + Optuna + framework-specific gotchas.
- **Kimi K2.6 via opencode-go** (Moonshot, Chinese-trained, **NEVER
  via z.ai**) — catches Python/threading/Windows-specific gotchas + has
  different blind spots than US-trained models.

This session: Codex caught HIGH bugs Claude missed in T-06 (template
divergence, missed call-site, weak test) and T-11 (trial-count overrun,
non-atomic writes, study-name collisions, fingerprint-not-enforced).
Kimi caught MAJOR/MINOR bugs both Claude and Codex missed
(`splitlines()` splits on `\r`, Nuitka `dir()` vs `globals()`, run_id
race in fingerprint check, SQLite locking).

### CRITICAL routing rules for sub-agent dispatch

- **NEVER dispatch GLM 5.1 through opencode-go.** It bills against the
  user's z.ai subscription. An agent did this on the morning of
  2026-04-30 and it cost money. For GLM 5.1, ONLY use the `llm-call`
  skill (which routes through z.ai direct subscription) or ask the user
  first.
- For Kimi K2.6, DeepSeek, MiniMax, Qwen, MiMo: opencode-call IS fine
  (those go through opencode-go's flat-rate plan).
- For codex-reviewer / Codex CLI: separate path, unaffected.
- For peer-review skill (multi-model panel): the panel uses GLM by
  default — do NOT use peer-review skill without first confirming with
  the user, since it may invoke GLM and consume z.ai credits.

This rule overrides any default routing in opencode-call or peer-review
skill descriptions.

### Five recurring false-alarm patterns to watch for

- **sklearn-instinct false alarm** (T-26 first-pass, T-06 `rng.choice()`
  proposal) — claiming a chemometrics function is "buggy" because it
  doesn't match an sklearn convention. Often the field has its own
  pattern that dasp already matches (or that dasp invented for a real
  reason — see CARS-Tree below).
- **Defensive code in unreachable branch** (T-32) — code looks wrong
  but the only path to it is gated out by upstream logic. Verify the
  gate, then defer.
- **Display-economy / code-style** (search.py:2855 top_n_vars,
  bayesian random_state) — appears to be a correctness bug but is
  actually a UI display choice or a code-style consistency issue.
- **dasp already matches the leading program** (T-26 final disposition) —
  dasp's current behavior turns out to be PLS_Toolbox's default. No
  code change needed; matches the field already.
- **Real finding, zero-code disposition** (T-21 final disposition,
  T-08 final disposition). Sometimes the cheapest defensible move is
  documenting why we're not acting.
- **Wrong-line-number citation** (T-08 — cited 1519-1522 are
  PLS-mode branch; tree-mode is at 1499-1507). Always read the actual
  control flow before believing the framing's claim.

### Project-specific knowledge (do NOT re-derive)

These are saved in user auto-memory and will surface automatically when
relevant. Do NOT search the literature for canonical references to
these — the answers are already known:

1. **CARS-Tree is dasp's invention** (`project_cars_tree_origin.md`).
   Don't search Li 2009 for canonical "tree-mode CARS" — it doesn't
   exist. dasp invented it because canonical CARS depends on PLS
   regression coefficients which tree models don't expose. The frame
   for any future CARS-Tree ticket is "does our intentional invention
   work?" not "does it match canon?"
2. **T-15 LOGO is dropped** (`project_t15_dropped_t16_reframed.md`).
   User decided after gate investigation. Don't re-open unless user
   revives the FTIR Bone PLS paper push that depends on LOGO, or
   demands T-15 + T-16 paired implementation.
3. **T-19 user framing** (`project_t19_user_framing.md`). NOT
   "publication reproducibility framework." IS "expose model-native
   imbalance abilities OR auto-detect via existing
   `detect_class_imbalance()`." Yesterday's design doc captured detail;
   don't re-investigate from scratch.

### Workflow per ticket

1. Read the ticket's body in `RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`.
2. Check `docs/bugfix_validation/` for an existing investigation/verdict
   (T-08, T-11, T-15, T-16, T-19 already have findings docs from this
   session).
3. If there's no existing findings doc, dispatch a parallel investigation
   agent (general-purpose subagent) following the pattern of T-04 / T-21
   / T-08 / T-15 / T-16 prompts. Brief: read the validation gate
   methodology + master rule, do the six steps above, write findings to
   `docs/bugfix_validation/T<N>_findings.md`, do NOT write the verdict.
4. Synthesize the verdict yourself based on findings. Write
   `docs/bugfix_validation/T<N>_<short_name>.md` with the same structure
   as existing APPROVED notes (use `T11_pause_resume_hardening.md` as
   the most-current exemplar — full cross-family review fold-in).
5. **For non-trivial implementations: dispatch Codex + Kimi K2.6
   reviewers in parallel** before merge. They catch real bugs that
   would have shipped (this session: Codex HIGH on T-06 template, T-11
   trial-count overrun; Kimi MAJOR on T-11 run_id race, T-11 splitlines
   bug).
6. If APPROVED: rebase the branch onto current main (resolves any
   docs-deletion noise — same conflict pattern every time, takes ~5
   min). Then run T-X tests + regression sweep (260+ test bar). If all
   green, fast-forward merge into main, push (with user approval —
   this session the user explicitly held the push; check first).
7. If DROP / DEFER: still write the verdict note documenting why.
   Future agents will be tempted to re-implement the dropped ticket;
   the note prevents that.
8. **Update PROJECT_STATUS.md and SESSION_LOG.md after each merge or
   disposition.** Multi-machine sync depends on these.

### Outstanding items + suggested next-ticket priority order

**User decisions still pending (block bigger work):**

- **T-16 Shape A vs Shape B vs hybrid.** Shape A = chemometrics canon
  (jackknife wiring + Y-permutation), gets PLS_Toolbox parity +
  Unscrambler-adjacent parity, ~3-5 days, **closes T-13 simultaneously**.
  Shape B = ML canon (paired t-test/Wilcoxon for between-model), needs
  per-fold metric storage schema upgrade (~2-3 days hidden infra), then
  ~5-7 days for the comparison machinery itself. **Shape A matches
  user's "head start" framing; Shape B matches "comparing between
  models" framing.** They're different feature sets, not different
  implementations.
- **T-31 multi-class SIMCA.** Confirm "none of the above" output is
  useful for bone-FTIR / diagenesis science. If yes, scope as 1-2 week
  effort.
- **T-22 reframe.** Confirm bootstrap stability diagnostic investment
  (different from T-16 model-comparison machinery — this is varsel
  stability across resamples).
- **T-01 reframe.** Confirm external-test-set workflow scope.
- **T-04b** broader y_oc-as-target audit + **T-04c** one-class-native
  variable selection (Forina modeling power / Pomerantsev 2025 LOVE).

**Approved + ready to ship (no more user decision needed):**

1. **T-19 (smaller scope, ~2-3 days)** — extend the existing imbalance
   dropdown to dispatch native kwargs internally + fix 5 PLS-DA
   inner-LR sites + close T-32. Audit-trail labels for FTIR Bone PLS
   paper. Yesterday's design doc + Codex review captured detail; do NOT
   redesign. **This is the highest-value ready-to-ship item.**

**Approved smaller wins (short, defensible scopes):**

2. **T-14 hardcoded version-string mismatch** (~10 min, hygiene). Three
   sites disagree: `__init__.py:3` says `0.5.0b1`, `report.py:139` says
   `v0.4.0`, `code_generator.py:373` says `3.9.0`. Single version
   constant.
3. **T-30 Remove debug prints** (~30 min, hygiene). `search.py:2313,
   2789, 2834, 3091-3100, 3991-3999`. Replace with logger calls or
   remove.
4. **T-29 Replace bare `except:` in scoring.py with NaN + warning**
   (~1 hr, hygiene). `scoring.py:509, 517, 535`.
5. **T-27 Encoding handling on file I/O** (~1 hr, hygiene). Add
   `encoding='utf-8'` to `open()` and `pd.read_csv()` in `io.py`,
   `report.py`, `calibration_transfer.py`.
6. **T-28 Lower `min_wavelengths` from 100 to 10** (~30 min, practical).
   `io.py:140, 1864` rejects low-resolution instruments + peak-calculator
   outputs.
7. **T-25 Safe ZIP extraction** (~2-3 hrs, security). `model_io.py:417,
   1676` uses `extractall()` without path validation (Zip Slip).
   Validate ZIP entry names + UX warning at load.

**Larger items (need their own plan):**

8. **T-16 implementation** — once user picks Shape A or B. Shape A
   easier to start (closes T-13 simultaneously); Shape B needs per-fold
   metric schema work first.
9. **T-17 Multi-Y / PLS-2 workflow** (2-3 weeks). User-endorsed.
   Canonical multi-output PLS variant.
10. **T-18 Stratified Kennard-Stone for cal/test split** (1-2 days).
    Class-balanced KS per Kennard 1969.
11. **T-23 Add ENLR / LDA-on-PLS / PCR / MLR as model cards** (1 week).
    Classical chemometrics additions; ENLR already constructed in
    `nsga2_search.py:856`.
12. **T-20 Saved-model ↔ exported-script reproducibility test**
    (~2 days, testing hygiene).

**Smaller follow-ups that can be done in any order (already ranked above
where applicable):** T-09 (SPC multi-subfile silent data loss, half-day,
file-format-specific), T-04b/c follow-ups (see Outstanding above).

### How to start in a new session

Recommended first message to the assistant:

> "Pick up DASP roadmap from `docs/CONTINUATION_PROMPT_2026-04-30_v2.md`.
> Read the prompt, the project status, the session log, and the
> bugfix_validation README. Then propose which next ticket to work on
> first based on the priority order — but verify reachability + field
> alignment before committing to it (apply the gate methodology, don't
> trust prior framing blindly). For non-trivial investigations, dispatch
> a parallel investigation agent following the pattern of the prior
> T-08/T-11/T-15/T-16 prompts. For non-trivial implementations, dispatch
> Codex + Kimi K2.6 reviewers in parallel before merge."

If you want to skip the priority-ordering step:

- **Smallest defensible win:** T-14 (10-min hygiene). Verifies your gate
  methodology works on a different computer with minimum risk.
- **Highest ready-to-ship value:** T-19 (smaller scope, ~2-3 days).
  User-approved, design captured yesterday, audit-trail for FTIR paper.
- **If user decides on T-16 Shape A:** ~3-5 days, closes T-13
  simultaneously, gets PLS_Toolbox + Unscrambler-adjacent parity.

### Things NOT to do without explicit user approval

- Don't merge anything without running the regression test sweep first
  (260+ tests minimum).
- **Don't push to remote without explicit user approval.** This session
  the user committed locally but explicitly held the push.
- Don't fix code that turns out to be unreachable (T-32 lesson — verify
  the gate first).
- Don't add backend knobs that aren't reachable from the GUI (T-26
  lesson — bundled-app distribution requires GUI-level fix).
- Don't trust prior plan documents without the validation gate (T-21
  had three citation errors / scope gaps in the original plan; T-08's
  framing cited wrong line numbers).
- Don't delete worktrees without preserving branch refs (always
  `git worktree remove`, not `rm -rf .worktrees/*`).
- Don't squash empty `opencode draft snapshot` commits without
  confirming they're empty (`git show <hash> --stat`). Most are; some
  have content.
- **Don't dispatch GLM 5.1 through opencode-go ever.** Charges user's
  z.ai subscription. Use `llm-call` skill or ask first.
- **Don't re-investigate T-15, T-19 framing, or CARS-Tree origin** —
  these are saved in project memory and were extensively debated this
  session. Future agents should defer to the captured decisions.

### Recent commits to know about

```
(local-only on fix/T11-pause-resume-hardening, NOT pushed)
<COMMIT_HASH> fix(T-11): pause/resume hardening + Optuna SQLite + disk logging

(merged + pushed earlier today)
85063b8 perf(T-06b): parallelize canonical SPA seed loop via joblib threading
af44ad4 fix(T-06): canonical Araújo 2001 SPA enumeration
c8c02fe docs: add continuation prompt for multi-machine handoff
ddae018 docs: T-21 investigation findings (was uncommitted)
46b238c docs: roadmap — add validation gate outcomes section
d8f80aa docs(SESSION_LOG): bugfix validation gate session — full close-out
03297c9 docs: PROJECT_STATUS — bugfix branch validation gate session summary
3edee47 docs: T-04 status + T-21 plan banner + CLAUDE.md varsel note
a5eef70 fix(T-21): hide x-unit Convert button — eliminates non-uniform-grid SG bug
6beb5e8 fix(T-04): grey out UVE family in one-class mode (matches iPLS pattern)
1eb6c06 fix(T-05a): canonical Wold 2001 VIP at the two duplicate sites
50d5d05 refactor(bayesian_utils): use RANDOM_STATE constant instead of literal 42
0087cad (T-24 merge — Lin's CCC metric)
1b91d93 (T-07 merge — PDS even-window)
2c068cd (T-05 merge — VIP central formula)
fbeb50c (T-10 merge — PLS components clamp)
```

### Specific gotchas surfaced this session worth carrying forward

1. **`study.optimize(n_trials=N)` runs N MORE trials, not until total
   reaches N.** When using Optuna's `load_if_exists=True` resume,
   ALWAYS clamp `remaining = max(0, target - already_finished)` using
   `TrialState.COMPLETE | PRUNED` filter. Otherwise resumed runs
   silently double the work.
2. **`splitlines()` splits on `\r` AND `\n`.** For tee-style stdout
   capture, strip `\r` first then split on `\n` only. Otherwise tqdm
   bar updates spam the log file.
3. **`"__compiled__" in dir()` inside a function checks LOCAL namespace,
   not module globals.** Nuitka injects `__compiled__` at module scope.
   Use `"__compiled__" in globals()` for frozen-mode detection.
4. **`os.replace` is atomic on both POSIX and Windows since Python 3.3.**
   Use it for sidecar writes: tempfile in same dir + fsync + replace.
   `Path.write_text` is NOT atomic.
5. **Optuna `study_name` collisions are silent under `load_if_exists=True`.**
   If two runs with different config share a study name, trials get
   mixed and rankings corrupt. Include a config-fingerprint in the
   study name; deliberately omit `n_trials` (the trial-count clamp
   handles target-changes).
6. **SQLite default busy-timeout is short** — can cause "database is
   locked" on Windows. Append `?check_same_thread=False&timeout=30` to
   storage URLs.
7. **`logging.FileHandler` doesn't rotate; `RotatingFileHandler` does.**
   Use the rotating variant for any log that may run for hours.

End of continuation prompt.
