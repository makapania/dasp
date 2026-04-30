# Continuation prompt — DASP validation-gate ticket triage

> **How to use this:** On the other computer, `cd` into your DASP repo, `git pull`, then
> open Claude Code (or the assistant of your choice) and paste the prompt below into the
> first message of a new session. Update or trim sections that have already happened by
> the time you start.

---

## Prompt to paste

I'm continuing the DASP validation-gate ticket triage from a different computer. The
state of the work is fully captured in repo docs. Read these first, in order:

1. `CLAUDE.md` — project session protocol (mandatory).
2. `docs/PROJECT_STATUS.md` — top section is the most recent state. As of 2026-04-30, all
   of last night's bugfix run is closed (see the table). Any work below is on the broader
   roadmap.
3. `docs/SESSION_LOG.md` — top entry is "2026-04-30 — Bugfix branch validation gate
   session." Read it for the lessons-learned and the recurring overzealous-flag patterns.
4. `docs/bugfix_validation/README.md` — the validation gate methodology + 5 lessons.
   Re-read this every time before drafting a verdict.
5. `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` — top section "Validation gate
   outcomes (2026-04-30 session)" lists what's already done. Body lists per-ticket
   verdicts for the rest.

### Master rule (non-negotiable, ahead of any sklearn/ML instinct)

DASP implements chemometrics methods from the literature. The literature is the
specification. Validate every methodology decision against:

1. Chemometrics literature (Wold 2001, Centner 1996, Araujo 2001, Norgaard 2000, Wang
   1991, Lin 1989, Savitzky & Golay 1964, Pomerantsev/Kucheryavskiy/Rodionova 2025
   LOVE).
2. Applied domain (bone FTIR / paleoanthropology / collagen / diagenesis / isotopes).
3. **If sklearn / generic ML / genomics ML conventions conflict, chemometrics wins.**
4. Commercial software is a critical sanity check (Unscrambler, SIMCA, OPUS, GRAMS,
   PLS_Toolbox). If those tools don't do something, dasp shouldn't invent it.
5. **DASP ships as a bundled Inno Setup desktop app** to non-technical users — there's no
   Python REPL in their environment. A "fix" reachable only programmatically is dead
   code for the actual user base.
6. A real finding can warrant zero action (T-26 lesson — main behavior already matched
   PLS_Toolbox default; no fix needed).

### Validation gate methodology (apply to EVERY ticket before fixing)

For each ticket considered, do this in order:

1. **Verify reality in the codebase.** Read the flagged file/line directly. Check the
   surrounding control flow. Don't trust framings from earlier docs without verifying.
   Multiple flags from the prior re-evaluation turned out to be wrong on actual code
   reading (T-26, T-32, search.py:2855, bayesian random_state).
2. **Verify reachability in the GUI.** Trace the code path from a button click. Many
   "bugs" turn out to be defensive code in unreachable branches. Specifically check:
   does the user click anything that triggers it? Is there an upstream guard?
3. **Verify field alignment with documentation lookup, not generic intuition.** Search
   the actual paper PDF for canonical formulas. Search Eigenvector / Sartorius docs for
   commercial behavior. Don't claim "universal practice" without a citation.
4. **Empirical demonstration on a representative dataset.** Construct a small synthetic
   case where the bug should bite and verify the magnitude. T-21 was scoped down by
   running an actual SG-on-converted-grid measurement (median 22% / max 60% rel error
   in peak regions).
5. **Test sweep.** Before committing any fix: run T-X-specific tests, then
   `tests/smoke/test_imports.py` + `tests/test_cv_strategy.py` +
   `tests/test_search_comprehensive.py` + relevant domain tests
   (`tests/test_calibration_transfer*` for PDS, `tests/test_plsda_importance.py` for
   VIP, etc.). 200+ tests pass per ticket has been the bar.
6. **Distribution-model check before claiming "fixed."** Backend-only knobs that aren't
   reachable from the GUI deliver zero value to bundled-app users.

### Five recurring false-alarm patterns to watch for

- **sklearn-instinct false alarm** (T-26 first-pass) — claiming a chemometrics function
  is "buggy" because it doesn't match a sklearn convention. Often the field has its own
  pattern that dasp already matches.
- **Defensive code in unreachable branch** (T-32) — code looks wrong but the only path
  to it is gated out by upstream logic. Verify the gate, then defer.
- **Display-economy / code-style** (search.py:2855 top_n_vars, bayesian random_state) —
  appears to be a correctness bug but is actually a UI display choice or a code-style
  consistency issue.
- **dasp already matches the leading program** (T-26 final disposition) — dasp's
  current behavior turns out to be PLS_Toolbox's default. No code change needed; matches
  the field already.
- **Real finding, zero-code disposition** (T-21 final disposition — hide the
  bug-creating button rather than fix every SG callsite). Sometimes the cheapest
  defensible move is removing the surface that creates the bug.

### Workflow per ticket

1. Read the ticket's body in `RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`.
2. Check `docs/bugfix_validation/` for an existing investigation/verdict (some tickets
   already have findings docs from the 2026-04-30 session).
3. If there's no existing findings doc, dispatch a parallel investigation agent
   (general-purpose subagent) following the pattern of T-04/T-05/T-07/T-21 prompts in
   the prior session. Brief: read the validation gate methodology + master rule, do the
   six steps above, write findings to `docs/bugfix_validation/T<N>_findings.md`, do
   NOT write the verdict.
4. Synthesize the verdict yourself based on findings. Write
   `docs/bugfix_validation/T<N>_<short_name>.md` with the same structure as existing
   APPROVED notes.
5. If APPROVED: rebase the branch onto current main (resolves the docs-deletion noise
   from pre-reframing branches — same conflict pattern every time, takes ~5 min).
   Then run T-X tests + regression sweep (will likely take 1-3 min). If all green,
   fast-forward merge into main, push, update bugfix_validation/README.md status.
6. If DROP / DEFER: still write the verdict note documenting why. Future agents will
   be tempted to re-implement the dropped ticket; the note prevents that.
7. **Update PROJECT_STATUS.md and SESSION_LOG.md after each merge or disposition.**
   Multi-machine sync depends on these.

### Outstanding items still pending decisions

User decisions needed before further work on these:

- **T-31** (multi-class SIMCA): does "none of the above" / "belongs to multiple classes"
  output deliver scientific value in bone-FTIR / diagenesis / consolidant-detection
  workflow? Needs explicit yes/no.
- **T-01 reframe**: confirm external-test-set workflow (RMSEP on independent samples)
  over per-fold variable selection as the correct chemometrics solution to performance-
  bias reporting.
- **T-22 reframe**: confirm whether to invest in a bootstrap-stability variable-selection
  diagnostic (using the existing `varsel_transformer.py` from the paused
  `fix/varsel-leakage` branch as the engine).
- **T-04b**: broader y_oc-as-target audit (the same critique that made T-04 a real bug
  applies to `compute_one_class_importances`, `spa`, `cars`, `ga`, `vcpa-iriv` for
  one-class, and `preprocessing_discovery._quick_evaluate` one-class branch). Decide
  whether to scope this as a follow-up.
- **T-04c**: implement a one-class-native variable selection (Forina modeling power /
  Pomerantsev 2025 LOVE / 2025 OGA). Multi-week scope; probably blocked on T-04b first.

### Suggested next-ticket priority order (from the original re-evaluation)

After the bugfix close-out, these remain on the open roadmap:

1. **T-15 LeaveOneGroupOut / GroupKFold** — by site / batch / instrument. High priority
   for the user's bone-FTIR transferability work. Foundation for T-16. Bigger ticket
   (~1-2 weeks).
2. **T-16 Bootstrap CIs + paired permutation tests** — needed for publication-defensible
   model comparisons. Depends on T-15 for site-level block bootstrap.
3. **T-11 Pause/resume + Optuna SQLite persistence** — productivity win; user has lost
   long searches in the past.
4. **T-19 Model-native imbalance handling (loss reweighting)** — real reproducibility
   gap for boosting models (XGBoost scale_pos_weight, LightGBM class weighting,
   CatBoost auto_class_weights). Will also resolve the deferred T-32 bug.
5. **T-12 Disk-mirrored logging** — needed because GUI text widget logs are capped and
   bundled-app prints disappear.
6. **T-21 follow-up** (resample-on-convert) — IF the user actually wants the convert
   button back. Currently hidden. ~2-hour fix using `scipy.interpolate.interp1d`.

Smaller follow-ups that can be done in any order:

- **T-06 SPA random_starts** — `n_random_starts` parameter is non-functional today.
  Verified buggy. ~1-hour fix.
- **T-08 CARS tree-mode weight-update bias** — ~half-day per the original analysis.
- **T-09 SPC multi-subfile silent data loss** — I/O bug, file-format-specific,
  ~half-day.
- **T-13 Wire jackknife prediction intervals into GUI** — backend exists, GUI not
  wired.
- **T-14 Hardcoded version-string mismatch** — hygiene, ~10 min.
- **T-18 Stratified Kennard-Stone** — chemometrics-valid improvement.
- **T-20 Saved-model ↔ exported-script reproducibility test** — testing hygiene.
- **T-23 ENLR / LDA-on-PLS / PCR / MLR model cards** — classical chemometrics
  additions.
- **T-25 Safe ZIP extraction** — security.
- **T-27 Encoding handling** — hygiene.
- **T-28 Lower min_wavelengths** — practical file-loading improvement.
- **T-29 Replace bare excepts in scoring** — hygiene.
- **T-30 Remove debug prints** — hygiene.
- **T-04b/c** — see "Outstanding items" above.

### How to start in a new session

Recommended first message to me/Claude in the continuation session:

> "Pick up DASP validation-gate ticket triage from
> `docs/CONTINUATION_PROMPT_2026-04-30.md`. Read the prompt, the project status, the
> session log, and the bugfix_validation README. Then propose which next ticket to
> work on first based on the priority order — but verify reachability + field alignment
> before committing to it (apply the gate methodology, don't trust the prior framing
> blindly). For non-trivial investigations, dispatch a parallel investigation agent
> following the pattern of the prior T-04/T-21 prompts."

If you want to skip the priority-ordering step and just go directly to the highest
suggested item: **T-15 LeaveOneGroupOut / GroupKFold by site**. It's foundational for
the user's transferability work and unblocks T-16. Big-enough scope that it needs a
plan first; brainstorming → plan-writing → executing-plans skills apply.

If you want a smaller win first to verify the gate methodology still works on a
different computer: **T-06 SPA random_starts non-functionality**. ~1-hour fix, 1
file changed, narrow scope, easy to verify.

### Things NOT to do without explicit user approval

- Don't merge anything without running the regression test sweep first (200+ tests min).
- Don't fix code that turns out to be unreachable (T-32 lesson — verify the gate first).
- Don't add backend knobs that aren't reachable from the GUI (T-26 lesson —
  bundled-app distribution requires GUI-level fix).
- Don't trust prior plan documents without the validation gate (T-21 had three citation
  errors / scope gaps in the original plan).
- Don't delete worktrees without preserving branch refs (always `git worktree remove`,
  not `rm -rf .worktrees/*`).
- Don't squash empty `opencode draft snapshot` commits without confirming they're
  empty (`git show <hash> --stat`). Most are; some have content.

### Recent commits to know about (top of `git log` as of 2026-04-30 end-of-session)

```
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

End of continuation prompt.
