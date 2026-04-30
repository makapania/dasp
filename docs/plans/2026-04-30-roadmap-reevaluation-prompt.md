# Roadmap Re-evaluation Agent Prompt

> Use this as a self-contained prompt to dispatch a fresh agent (general-purpose subagent / Sonnet) to re-evaluate the dasp roadmap under the chemometrics-literature master rule. Designed to NOT inherit prior agent framings.

---

## Prompt to dispatch

You are a methodology reviewer for the dasp project at `C:\Users\sponheim\git\dasp`. Your job is to re-evaluate the entire active ticket list under a specific lens, and to decide what (if anything) in the in-flight `fix/varsel-leakage` worktree is worth keeping.

### THE MASTER RULE (do not deviate)

Every methodology decision in dasp is judged against:
1. **Chemometrics literature** — Li 2009 (CARS), Centner 1996 (UVE), Araujo 2001 (SPA), Norgaard 2000 (iPLS), Wold 2001 (PLS/VIP), Wang 1991 (PDS), Savitzky-Golay 1964, Lin 1989 (CCC), and similar canonical references.
2. **The applied domain** — paleoanthropology / archaeology + bone FTIR + stable isotopes + diagenesis. Methods, metrics, and validation approaches that are standard in this domain take precedence over generic-ML conventions.

When sklearn / genomics-ML / generic-data-science conventions conflict with chemometrics + application convention, **chemometrics wins** unless the user has explicitly said otherwise.

**The recurring failure mode you must guard against:** prior agents (including the one that wrote the T-01 audit) imported sklearn-pipeline-purity instincts and mislabeled standard chemometrics workflow as "data leakage." Do not repeat this. Before flagging anything as a bug, validate against the relevant literature.

### Required reading before starting

Read in this order:

1. `~/.claude/projects/C--Users-sponheim-git-dasp/memory/feedback_validate_against_chemometrics_and_application_lit.md` — the master rule and instances.
2. `~/.claude/projects/C--Users-sponheim-git-dasp/memory/feedback_chemometrics_conventions.md`
3. `~/.claude/projects/C--Users-sponheim-git-dasp/memory/feedback_leakage_false_alarms.md`
4. `~/.claude/projects/C--Users-sponheim-git-dasp/memory/feedback_varsel_finds_chemistry_not_signal.md`
5. `docs/PROJECT_STATUS.md` (top — recent updates explain what just happened)
6. `docs/T01_VARSEL_LEAKAGE_AUDIT.md` (read the banner at top first — explains why the body's framing was wrong)
7. `docs/analysis_vs_chemometrics_lit/leakage_validation_GLM_2026-04-30.md` (282 lines, 12 references)
8. `docs/analysis_vs_chemometrics_lit/leakage_validation_codex_2026-04-30.txt`
9. `docs/RECONCILED_ROADMAP_2026-04-29.md` — the full ticket list to re-evaluate.
10. `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` — the user's actual published-application context.
11. Spot-check the user's downloaded chemometrics papers in `~/Downloads/` (start with `Li et al. 2009 - Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration.pdf`; grep for other relevant chemometrics PDFs).

### Deliverable 1: Re-evaluated roadmap — EVERY TICKET, NOT JUST SUSPECT ONES

Produce `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`. **You must produce a verdict for EVERY one of the 32 tickets (T-01 through T-32) PLUS the deferred items (T-05a, T-10b, T-31 PENDING, the P3 drop list).** Do not skip "obviously fine" tickets — even hygiene fixes (debug prints, encoding nits, safe ZIP extraction) should get a one-line confirmation that they survive the re-evaluation. The whole point is to catch the cases where prior agents stamped a verdict that doesn't survive the master rule.

For each ticket, output a verdict from this set:

- **KEEP** — ticket is genuinely valuable under chemometrics convention + application context. Explain why with literature reference where relevant.
- **REFRAME** — the issue is real but the proposed framing/fix imports the wrong literature. Describe what the *correct* framing/fix is.
- **DROP** — ticket was based on wrong-literature framing. Explain.
- **DEFER** — real but lower priority than current state suggests; explain.
- **NEEDS_USER_DECISION** — depends on a workflow choice the user must make.

**Format per ticket** (mandatory, no shortcuts):
```
### T-XX: [name from existing roadmap]
**Verdict:** KEEP / REFRAME / DROP / DEFER / NEEDS_USER_DECISION
**Why:** [one paragraph max — cite the relevant chemometrics paper or applied-domain reference, or note "no methodology question, this is a [bug fix / hygiene / security]"]
**If REFRAME:** [what the correct framing/fix is]
**If DROP:** [explanation of which literature was incorrectly imported]
```

Particular tickets to scrutinize hard (these are most at-risk of false-alarm framing — but you must still evaluate ALL the others too):
- **T-01** (varsel leakage): banner already says reconsidered. Confirm or push back.
- **T-02** (ensemble OOF preprocessor leakage at `ensemble.py:315-316`): is this actually a problem in chemometrics convention, or is it the same false alarm as T-01? Trace the ensemble code.
- **T-03** (preprocessing-discovery full-data leakage at `preprocessing_discovery.py:217`): same question. The "discovered preprocessing is selected on leaked information" framing — does this actually matter in chemometrics where preprocessing choice is typically reported as a methodology decision rather than a CV-tuned hyperparameter?
- **T-04** (one-class UVE prefilter on outlier-contaminated labels): probably a real issue (the *labels* used for UVE may be wrong, distinct from the where-CV-runs question). Verify against one-class chemometrics literature (PCA-SIMCA, OneClassSVM in spectroscopy contexts).
- **T-15** (LeaveOneGroupOut by site): genuinely valuable for transferability claims; verify the user's published peers in bone FTIR use group-CV.
- **T-16** (bootstrap CIs / paired permutation tests): two-sample inference machinery; verify against application literature.
- **T-17** (multi-Y / PLS-2): user already endorsed; sanity-check the canonical chemometrics PLS-2 references (Wold) match the design.
- **T-19** (model-native loss reweighting): boosting models / PLS-DA inner LR; verify against imbalance handling in spectroscopic classification literature.
- **T-21** (SG wavelength uniformity guard): plan exists but not implemented. **Verify against chemometrics SG implementations** — do canonical chemometrics SG implementations check uniform spacing, or is the user assumed to provide it?
- **T-22** (multi-source consensus wavenumber selection): bootstrap-style stability analysis is closer to "what's the right diagnostic for finding real chemistry" — connect to the varsel-finds-chemistry memory.
- **T-25** (safe ZIP extraction): genuinely a security issue, no chemometrics conflict.
- **T-26** through **T-30**: small QoL / hygiene fixes — should mostly be KEEP.
- **T-31** (multi-class SIMCA): pending user decision; the user's domain context (sites + consolidants) supports this.

For each ticket, cite at least one chemometrics paper or applied-domain reference where the framing question is non-trivial.

### Deliverable 2: Verdict on `fix/varsel-leakage` worktree

The worktree at `.worktrees/varsel-leakage-fix` on branch `fix/varsel-leakage` contains:
- 4 plan-revision commits + 5 Phase-1 implementation commits (`70738c9` → `ac4aae1`)
- New file `src/spectral_predict/varsel_transformer.py` (~500 LOC, 4 classes: `VarselTransformer`, `ModelImportanceVarselTransformer`, `SubsetVarselTransformer`, `OneClassVarselTransformer`)
- New file `tests/test_varsel_transformer.py` (~243 LOC, 32 passing tests)
- The (now-banner'd) audit doc `docs/T01_VARSEL_LEAKAGE_AUDIT.md`
- The 2900-line plan `docs/plans/2026-04-29-varsel-leakage-fix.md` (also represents wrong framing)
- Codex review files (`codex_review_*.txt`)
- The two literature validation files (already copied to main repo at `docs/analysis_vs_chemometrics_lit/`)

For EACH artifact, output a verdict:
- **MERGE** — keep it as-is.
- **MERGE_AS_OPTIONAL_TOOL** — code/feature is valuable as a non-default diagnostic (e.g., bootstrap stability of varsel selections — connect to T-22). Document as such.
- **CHERRY_PICK** — only specific commits / specific functions / specific tests are worth keeping; specify which.
- **DROP** — discard the entire artifact.

Particular questions:
- Is the `VarselTransformer` family architecturally useful for ANY purpose (bootstrap stability diagnostics? expert mode for users with external test sets? reproducing some specific paper?), or is it dead code under the master rule?
- Does the audit doc have value as a *description of the code's CV-honesty status* (renamed and reframed) even if the "leakage" claim was wrong?
- Are there any insights in the plan / Codex reviews / GLM literature analysis worth extracting into a separate doc?

### Deliverable 3: Recommendations

After Deliverable 1 and 2:

1. **Top 5 next actions** under the new framing (which tickets to actually do, in what order).
2. **Tickets to add** that aren't in the current roadmap but are obvious gaps once you apply the master rule (e.g., better external-test-set workflow if that's the right answer to "honest performance reporting", or a varsel-stability-diagnostic if T-22 should be reframed that way).
3. **Tickets the user should make decisions on** before any agent proceeds.

### Constraints

- **NO source code changes.** This is analysis/planning work only. You may move/copy doc files within `docs/` if that helps organization, but no `src/spectral_predict/` edits.
- If you find what looks like a real bug while tracing code (similar to how Codex caught the active sample_weight length-mismatch at `search.py:3883` while reviewing T-19), note it as a *potential new ticket*, do not fix it.
- **Do NOT trust the existing audit doc, plan doc, or roadmap-from-2026-04-29 as authoritative.** They were produced under the wrong-literature framing. Treat them as suspect inputs.
- **Do NOT trust your own sklearn/ML training instincts** when they conflict with explicit chemometrics convention. When in doubt, dispatch a literature-search sub-task or ask the user.
- **Do dispatch sub-agents (Codex CLI for fresh-eyes review, GLM via opencode-go for grunt literature lookups)** when a single ticket evaluation needs deep paper analysis. Do not do everything in one head.
- Cite specific papers / file:line evidence for every non-trivial verdict.

### Output format

Save to:
1. `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` — Deliverable 1
2. `docs/varsel_leakage_worktree_disposition_2026-04-30.md` — Deliverable 2
3. Append to `docs/PROJECT_STATUS.md` with a "Last updated" entry summarizing your verdicts (one paragraph)
4. Append to `docs/SESSION_LOG.md` with non-obvious findings

Do NOT push.

### Final report

When done, summarize back to the user:
- Number of tickets KEEP / REFRAME / DROP / DEFER / NEEDS_USER_DECISION
- Verdict on the worktree (MERGE_AS_OPTIONAL_TOOL / DROP / cherry-pick list)
- Top 5 next actions
- Anything that needs the user's explicit decision before further work.
