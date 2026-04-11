# Session-End Skill Design

**Date:** 2026-04-09
**Location:** `~/.claude/commands/session-end.md` (global, all projects)
**Replaces:** `~/.claude/agents/day-closer.md`

## Problem

Without a systematic cleanup process, repos accumulate:
- Test output dumps in root (26 MB+ in dasp)
- One-off report MDs scattered in root
- Stale docs that don't reflect reality
- Build artifacts not gitignored
- Uncommitted/unpushed work left dangling

## Design Decisions

- **Skill, not agent** — runs in conversation context so it knows what happened this session. Better doc updates than an agent re-discovering via git diff.
- **Moves are automatic, deletes need permission** — moves are safe and reversible. Deletes are not.
- **`archive/` with subfolders** — `reports/`, `test-outputs/`, `backups/`, `misc/`. Gitignored (local holding pen, never committed).
- **PROJECT_STATUS.md + SESSION_LOG.md** — standardized across all projects. Offered if missing, not forced.
- **Check remote before committing** — avoids half-synced state across machines.

## Five Phases

1. **Clutter scan and move** — detect junk in root, auto-move to archive/ subfolders, update .gitignore to prevent recurrence
2. **Repo health checks** — .gitignore gaps, large tracked files, stale branches, notebook outputs, TODO/FIXME in diff, dependency drift
3. **Update docs** — PROJECT_STATUS.md and SESSION_LOG.md using conversation context
4. **Git hygiene** — check remote, stage, commit, push, verify
5. **Summary** — brief report of actions taken and items flagged

## PROJECT_STATUS.md Template

```markdown
## Status: [Working / Broken / In Progress]
## Last updated: [YYYY-MM-DD] [machine]

## What Works
- [bullet points]

## Known Issues
- [bullet points]

## Next Steps (prioritized)
1.
2.

## Environment Notes
- Python version, key deps, OS

## Active Branch
- [branch name and purpose]
```

## Input Sources

- Code reviewer agent: suggested .gitignore health checks, stale branches, flat date-based archive (we chose categorized subfolders instead), check remote before commit
- General review agent: suggested Environment Notes in PROJECT_STATUS.md, dependency drift check, TODO/FIXME scan, notebook output warnings, cautioned about auto-moves breaking relative paths (mitigated by skip-list of known safe root files)
