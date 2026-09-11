# Agent Guide — see CLAUDE.md

This project keeps **one** agent guide, and it is [`CLAUDE.md`](CLAUDE.md).

**Read `CLAUDE.md` now, in full, before doing any work in this repo.** It carries
the mandatory session protocol (read `docs/PROJECT_STATUS.md` first; append to
`docs/SESSION_LOG.md` as you learn things; update and push both at session end),
the entry point, the directory map, and the rule that any script calling the
backend must first read `docs/AGENT_COMPOSITION.md`.

This file exists only because some agents look for `AGENTS.md` rather than
`CLAUDE.md`. It is deliberately a pointer and not a copy: an earlier copy drifted
out of date and left Codex reading a stale guide that still implied a CLI existed.
Do not paste the contents of `CLAUDE.md` in here — edit `CLAUDE.md` instead.
