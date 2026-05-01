# T-35: T-11 type-design follow-ups

**Status:** PLANNED — deferred from T-11 PR #6 (type-design-analyzer suggestions).
**Filed:** 2026-04-30.
**Source:** type-design-analyzer pass during T-11 PR #6 review.
**Priority:** LOW — none of these are correctness bugs; they're maintainability + future-proofing improvements.

## Three independent items

### Item 1: `RunMetadata` should be `frozen=True, slots=True` with `__post_init__` validation

**Current state:** `src/spectral_predict/run_state.py:RunMetadata` is a
mutable `@dataclasses.dataclass` with no field validation.

**Problem:** Anyone holding a reference can mutate after construction
(`meta.run_id = "lol"`), which would diverge in-memory state from the
on-disk sidecar. T-11 already has the pattern locked down in practice
(no mutators exist in the call graph), but the type doesn't enforce it.

**Fix:**

```python
@dataclasses.dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    storage_path: str
    storage_url: str
    label: str | None
    dataset_fingerprint: str | None
    model_names: tuple[str, ...]   # tuple, not list — frozen requires hashable-ish
    n_trials_per_model: int | None
    started_iso: str

    def __post_init__(self):
        if not self.run_id or len(self.run_id) != 12:
            raise ValueError(f"run_id must be 12 hex chars, got {self.run_id!r}")
        if not self.storage_url.startswith("sqlite:///"):
            raise ValueError(f"storage_url must be sqlite:///..., got {self.storage_url!r}")
        if self.n_trials_per_model is not None and self.n_trials_per_model <= 0:
            raise ValueError(f"n_trials_per_model must be positive, got {self.n_trials_per_model}")
        # Don't validate started_iso parses as datetime — older sidecars
        # may have non-strict ISO formats; tolerant read is preferred.
```

**Knock-on changes:**
- `model_names: list[str]` → `tuple[str, ...]`. Anywhere the list is
  appended to (T-34's `models_completed` would have been the case;
  switch that field to `frozenset` or `tuple` and rewrite via
  `dataclasses.replace`).
- `from_dict(cls, data)` may need to coerce `data["model_names"]`
  from JSON list to tuple.
- `to_dict` is unchanged — `dataclasses.asdict` flattens tuples to
  lists automatically for JSON.

**Cost:** ~30 minutes plus test pass. Risk: low (frozen dataclasses
are mature in Python 3.10+; we're on 3.12).

### Item 2: Extract `_LineBuffer` helper from `_TeeStream`

**Current state:** `src/spectral_predict/run_logging.py:_TeeStream` has
twin counters `_buffer: list[str]` + `_buffer_bytes: int` that must
maintain the invariant `_buffer_bytes == sum(len(p) for p in _buffer)`.
Four mutation sites (`write` append, `write` flush-then-tail-rebuf,
`flush` drain) all need to keep them in sync.

**Problem:** Disciplinary invariant, not structural. A future
contributor adding a fifth mutation point (or refactoring an existing
one) can easily violate it.

**Fix:** Extract a tiny helper:

```python
class _LineBuffer:
    """Holds line fragments + a maintained byte count.

    Keeps `_buffer_bytes == sum(len(p) for p in _buffer)` invariant
    structural rather than disciplinary. All callers go through
    `append` / `drain` / `__len__`.
    """
    def __init__(self):
        self._parts: list[str] = []
        self._bytes = 0

    def append(self, text: str) -> None:
        self._parts.append(text)
        self._bytes += len(text)

    def drain(self) -> str:
        joined = "".join(self._parts)
        self._parts = []
        self._bytes = 0
        return joined

    def __len__(self) -> int:
        return self._bytes
```

`_TeeStream` then composes one `_LineBuffer` instance and `with self._lock:`
brackets calls to `append` / `drain`. The lock stays in `_TeeStream`
(it's about cross-thread coordination, not buffer integrity).

**Cost:** ~15 minutes. Risk: low. Test surface unchanged (the existing
4 tee tests cover the buffer behavior end-to-end).

### Item 3: Add missing file-protocol attrs to `_TeeStream`

**Current state:** `_TeeStream` implements `write` / `flush` / `isatty`
/ `fileno`. Missing: `closed`, `encoding`, `errors`, `newlines`, `mode`,
`name`, `writable()`, `readable()`, `seekable()`, `__iter__`,
`writelines()`, `__enter__`/`__exit__`.

**Problem:** Some libraries probe these when given a `sys.stdout`
replacement. `print()` only needs write+flush, but `sys.stdout.encoding`
is consulted by:
- `click` for terminal color decisions
- some Optuna formatters (verbose progress paths)
- `prompt-toolkit` (not used by dasp directly but transitively pulled in
  by some sklearn extensions)

When these probe and the attr is missing, they raise `AttributeError`.
In the bundle this is invisible (no console anyway), but in dev the
worker thread could crash.

**Fix (minimum viable):**

```python
class _TeeStream:
    closed = False  # We never explicitly close; tee lives for the process.
    encoding = "utf-8"
    errors = "strict"
    mode = "w"

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def writelines(self, lines):
        for line in lines:
            self.write(line)
```

`name`, `__iter__`, context-manager support are deferred unless a real
caller needs them.

**Cost:** ~10 minutes. Risk: very low.

## Combined commit shape

Recommend a single PR with three small commits:

1. `refactor(run_state): freeze RunMetadata + add __post_init__ validation`
2. `refactor(run_logging): extract _LineBuffer to make buffer invariant structural`
3. `refactor(run_logging): add file-protocol attrs to _TeeStream for compat`

Each commit independently revertable. Each touches its own area.

## Open questions

1. **Should `model_names` be `frozenset` or `tuple`?** Frozenset disallows
   duplicates and is order-insensitive, which matches the semantic ("the
   set of models we're running"). Tuple preserves insertion order and
   permits duplicates. Order-preservation matters for the GUI's per-model
   progress display, so tuple wins.
2. **Should `__post_init__` validate `started_iso` parses?** Tempting,
   but `datetime.fromisoformat` semantics changed across Python versions.
   Skip for now; tolerant read.
3. **`_TeeStream.close()` — needed?** `sys.stdout` doesn't get explicitly
   closed in dasp's bundle path; the tee lives for the process. Adding
   `close()` would invite "do I need to teardown?" questions. Skip.

## Success criteria

- All 47 T-11 tests still pass after the refactor.
- A new test verifies `dataclasses.replace(meta, label="...")` works as
  expected on the frozen `RunMetadata`.
- A new test verifies `_TeeStream.encoding == "utf-8"` and `writable()
  is True`.
- No behavior change visible to callers.

## Estimated effort

~1 hour for all three items + tests.
