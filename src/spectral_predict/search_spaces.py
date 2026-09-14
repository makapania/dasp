"""Opt-in extra hyperparameter axes for the unified Bayesian search (T-51).

This module NEVER defines the default search space. Defaults live where they always
have, in ``unified_bayesian.suggest_model_params`` / ``suggest_one_class_params``, and
are not touched. Everything here is additive and inert unless a caller names a bundle id.

Invariants (see ``docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md``):

- ``apply_extra_axes`` with nothing resolved returns ``params`` before doing any work.
- ``resolve_bundles`` is the single canonicalisation step. Its output feeds both the
  study identity and trial-time application, so the two cannot disagree.
- A bundle may only open parameters the base sampler does not already suggest. Optuna
  refuses a second ``suggest_*`` of the same name, so collisions are rejected up front
  with :class:`ExtraAxesConfigError`, before any storage is touched or trial runs.
- Bundle specs hold no callables. Write predicates are referenced by id from
  :data:`PREDICATES`, so a serialised identity fully determines behaviour.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

SPACE_SCHEMA_VERSION = 1

AxisKind = Literal["int", "float", "categorical"]

# Optuna names the unified objective suggests OUTSIDE the two model samplers. A bundle
# must not reuse them. tests/test_t51_extra_axes_mechanism.py checks this list against
# the objective's source so it cannot drift.
OBJECTIVE_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "preprocessing",
        "apply_baseline",
        "apply_smoothing",
        "apply_autoscale",
        "savgol_window",
        "subset_type",
        "n_vars",
        "region_id",
    }
)


class ExtraAxesConfigError(ValueError):
    """An extra-axes configuration is invalid.

    Raised before optimisation starts, and re-raised by the objective rather than being
    converted into a penalty trial.
    """


# Write predicates: decide whether a suggested value is written into model params.
# Suggestion itself is always unconditional (uniform parameter names for TPE). Adding
# or changing a predicate's meaning requires a NEW id, because ids are hashed.
PREDICATES: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "kernel_is_rbf": lambda p: p.get("kernel") == "rbf",
    "oc_kernel_is_poly": lambda p: p.get("kernel") == "poly",
    "oc_kernel_poly_or_sigmoid": lambda p: p.get("kernel") in ("poly", "sigmoid"),
}


@dataclass(frozen=True)
class AxisSpec:
    """One opt-in hyperparameter axis.

    Attributes:
        key: Key written into the model params dict.
        kind: Optuna distribution kind.
        low: Lower bound for int/float axes.
        high: Upper bound for int/float axes.
        choices: Ordered choices for categorical axes.
        log: Log-scale sampling for int/float axes.
        step: Step for int axes.
        param_name: Optuna parameter name; defaults to ``key``.
        applies_when_id: Key into :data:`PREDICATES` gating the write, or ``None``.
    """

    key: str
    kind: AxisKind
    low: Any = None
    high: Any = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    step: int | None = None
    param_name: str | None = None
    applies_when_id: str | None = None

    @property
    def optuna_name(self) -> str:
        return self.param_name or self.key


@dataclass(frozen=True)
class BundleSpec:
    """A named, curated group of extra axes for one or more model families.

    Attributes:
        id: Stable bundle id, as passed in ``enabled_extra_axes``.
        families: Canonical model names (after ``run_unified_bayesian`` normalisation).
        task_types: Task types the bundle applies to.
        axes: Axes suggested on every trial.
        constants: Fixed params the bundle also writes.
        label: Short GUI label.
        help: GUI/doc help text.
        revision: Bump on any semantic change to this bundle.
    """

    id: str
    families: frozenset[str]
    task_types: frozenset[str]
    axes: tuple[AxisSpec, ...]
    constants: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""
    help: str = ""
    revision: int = 1


# Curated registry. Empty in PR A: the mechanism ships first, bundles land in PR B/C.
BUNDLES: dict[str, BundleSpec] = {}


class _RecordingTrial:
    """Stand-in trial that records suggested names and follows a fixed branch path."""

    def __init__(self, categorical_choice: Mapping[str, int]) -> None:
        self._choice = categorical_choice
        self.names: list[str] = []
        self.categoricals: dict[str, int] = {}
        self.params: dict[str, Any] = {}

    def _record(self, name: str, value: Any) -> Any:
        self.names.append(name)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, **_: Any) -> int:
        return self._record(name, low)

    def suggest_float(self, name: str, low: float, high: float, **_: Any) -> float:
        return self._record(name, low)

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        self.categoricals[name] = len(choices)
        return self._record(name, choices[self._choice.get(name, 0)])


def discover_suggested_names(sampler: Callable[[Any], Any], max_paths: int = 512) -> frozenset[str]:
    """Return every Optuna name ``sampler(trial)`` can suggest, across categorical branches.

    Numeric suggestions take their lower bound; every combination of categorical choices
    that the sampler reveals is explored, so names suggested only on some branches (e.g.
    LightGBM ``num_leaves``) are found.
    """
    names: set[str] = set()
    seen: set[tuple[tuple[str, int], ...]] = set()
    frontier: list[dict[str, int]] = [{}]
    while frontier:
        path = frontier.pop()
        key = tuple(sorted(path.items()))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > max_paths:
            raise RuntimeError("discover_suggested_names: branch explosion; raise max_paths")
        trial = _RecordingTrial(path)
        sampler(trial)
        names.update(trial.names)
        for name, n_choices in trial.categoricals.items():
            if name in path:
                continue
            for index in range(n_choices):
                frontier.append({**path, name: index})
    return frozenset(names)


def resolve_bundles(
    model_name: str,
    task_type: str,
    enabled_extra_axes: Sequence[str] | None,
    search_space: Mapping[str, BundleSpec] | None = None,
    base_param_names: frozenset[str] = frozenset(),
) -> tuple[BundleSpec, ...]:
    """Validate and canonicalise enabled bundle ids for one model run.

    Args:
        model_name: Canonical model name.
        task_type: ``'regression'``, ``'classification'`` or ``'one_class'``.
        enabled_extra_axes: Bundle ids requested by the caller.
        search_space: Replaces :data:`BUNDLES` as the registry when given.
        base_param_names: Optuna names the base sampler can suggest for this model.

    Returns:
        Deep-copied applicable bundles, de-duplicated and sorted by id.

    Raises:
        ExtraAxesConfigError: Unknown bundle id, unknown predicate id, or a name
            collision with the base sampler, the objective, or another bundle.
    """
    if not enabled_extra_axes and search_space is None:
        return ()
    registry = BUNDLES if search_space is None else search_space
    requested = sorted(set(enabled_extra_axes or ()))
    unknown = [bundle_id for bundle_id in requested if bundle_id not in registry]
    if unknown:
        raise ExtraAxesConfigError(
            f"Unknown extra-axes bundle id(s) {unknown}. Valid ids: {sorted(registry)}"
        )
    applicable = [
        registry[bundle_id]
        for bundle_id in requested
        if model_name in registry[bundle_id].families
        and task_type in registry[bundle_id].task_types
    ]
    reserved = set(base_param_names) | OBJECTIVE_RESERVED_NAMES
    claimed: dict[str, str] = {}
    for bundle in applicable:
        if bundle.id != bundle.id.strip() or not bundle.id:
            raise ExtraAxesConfigError(f"Invalid bundle id {bundle.id!r}")
        for axis in bundle.axes:
            if axis.applies_when_id is not None and axis.applies_when_id not in PREDICATES:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis {axis.key!r}: unknown applies_when_id "
                    f"{axis.applies_when_id!r}. Valid ids: {sorted(PREDICATES)}"
                )
            name = axis.optuna_name
            if name in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis {name!r} collides with a parameter the base "
                    f"search already suggests for {model_name}; it cannot be opened additively."
                )
            if name in claimed:
                raise ExtraAxesConfigError(
                    f"Bundles {claimed[name]!r} and {bundle.id!r} both suggest {name!r}"
                )
            claimed[name] = bundle.id
    return tuple(copy.deepcopy(bundle) for bundle in applicable)


def _tagged(value: Any) -> Any:
    """Type-tag a value so ``1``, ``1.0``, ``True`` and ``'1'`` serialise distinctly."""
    if isinstance(value, (list, tuple)):
        return ["seq", [_tagged(v) for v in value]]
    if isinstance(value, Mapping):
        return ["map", [[str(k), _tagged(value[k])] for k in sorted(value, key=str)]]
    return [type(value).__name__, value if isinstance(value, (bool, int, float, str)) or value is None else repr(value)]


def canonical_space_identity(
    resolved: Sequence[BundleSpec], search_space_given: bool
) -> str | None:
    """Return a short digest of the effective extra-axes space, or ``None`` for the default.

    ``None`` means "append nothing to the study identity", which keeps default study names
    byte-for-byte unchanged. A caller-supplied ``search_space`` always yields a digest.
    """
    if not resolved and not search_space_given:
        return None
    payload = {
        "schema": SPACE_SCHEMA_VERSION,
        "custom_space": bool(search_space_given),
        "bundles": [
            {
                "id": bundle.id,
                "revision": bundle.revision,
                "constants": _tagged(dict(bundle.constants)),
                "axes": [
                    {
                        "key": axis.key,
                        "optuna_name": axis.optuna_name,
                        "kind": axis.kind,
                        "low": _tagged(axis.low),
                        "high": _tagged(axis.high),
                        "log": axis.log,
                        "step": _tagged(axis.step),
                        "choices": _tagged(axis.choices),
                        "applies_when_id": axis.applies_when_id,
                    }
                    for axis in bundle.axes
                ],
            }
            for bundle in sorted(resolved, key=lambda b: b.id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _suggest(trial: Any, axis: AxisSpec) -> Any:
    name = axis.optuna_name
    if axis.kind == "int":
        if axis.log:
            return trial.suggest_int(name, axis.low, axis.high, log=True)
        return trial.suggest_int(name, axis.low, axis.high, step=axis.step or 1)
    if axis.kind == "float":
        return trial.suggest_float(name, axis.low, axis.high, log=axis.log)
    if axis.kind == "categorical":
        return trial.suggest_categorical(name, list(axis.choices or ()))
    raise ExtraAxesConfigError(f"Unknown axis kind {axis.kind!r} for {name!r}")


def apply_extra_axes(
    trial: Any, params: dict[str, Any], resolved: Sequence[BundleSpec]
) -> dict[str, Any]:
    """Suggest and apply resolved extra axes. A no-op when nothing is resolved.

    Every axis is suggested on every trial so the parameter-name set stays uniform for
    TPE. A value is written only when its predicate holds, so inapplicable values do not
    split the fit fingerprint of otherwise identical fits.
    """
    if not resolved:
        return params
    already = set(trial.params)
    suggested: list[tuple[AxisSpec, Any]] = []
    for axis in itertools.chain.from_iterable(bundle.axes for bundle in resolved):
        if axis.optuna_name in already:
            raise ExtraAxesConfigError(
                f"Extra axis {axis.optuna_name!r} was already suggested in this trial"
            )
        suggested.append((axis, _suggest(trial, axis)))
    out = dict(params)
    for axis, value in suggested:
        if axis.applies_when_id is None or PREDICATES[axis.applies_when_id](params):
            out[axis.key] = value
    for bundle in resolved:
        out.update(bundle.constants)
    return out
