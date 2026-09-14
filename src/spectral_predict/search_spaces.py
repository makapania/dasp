"""Opt-in extra hyperparameter axes for the unified Bayesian search (T-51).

This module NEVER defines the default search space. Defaults live where they always
have, in ``unified_bayesian.suggest_model_params`` / ``suggest_one_class_params``, and
are not touched. Everything here is additive and inert unless a caller names a bundle id.

Invariants (see ``docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md``):

- ``apply_extra_axes`` with nothing resolved returns ``params`` before doing any work.
- ``resolve_bundles`` is the single canonicalisation step. Its output feeds both the
  study identity and trial-time application, so the two cannot disagree.
- A bundle may only open parameters the base sampler does not already suggest. Optuna 5
  does not fail on a second ``suggest_*`` of the same name: it warns and silently returns
  the first value, so a colliding bundle would do nothing. Collisions are therefore
  rejected up front with :class:`ExtraAxesConfigError`, before any storage is touched or
  trial runs.
- An effective space with no extra axes has no identity, so it never renames a study.
- Bundle specs hold no callables. Write predicates are referenced by id from
  :data:`PREDICATES`, so a serialised identity fully determines behaviour.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import numbers
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

    Guarantee is categorical branches only: a name suggested only for some *numeric*
    suggested value would be missed. No current base sampler has such a gate; if one is
    added, the runtime guard in :func:`apply_extra_axes` still aborts the run.
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
            raise ExtraAxesConfigError(
                "discover_suggested_names: branch explosion; raise max_paths"
            )
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
    if isinstance(enabled_extra_axes, str):
        raise ExtraAxesConfigError(
            f"enabled_extra_axes must be a sequence of bundle ids, not the string "
            f"{enabled_extra_axes!r}; wrap it: ({enabled_extra_axes!r},)"
        )
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
    claimed_names: dict[str, str] = {}
    claimed_keys: dict[str, str] = {}
    for bundle_id in requested:
        bundle = registry[bundle_id]
        if bundle.id != bundle_id:
            raise ExtraAxesConfigError(
                f"Registry key {bundle_id!r} holds a bundle whose id is {bundle.id!r}"
            )
        # Validate every requested bundle, not just those applying to this model, so a
        # shared multi-model selection fails on the first model, not mid-batch.
        if not bundle.axes and not bundle.constants:
            raise ExtraAxesConfigError(f"Bundle {bundle_id!r} has no axes and no constants")
        for axis in bundle.axes:
            _validate_axis(bundle_id, axis)
        for key, value in bundle.constants.items():
            _require_name(bundle_id, "constant key", key)
            _require_literal(bundle_id, f"constant {key!r}", value)
    for bundle in applicable:
        if not bundle.id or bundle.id != bundle.id.strip():
            raise ExtraAxesConfigError(f"Invalid bundle id {bundle.id!r}")
        for axis in bundle.axes:
            name = axis.optuna_name
            if axis.key in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis writes {axis.key!r}, which the base search "
                    f"already suggests for {model_name}; an alias Optuna name does not make "
                    "that additive."
                )
            if name in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis {name!r} collides with a parameter the base "
                    f"search already suggests for {model_name}; it cannot be opened additively."
                )
            if name in claimed_names:
                raise ExtraAxesConfigError(
                    f"Bundles {claimed_names[name]!r} and {bundle.id!r} both suggest {name!r}"
                )
            claimed_names[name] = bundle.id
            _claim_key(claimed_keys, axis.key, bundle.id)
        for key in bundle.constants:
            if key in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} constant {key!r} would override a parameter the "
                    f"search suggests for {model_name}"
                )
            _claim_key(claimed_keys, key, bundle.id)
    return tuple(copy.deepcopy(bundle) for bundle in applicable)


_LITERAL_TYPES = (bool, int, float, str, type(None))


def _require_literal(bundle_id: str, what: str, value: Any) -> None:
    # Identity hashing and Params round-trips need plain, order-stable literals.
    if not isinstance(value, _LITERAL_TYPES):
        raise ExtraAxesConfigError(
            f"Bundle {bundle_id!r} {what}: {type(value).__name__} is not allowed; "
            "use bool, int, float, str or None"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} {what}: non-finite {value!r}")


def _require_name(bundle_id: str, what: str, value: Any) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExtraAxesConfigError(
            f"Bundle {bundle_id!r} {what} must be a non-empty string, got {value!r}"
        )


def _claim_key(claimed: dict[str, str], key: str, bundle_id: str) -> None:
    if key in claimed:
        raise ExtraAxesConfigError(
            f"Bundles {claimed[key]!r} and {bundle_id!r} both write model param {key!r}"
        )
    claimed[key] = bundle_id


def _validate_axis(bundle_id: str, axis: AxisSpec) -> None:
    """Reject malformed axes before optimisation, so they never become penalty trials."""
    where = f"Bundle {bundle_id!r} axis {axis.key!r}"
    _require_name(bundle_id, "axis key", axis.key)
    if axis.param_name is not None:
        _require_name(bundle_id, "axis param_name", axis.param_name)
    if not isinstance(axis.log, bool):
        raise ExtraAxesConfigError(f"{where}: log must be a bool")
    if axis.applies_when_id is not None and axis.applies_when_id not in PREDICATES:
        raise ExtraAxesConfigError(
            f"{where}: unknown applies_when_id {axis.applies_when_id!r}. "
            f"Valid ids: {sorted(PREDICATES)}"
        )
    if axis.kind == "categorical":
        if any(v is not None for v in (axis.low, axis.high, axis.step)) or axis.log:
            raise ExtraAxesConfigError(f"{where}: categorical axes take choices only")
        if not axis.choices:
            raise ExtraAxesConfigError(f"{where}: categorical axis needs non-empty choices")
        for choice in axis.choices:
            _require_literal(bundle_id, f"axis {axis.key!r} choice", choice)
        return
    if axis.kind not in ("int", "float"):
        raise ExtraAxesConfigError(f"{where}: unknown kind {axis.kind!r}")
    if axis.choices is not None:
        raise ExtraAxesConfigError(f"{where}: numeric axes do not take choices")
    numeric = numbers.Integral if axis.kind == "int" else numbers.Real
    for bound in (axis.low, axis.high):
        if isinstance(bound, bool) or not isinstance(bound, numeric):
            raise ExtraAxesConfigError(f"{where}: {axis.kind} bounds must be {axis.kind}s")
        if not math.isfinite(bound):
            raise ExtraAxesConfigError(f"{where}: bounds must be finite, got {bound!r}")
    if axis.low > axis.high:
        raise ExtraAxesConfigError(f"{where}: low {axis.low!r} > high {axis.high!r}")
    if axis.log and axis.low <= 0:
        raise ExtraAxesConfigError(f"{where}: log scale requires low > 0")
    if axis.step is not None and (
        axis.kind != "int"
        or axis.log
        or isinstance(axis.step, bool)
        or not isinstance(axis.step, numbers.Integral)
        or axis.step < 1
    ):
        raise ExtraAxesConfigError(f"{where}: step is only valid for non-log int axes, int >= 1")


def _tagged(value: Any) -> Any:
    """Type-tag a value so ``1``, ``1.0``, ``True`` and ``'1'`` serialise distinctly.

    NumPy scalars are normalised to ``int``/``float`` first, so a bundle built from
    ``np.int64`` bounds hashes like its plain-Python equivalent.
    """
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
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
    byte-for-byte unchanged. Identity depends only on the effective space: a custom
    ``search_space`` that resolves to no applicable bundles is search-identical to the
    default, so it also yields ``None`` and resumes the default study. The same bundle
    content from the curated registry or a custom space hashes identically.
    ``search_space_given`` is accepted for call-site clarity and deliberately unused.
    """
    del search_space_given
    if not resolved:
        return None
    payload = {
        "schema": SPACE_SCHEMA_VERSION,
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

    Predicates see the base sampler's ``params`` only, never other extra axes or bundle
    constants, so a gate cannot depend on the order bundles are applied in.
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
