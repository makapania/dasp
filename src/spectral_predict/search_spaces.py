"""Opt-in extra hyperparameter axes for the unified Bayesian search (T-51).

This module NEVER defines the default search space. Defaults live where they always
have, in ``unified_bayesian.suggest_model_params`` / ``suggest_one_class_params``, and
are not touched. Everything here is additive and inert unless a caller names a bundle id.

Invariants (see ``docs/plans/2026-09-13-T51-optuna-axes-implementation-plan.md``):

- ``apply_extra_axes`` with nothing resolved returns ``params`` before doing any work.
- ``resolve_bundles`` is the single canonicalisation step. Its output feeds both the
  study identity and trial-time application, so the two cannot disagree.
- A bundle may only open parameters the base sampler neither suggests nor derives from
  a suggestion. Optuna 5.0 gives no reliable error for a clash (probed 2026-09-14): a
  same-kind duplicate ``suggest_*`` returns the first value, silently when the
  distribution is identical and with only an "Inconsistent parameter values" warning
  when it differs; only a different-kind duplicate raises. Overwriting a derived key
  such as MLP ``hidden_layer_sizes`` is invisible to Optuna altogether. Collisions are therefore
  rejected up front with :class:`ExtraAxesConfigError`, before any storage is touched or
  trial runs.
- An effective space with no extra axes has no identity, so it never renames a study.
  Identity is computed from each axis's effective Optuna distribution, so equivalent
  spellings (``0`` vs ``0.0``, ``step=None`` vs ``step=1``, NumPy vs Python scalars)
  share a study.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

SPACE_SCHEMA_VERSION = 1

AxisKind = Literal["int", "float", "categorical"]

# Largest integer bound accepted: TPE samples in float space, so integers beyond 2**53
# are not exactly representable there.
_MAX_INT_BOUND = 2**53

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
    """Stand-in trial: records suggestions, follows a fixed branch path and numeric mode.

    ``numeric_mode`` is ``("frac", f)`` — every numeric suggestion at fraction ``f`` of its
    range (geometric for log axes, snapped to the step grid for ints) — or
    ``("one", name)``: every numeric suggestion at its low bound except ``name`` at high.
    """

    def __init__(self, categorical_choice: Mapping[str, int], numeric_mode: Any) -> None:
        self._choice = categorical_choice
        self._mode = numeric_mode
        self.names: list[str] = []
        self.numeric_names: list[str] = []
        self.categoricals: dict[str, int] = {}
        self.params: dict[str, Any] = {}
        self.number = 0

    def _record(self, name: str, value: Any) -> Any:
        self.names.append(name)
        self.params[name] = value
        return value

    def _numeric(
        self, name: str, low: Any, high: Any, integral: bool, step: Any, log: bool
    ) -> Any:
        self.numeric_names.append(name)
        mode = self._mode
        if mode[0] == "one":
            fraction = 1.0 if mode[1] == name else 0.0
        else:
            fraction = mode[1]
        return self._record(name, _probe_value(low, high, fraction, integral, step, log))

    def suggest_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False, **_: Any
    ) -> int:
        return self._numeric(name, low, high, integral=True, step=step, log=log)

    def suggest_float(
        self, name: str, low: float, high: float, step: Any = None, log: bool = False, **_: Any
    ) -> float:
        return self._numeric(name, low, high, integral=False, step=step, log=log)

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        self.categoricals[name] = len(choices)
        return self._record(name, choices[self._choice.get(name, 0)])

    def set_user_attr(self, key: str, value: Any) -> None:  # tolerated, not recorded
        return None


_PROBE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _probe_value(
    low: Any, high: Any, fraction: float, integral: bool, step: Any, log: bool
) -> Any:
    """A valid point at ``fraction`` of a distribution's range, without overflow."""
    if fraction <= 0.0:
        return low
    if fraction >= 1.0:
        return high
    if log and low > 0:
        value = math.exp(math.log(low) + fraction * (math.log(high) - math.log(low)))
    else:
        value = low * (1.0 - fraction) + high * fraction
    value = min(max(value, low), high)
    if integral:
        grid = int(step or 1)
        value = int(low) + int((value - low) // grid) * grid
        value = min(max(value, int(low)), int(high))
    return value


def _explore(
    sampler: Callable[[Any], Any], max_paths: int
) -> Iterator[tuple[tuple[tuple[str, int], ...], _RecordingTrial, Any]]:
    """Run ``sampler`` on every categorical branch in several numeric modes.

    Modes per branch: all numeric suggestions at 0, 25, 50, 75 and 100 % of their ranges,
    plus each numeric suggestion alone at high (so two inputs that cancel at shared
    endpoints, e.g. ``x - y``, still expose a derived key). Yields the branch path too.
    """
    seen: set[tuple[tuple[str, int], ...]] = set()
    frontier: list[dict[str, int]] = [{}]
    while frontier:
        path = frontier.pop()
        path_key = tuple(sorted(path.items()))
        if path_key in seen:
            continue
        seen.add(path_key)
        if len(seen) > max_paths:
            raise ExtraAxesConfigError("base-sampler discovery: branch explosion; raise max_paths")
        low_trial = _RecordingTrial(path, ("frac", 0.0))
        yield path_key, low_trial, sampler(low_trial)
        modes: list[Any] = [("frac", f) for f in _PROBE_FRACTIONS[1:]]
        modes.extend(("one", name) for name in dict.fromkeys(low_trial.numeric_names))
        for mode in modes:
            trial = _RecordingTrial(path, mode)
            yield path_key, trial, sampler(trial)
        for name, n_choices in low_trial.categoricals.items():
            if name not in path:
                frontier.extend({**path, name: index} for index in range(n_choices))


def discover_suggested_names(sampler: Callable[[Any], Any], max_paths: int = 512) -> frozenset[str]:
    """Return every Optuna name ``sampler(trial)`` can suggest, across categorical branches.

    Guarantee is categorical branches only: a name suggested only for some *numeric*
    suggested value would be missed. No current base sampler has such a gate; if one is
    added, the runtime guard in :func:`apply_extra_axes` still aborts the run.
    """
    names: set[str] = set()
    for _, trial, _ in _explore(sampler, max_paths):
        names.update(trial.names)
    return frozenset(names)


def discover_derived_keys(sampler: Callable[[Any], Any], max_paths: int = 512) -> frozenset[str]:
    """Return output keys whose value depends on a suggestion (e.g. MLP ``hidden_layer_sizes``).

    A key is derived if its value differs across categorical branches or across the
    numeric probes (low, high, midpoint, and each input alone at high). Keys that always
    hold one fixed value (e.g. LightGBM ``reg_alpha``) are pinned constants and stay open
    to bundles. So does a key written with one fixed value on only some branches (SVM
    ``gamma='scale'`` under rbf), which is what gated bundles such as ``svm_gamma`` open.

    This is a probe, not a proof: a key that changes only at values no probe hits (e.g.
    ``x // 1000`` constant at every probe point) would be missed, and the runtime guard
    sees only ``trial.params``, not derived keys. The base samplers are pinned by hash in
    ``tests/test_t51_extra_axes_mechanism.py``, so any edit to them must re-audit this.
    """
    values: dict[str, set[str]] = {}
    presence: dict[tuple[tuple[str, int], ...], list[frozenset[str]]] = {}
    for path_key, _, output in _explore(sampler, max_paths):
        keys = frozenset(str(k) for k in output) if isinstance(output, Mapping) else frozenset()
        presence.setdefault(path_key, []).append(keys)
        if isinstance(output, Mapping):
            for key, value in output.items():
                values.setdefault(str(key), set()).add(repr(value))
    derived = {key for key, seen in values.items() if len(seen) > 1}
    # A key present for some numeric values but not others on the SAME categorical branch
    # depends on a numeric suggestion, even if only one value was ever observed. Presence
    # that varies only BETWEEN branches (SVM gamma under rbf) stays open.
    for runs in presence.values():
        derived |= frozenset().union(*runs) - frozenset.intersection(*runs)
    return frozenset(derived)


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
        base_param_names: Optuna names the base sampler suggests plus model-param keys it
            derives from them, for this model. Neither may be suggested or written.

    Returns:
        Deep-copied applicable bundles, de-duplicated and sorted by id.

    Raises:
        ExtraAxesConfigError: Malformed selection or bundle, unknown bundle or predicate
            id, or a name/key collision with the base sampler, the objective, or another
            bundle.
    """
    if isinstance(enabled_extra_axes, str):
        raise ExtraAxesConfigError(
            f"enabled_extra_axes must be a sequence of bundle ids, not the string "
            f"{enabled_extra_axes!r}; wrap it: ({enabled_extra_axes!r},)"
        )
    if enabled_extra_axes is not None and not isinstance(
        enabled_extra_axes, (list, tuple, set, frozenset)
    ):
        # Concrete collections only: a generator would be exhausted by the first read,
        # and run_unified_bayesian reads the selection more than once.
        raise ExtraAxesConfigError(
            f"enabled_extra_axes must be a sequence of bundle ids, got {enabled_extra_axes!r}"
        )
    if not enabled_extra_axes and search_space is None:
        return ()
    ids = list(enabled_extra_axes or ())
    bad_ids = [bundle_id for bundle_id in ids if not isinstance(bundle_id, str)]
    if bad_ids:
        raise ExtraAxesConfigError(f"Bundle ids must be strings, got {bad_ids!r}")
    registry = BUNDLES if search_space is None else search_space
    requested = sorted(set(ids))
    unknown = [bundle_id for bundle_id in requested if bundle_id not in registry]
    if unknown:
        raise ExtraAxesConfigError(
            f"Unknown extra-axes bundle id(s) {unknown}. Valid ids: {sorted(registry)}"
        )
    # Validate every requested bundle, not just those applying to this model, so a
    # shared multi-model selection fails on the first model, not mid-batch.
    for bundle_id in requested:
        _validate_bundle(bundle_id, registry[bundle_id])
    applicable = [
        registry[bundle_id]
        for bundle_id in requested
        if model_name in registry[bundle_id].families
        and task_type in registry[bundle_id].task_types
    ]
    reserved = set(base_param_names) | OBJECTIVE_RESERVED_NAMES
    # Owners are structured tuples: string owners could be forged by a key such as
    # "const:tol" and make an axis indistinguishable from a constant.
    names: dict[str, tuple[str, ...]] = {}  # Optuna name -> owner
    keys: dict[str, tuple[str, ...]] = {}  # written model-param key -> owner
    for bundle in applicable:
        for index, axis in enumerate(bundle.axes):
            owner = ("axis", bundle.id, str(index))
            name = axis.optuna_name
            if axis.key in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis writes {axis.key!r}, which the base search "
                    f"already suggests or derives for {model_name}; an alias Optuna name "
                    "does not make that additive."
                )
            if name in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} axis {name!r} collides with a parameter the base "
                    f"search already suggests for {model_name}; it cannot be opened additively."
                )
            _claim(names, name, owner, "both suggest Optuna parameter")
            _claim(keys, axis.key, owner, "both write model param")
            _no_cross(names, keys, name, axis.key, owner)
        for key in bundle.constants:
            owner = ("constant", bundle.id, key)
            if key in reserved:
                raise ExtraAxesConfigError(
                    f"Bundle {bundle.id!r} constant {key!r} would override a parameter the "
                    f"search suggests or derives for {model_name}"
                )
            _claim(keys, key, owner, "both write model param")
            _no_cross(names, keys, None, key, owner)
    return tuple(copy.deepcopy(bundle) for bundle in applicable)


def _claim(
    claimed: dict[str, tuple[str, ...]], item: str, owner: tuple[str, ...], what: str
) -> None:
    if item in claimed:
        raise ExtraAxesConfigError(f"{claimed[item]!r} and {owner!r} {what} {item!r}")
    claimed[item] = owner


def _no_cross(
    names: dict[str, tuple[str, ...]],
    keys: dict[str, tuple[str, ...]],
    name: str | None,
    key: str,
    owner: tuple[str, ...],
) -> None:
    # One axis's Optuna name must not be another axis's (or constant's) written key: the
    # trial would record one value under that name while the model is fitted with another.
    if name is not None and keys.get(name, owner) != owner:
        raise ExtraAxesConfigError(
            f"{owner!r} suggests Optuna parameter {name!r}, which {keys[name]!r} writes as a key"
        )
    if names.get(key, owner) != owner:
        raise ExtraAxesConfigError(
            f"{owner!r} writes key {key!r}, which {names[key]!r} suggests as an Optuna parameter"
        )


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


def _require_str_set(bundle_id: str, what: str, value: Any) -> None:
    # A bare string would substring-match ("PLS" in "PLS-DA").
    if isinstance(value, (str, bytes)) or not isinstance(value, (set, frozenset, tuple, list)):
        raise ExtraAxesConfigError(
            f"Bundle {bundle_id!r} {what} must be a set of strings, got {value!r}"
        )
    for item in value:
        _require_name(bundle_id, what, item)


def _validate_bundle(bundle_id: str, bundle: BundleSpec) -> None:
    if bundle.id != bundle_id:
        raise ExtraAxesConfigError(
            f"Registry key {bundle_id!r} holds a bundle whose id is {bundle.id!r}"
        )
    _require_name(bundle_id, "id", bundle.id)
    _require_str_set(bundle_id, "families", bundle.families)
    _require_str_set(bundle_id, "task_types", bundle.task_types)
    if not bundle.families or not bundle.task_types:
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} must name families and task_types")
    if not isinstance(bundle.axes, (tuple, list)) or not all(
        isinstance(axis, AxisSpec) for axis in bundle.axes
    ):
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} axes must be a tuple of AxisSpec")
    if not isinstance(bundle.constants, Mapping):
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} constants must be a mapping")
    if (
        isinstance(bundle.revision, bool)
        or not isinstance(bundle.revision, int)
        or bundle.revision < 1
    ):
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} revision must be an int >= 1")
    if not bundle.axes and not bundle.constants:
        raise ExtraAxesConfigError(f"Bundle {bundle_id!r} has no axes and no constants")
    for axis in bundle.axes:
        _validate_axis(bundle_id, axis)
    for key, value in bundle.constants.items():
        _require_name(bundle_id, "constant key", key)
        _require_literal(bundle_id, f"constant {key!r}", value)


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
        tagged = [json.dumps(_tagged(choice)) for choice in axis.choices]
        if len(set(tagged)) != len(tagged):
            raise ExtraAxesConfigError(f"{where}: duplicate categorical choices")
        return
    if axis.kind not in ("int", "float"):
        raise ExtraAxesConfigError(f"{where}: unknown kind {axis.kind!r}")
    if axis.choices is not None:
        raise ExtraAxesConfigError(f"{where}: numeric axes do not take choices")
    numeric = numbers.Integral if axis.kind == "int" else numbers.Real
    for bound in (axis.low, axis.high):
        if isinstance(bound, bool) or not isinstance(bound, numeric):
            raise ExtraAxesConfigError(f"{where}: {axis.kind} bounds must be {axis.kind}s")
        if axis.kind == "int":
            if abs(int(bound)) > _MAX_INT_BOUND:
                raise ExtraAxesConfigError(f"{where}: int bounds must be within +/-2**53")
        else:
            try:
                finite = math.isfinite(float(bound))
            except OverflowError:
                finite = False
            if not finite:
                raise ExtraAxesConfigError(f"{where}: bounds must be finite, got {bound!r}")
    if axis.low > axis.high:
        raise ExtraAxesConfigError(f"{where}: low {axis.low!r} > high {axis.high!r}")
    if axis.kind == "float" and not math.isfinite(float(axis.high) - float(axis.low)):
        raise ExtraAxesConfigError(f"{where}: bound span overflows a float")
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
    if axis.step is not None and (int(axis.high) - int(axis.low)) % int(axis.step) != 0:
        # Optuna would silently lower `high`; requiring alignment keeps one spelling per
        # effective distribution (and so one study identity).
        raise ExtraAxesConfigError(f"{where}: (high - low) must be a multiple of step")


def _tagged(value: Any) -> Any:
    """Type-tag a value so ``1``, ``1.0``, ``True`` and ``'1'`` serialise distinctly.

    Numeric and ``str`` subclasses (e.g. NumPy scalars used as int/float *bounds*, or
    ``np.str_`` choices) are normalised to ``int``/``float``/``str`` first, and ``-0.0``
    to ``0.0``, so equivalent values hash alike. NumPy numeric scalars are still rejected
    as categorical choices and constants by validation, because those values are written
    into model params and must round-trip through the leaderboard ``Params`` string
    (``repr(np.int64(3))`` does not survive ``ast.literal_eval``).
    """
    if isinstance(value, str):
        value = str(value)
    elif isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value) + 0.0
    if isinstance(value, (list, tuple)):
        return ["seq", [_tagged(v) for v in value]]
    if isinstance(value, Mapping):
        return ["map", [[str(k), _tagged(value[k])] for k in sorted(value, key=str)]]
    return [type(value).__name__, value if isinstance(value, _LITERAL_TYPES) else repr(value)]


def _distribution_payload(axis: AxisSpec) -> dict[str, Any]:
    """Serialise an axis by its effective Optuna distribution, not its spelling."""
    if axis.kind == "categorical":
        return {"kind": "categorical", "choices": _tagged(axis.choices)}
    if axis.kind == "int":
        return {
            "kind": "int",
            "low": int(axis.low),
            "high": int(axis.high),
            "log": axis.log,
            "step": None if axis.log else int(axis.step or 1),
        }
    return {
        "kind": "float",
        "low": float(axis.low) + 0.0,
        "high": float(axis.high) + 0.0,
        "log": axis.log,
    }


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
                        "distribution": _distribution_payload(axis),
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
            return trial.suggest_int(name, int(axis.low), int(axis.high), log=True)
        return trial.suggest_int(name, int(axis.low), int(axis.high), step=int(axis.step or 1))
    if axis.kind == "float":
        return trial.suggest_float(name, float(axis.low), float(axis.high), log=axis.log)
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
        for clash in {axis.optuna_name, axis.key} & already:
            raise ExtraAxesConfigError(
                f"Extra axis {axis.optuna_name!r} (key {axis.key!r}) clashes with "
                f"{clash!r}, already suggested in this trial"
            )
        suggested.append((axis, _suggest(trial, axis)))
    for bundle in resolved:
        for clash in set(bundle.constants) & already:
            raise ExtraAxesConfigError(
                f"Bundle {bundle.id!r} constant {clash!r} clashes with a suggested parameter"
            )
    out = dict(params)
    for axis, value in suggested:
        if axis.applies_when_id is None or PREDICATES[axis.applies_when_id](params):
            out[axis.key] = value
    for bundle in resolved:
        out.update(bundle.constants)
    return out
