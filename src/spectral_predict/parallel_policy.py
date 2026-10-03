"""One thread-budget rule for cross-validation and model fitting.

Nested parallelism oversubscribes the CPU: a booster built with ``n_jobs=-1`` inside a
``Parallel(n_jobs=-1)`` fold pool starts (workers x cores) threads, and every fold
fights every other fold for the same cores. ``OMP_NUM_THREADS=1`` does not help,
because LightGBM/XGBoost/RandomForest resolve an explicit ``n_jobs=-1`` themselves.

The rule this module owns:

* Inside a parallel CV pool the cores are split, never multiplied: the pool has
  ``min(n_splits, physical cores)`` workers and each model fit gets
  ``physical cores // workers`` threads (``n_jobs``; CatBoost ``thread_count``).
  With as many folds as cores (LOO, many folds) that is ``n_jobs=1``.
* Tiny jobs (few folds x small data) run serially with single-threaded fits: pool
  dispatch and thread wake-up cost more than the fits.
* Model families listed in :data:`MODELS_PREFER_SERIAL_CV` run their folds serially
  and keep whatever threading they were built with.
* A fit outside any pool (refits, final models) keeps the model's own setting and
  may use all cores.
* One-class CV and SIMCA's cross-fitted null run under
  :func:`openmp_single_threaded`, because their cost on spectral-sized data is
  OpenMP start-up, not arithmetic.

The frozen (PyInstaller) bundle cannot spawn loky workers, so parallel plans use the
``threading`` backend there (see :func:`frozen_needs_threading_fallback`). Thread
workers share the process's BLAS pool, so :meth:`CVPlan.backend_context` caps it at
the per-fit budget; CatBoost folds run serially there (see :func:`plan_cv`).

Thread settings are runtime-only: callers apply :func:`limit_estimator_threads` to a
*clone* used for the folds, so ``n_jobs`` never changes captured result-row params,
fit fingerprints or study hashes.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

# Fold work below this many matrix cells (n_splits x n_samples x n_features) runs
# serially with single-threaded fits. Measured (24-core Windows box, warm loky pool,
# 5 folds): at 5-10k cells a pool only breaks even with a serial single-threaded
# loop, from ~20k cells up the pool wins 1.5-4x. Serial fits that keep n_jobs=-1 lose
# to serial n_jobs=1 at every size below ~50k cells. See docs/SESSION_LOG.md (QW1).
TINY_JOB_CELLS = 10_000

# Models whose folds run serially: the fits are so fast that pool overhead dominates
# (PLS, PLS-DA, Ridge, Lasso, ElasticNet), or the model's own threading conflicts
# with the pool (SVM).
MODELS_PREFER_SERIAL_CV = frozenset({"SVM", "PLS", "PLS-DA", "Ridge", "Lasso", "ElasticNet"})

_F = TypeVar("_F", bound=Callable[..., Any])


def frozen_needs_threading_fallback() -> bool:
    """Whether the current frozen build needs the threading-backend workaround.

    PyInstaller windowed bundles cannot safely use loky's spawn method
    regardless of Python version.  The frozen runtime hook's
    multiprocessing.freeze_support() crashes on argv parsing in child
    processes ("ValueError: not enough values to unpack (expected 2, got
    1)"), and the parent retries spawning -> fork-bomb of GUI windows.
    Falling back to the threading backend avoids the broken spawn entirely.
    """
    return bool(getattr(sys, "frozen", False) or "__compiled__" in globals())


@functools.lru_cache(maxsize=1)
def physical_cores() -> int:
    """Number of physical CPU cores (at least 1)."""
    from joblib import cpu_count

    try:
        return max(1, int(cpu_count(only_physical_cores=True)))
    # Parenthesised on purpose: the unparenthesised form is 3.14-only syntax and the
    # documented 3.12 rollback build must still import this module.
    except (TypeError, ValueError, OSError):  # fmt: skip
        return max(1, int(cpu_count()))


@dataclass(frozen=True)
class CVPlan:
    """How to run the folds of one cross-validation.

    Attributes:
        n_jobs: Outer pool size. 1 means a serial loop in the calling thread.
        backend: joblib backend for the pool (``"loky"`` or ``"threading"``), or
            ``"sequential"`` for a serial plan.
        model_threads: Threads each model fit may use, or ``None`` to leave the
            estimator as it was built.
    """

    n_jobs: int
    backend: str
    model_threads: int | None

    @property
    def parallel(self) -> bool:
        """True when the folds go to a worker pool."""
        return self.n_jobs > 1

    @contextlib.contextmanager
    def backend_context(self) -> Iterator[None]:
        """Context to run this plan's pool in. Wrap every parallel fold execution in it.

        * Routes joblib calls made inside sklearn (``cross_val_predict``,
          ``learning_curve``) to this plan's backend; without it they use loky, which
          the frozen bundle cannot spawn.
        * For the ``threading`` backend, caps the process-wide BLAS pools at
          ``model_threads``. Thread workers share one BLAS pool, and models without an
          ``n_jobs`` (MLP, PLS inside a pipeline ...) would otherwise each fan out to
          every core. Loky workers need no cap: joblib already sets the BLAS/OpenMP
          environment of each worker to cores // workers.
        """
        if not self.parallel:
            yield
            return
        from joblib import parallel_config

        with contextlib.ExitStack() as stack:
            stack.enter_context(parallel_config(backend=self.backend))
            if self.backend == "threading" and self.model_threads is not None:
                stack.enter_context(native_thread_limit(self.model_threads, "blas"))
            yield


def plan_cv(
    n_splits: int,
    n_samples: int,
    n_features: int,
    *,
    model_name: str | None = None,
    requested_n_jobs: int | None = -1,
) -> CVPlan:
    """Decide pool size, backend and per-model threads for one cross-validation.

    Args:
        n_splits: Number of fits the CV performs (folds x repeats).
        n_samples: Rows in the matrix being cross-validated.
        n_features: Columns in the matrix being cross-validated.
        model_name: Model family; families in :data:`MODELS_PREFER_SERIAL_CV` run
            serially with their own threading.
        requested_n_jobs: Caller's upper bound on the pool, in joblib's convention.
            ``1`` forces a serial loop that leaves the model untouched; ``-1``/``None``
            means no bound; other negatives count back from the logical CPU count
            (``-2`` = all but one). ``0`` is invalid, as in joblib.

    Returns:
        The plan to run the folds with.

    Raises:
        ValueError: If ``requested_n_jobs`` is 0 (an earlier version silently treated
            it as "no bound").

    Note:
        A single split above :data:`TINY_JOB_CELLS` gets ``model_threads=None``: the
        one fit keeps the estimator's own ``n_jobs`` (often all cores). For
        LightGBM/XGBoost the thread count can change histogram-reduction order, so
        that fit may differ in the last bits from a single-threaded one.
    """
    if requested_n_jobs == 0:
        raise ValueError("requested_n_jobs == 0 has no meaning (joblib convention); use 1 or -1")
    if requested_n_jobs == 1 or (model_name is not None and model_name in MODELS_PREFER_SERIAL_CV):
        return CVPlan(n_jobs=1, backend="sequential", model_threads=None)

    n_splits = max(1, int(n_splits))
    tiny = n_splits * int(n_samples) * int(n_features) < TINY_JOB_CELLS
    if tiny:
        return CVPlan(n_jobs=1, backend="sequential", model_threads=1)
    if n_splits < 2:
        # One big fit: nothing to pool, so it may use every core.
        return CVPlan(n_jobs=1, backend="sequential", model_threads=None)

    backend = _pool_backend()
    if _catboost_needs_serial(backend, model_name):
        return CVPlan(n_jobs=1, backend="sequential", model_threads=None)

    workers = min(n_splits, physical_cores())
    if requested_n_jobs is not None and requested_n_jobs != -1:
        workers = min(workers, _resolve_n_jobs(requested_n_jobs))
    if workers <= 1:
        return CVPlan(n_jobs=1, backend="sequential", model_threads=None)

    return CVPlan(
        n_jobs=workers, backend=backend, model_threads=max(1, physical_cores() // workers)
    )


def _pool_backend() -> str:
    """joblib backend for a worker pool: threads in a frozen bundle, else processes."""
    return "threading" if frozen_needs_threading_fallback() else "loky"


def _catboost_needs_serial(backend: str, model_name: str | None) -> bool:
    """CatBoost runs serially, with its own threading, in a threading pool.

    Its post-fit feature importance and its predict calls ignore the constructor
    ``thread_count`` and use every core; in a shared-process thread pool those phases
    would multiply. (Each loky worker is its own process, so there it is tolerated.)
    """
    return backend == "threading" and model_name == "CatBoost"


def task_pool_plan(n_jobs: int | None, *, model_name: str | None = None) -> CVPlan:
    """Plan for a caller-sized pool of independent tasks (one per candidate config).

    Workers are ``n_jobs`` capped at physical cores (:func:`pool_workers`); each task's
    fits get cores // workers threads; the backend follows the frozen rule; CatBoost
    under the threading backend runs serially, as in :func:`plan_cv`. Run the pool as
    ``Parallel(n_jobs=plan.n_jobs, backend=plan.backend)`` inside
    ``plan.backend_context()``.
    """
    backend = _pool_backend()
    workers = pool_workers(n_jobs)
    if workers <= 1 or _catboost_needs_serial(backend, model_name):
        return CVPlan(n_jobs=1, backend="sequential", model_threads=None)
    return CVPlan(n_jobs=workers, backend=backend, model_threads=pool_model_threads(n_jobs))


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """joblib's n_jobs convention resolved to a worker count (at least 1)."""
    from joblib import effective_n_jobs

    return max(1, int(effective_n_jobs(n_jobs)))


def pool_workers(n_jobs: int | None) -> int:
    """Pool size to use for a caller-sized joblib pool: ``n_jobs`` capped at physical cores.

    ``n_jobs=-1`` means every *logical* CPU to joblib; on a hyper-threaded machine that
    is twice the physical cores, and single-threaded fits would still oversubscribe.
    """
    return min(_resolve_n_jobs(n_jobs), physical_cores())


def pool_model_threads(n_jobs: int | None) -> int:
    """Threads each model fit may use inside a pool of :func:`pool_workers` (``n_jobs``).

    For pools this module does not size itself (e.g. one task per candidate
    configuration): the cores are split between the workers, never multiplied. Size
    the pool itself with :func:`pool_workers`, not the raw ``n_jobs``.
    """
    return max(1, physical_cores() // pool_workers(n_jobs))


def _is_estimator(value: Any) -> bool:
    return hasattr(value, "get_params") and not isinstance(value, type)


def _sub_estimators(est: Any) -> Iterator[Any]:
    """Estimators held directly by ``est``.

    Covers estimator-valued params (``GridSearchCV.estimator``, wrappers) and
    containers of them in any params: pipeline ``steps`` and ``VotingRegressor`` /
    ``StackingRegressor`` ``estimators`` (lists of ``(name, estimator)``),
    ``ColumnTransformer`` ``transformers`` (``(name, estimator, columns)``), plain
    lists/tuples of estimators, and dict values. ``"passthrough"``/``"drop"``/``None``
    entries are skipped. Objects without ``get_params`` but with a ``steps`` list are
    walked through it.
    """

    def from_container(value: Any) -> Iterator[Any]:
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            if _is_estimator(item):
                yield item
            elif isinstance(item, tuple) and len(item) >= 2 and _is_estimator(item[1]):
                yield item[1]

    if hasattr(est, "get_params") and not isinstance(est, type):
        values = est.get_params(deep=False).values()
    else:
        steps = getattr(est, "steps", None)
        values = [steps] if isinstance(steps, list) else []
    for value in values:
        if _is_estimator(value):
            yield value
        elif isinstance(value, (list, tuple, dict)):
            yield from from_container(value)


def _walk_estimators(est: Any) -> Iterator[Any]:
    """``est`` and every estimator nested in it, depth-first, each object once.

    No depth cap: an id-based visited set stops cycles instead, so long wrapper
    chains and ensembles of pipelines are walked to the bottom.
    """
    seen: set[int] = set()
    stack = [est]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend(reversed(list(_sub_estimators(current))))


def contains_catboost(estimator: Any) -> bool:
    """True if a CatBoost model is anywhere in ``estimator``.

    Walks the same structure :func:`limit_estimator_threads` caps: pipeline steps in
    any position, estimator-valued params and ensembles' ``estimators`` lists.
    """
    return any(
        (type(est).__module__ or "").startswith("catboost") for est in _walk_estimators(estimator)
    )


def _set_threads(est: Any, n_threads: int) -> None:
    """Set the thread count on ``est`` and on every estimator nested in it, in place.

    ``sklearn.clone`` (in :func:`limit_estimator_threads`) has already deep-copied the
    nested estimators, so capping them in place never touches the caller's object.
    """
    for current in _walk_estimators(est):
        if not hasattr(current, "set_params"):
            continue
        module = type(current).__module__ or ""
        if module.startswith("imblearn"):
            # Resamplers' n_jobs is deprecated and irrelevant to the thread budget.
            continue
        if module.startswith("catboost"):
            current.set_params(thread_count=n_threads)
            continue
        if "n_jobs" in current.get_params(deep=False):
            current.set_params(n_jobs=n_threads)


def limit_estimator_threads(estimator: Any, n_threads: int | None) -> Any:
    """Return a clone of ``estimator`` whose fits use at most ``n_threads`` threads.

    Covers ``n_jobs`` (sklearn, LightGBM, XGBoost) and CatBoost's ``thread_count``,
    on a bare estimator or on every step of a (possibly imblearn) Pipeline. The
    input is never mutated, so params captured from it stay as built.

    Args:
        estimator: Unfitted estimator or pipeline.
        n_threads: Thread count, or ``None`` to return ``estimator`` unchanged.
    """
    if n_threads is None:
        return estimator
    from sklearn.base import clone

    capped = clone(estimator)
    _set_threads(capped, int(n_threads))
    return capped


def estimator_threads(estimator: Any) -> dict[str, Any]:
    """Map each step of ``estimator`` to its ``n_jobs``/``thread_count`` (for tests and logs)."""
    out: dict[str, Any] = {}
    steps = getattr(estimator, "steps", None)
    items = steps if isinstance(steps, list) else [("estimator", estimator)]
    for name, step in items:
        if not hasattr(step, "get_params"):
            continue
        params = step.get_params(deep=False)
        if "thread_count" in params or (type(step).__module__ or "").startswith("catboost"):
            out[name] = params.get("thread_count")
        elif "n_jobs" in params:
            out[name] = params["n_jobs"]
    return out


_CONTROLLER: Any = None
_LIMIT_LOCK = threading.Lock()
# user_api -> {"limits": the n_threads of every open context (a multiset),
#              "originals": the first context's limiter on this API's libraries only,
#                           which holds their pre-cap thread counts}
_ACTIVE_LIMITS: dict[str, dict[str, Any]] = {}


def _controller(user_api: str) -> Any:
    """Cached threadpoolctl controller (building one scans loaded libraries, ~8 ms).

    Rebuilt when it knows no library for ``user_api`` (e.g. it was created before that
    runtime loaded). A runtime loaded later than one already known is not picked up;
    sklearn loads its OpenMP and BLAS runtimes at import, before any limit is taken.
    """
    global _CONTROLLER
    from threadpoolctl import ThreadpoolController

    if _CONTROLLER is None or not _CONTROLLER.select(user_api=user_api).lib_controllers:
        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER


@contextlib.contextmanager
def native_thread_limit(n_threads: int, user_api: str) -> Iterator[None]:
    """Cap the native ``user_api`` pools (``"openmp"`` or ``"blas"``) at ``n_threads``.

    Both limits are process-wide (verified for MSVC's vcomp: a cap set in one thread is
    seen by threads that already exist), so overlapping contexts from different
    threads must not restore each other's values. Under a lock, each API keeps the
    limits of all its open contexts: the cap in force is always the strictest open
    one (re-applied whenever a context enters or exits, in any order), and the
    original thread counts come back when the last context exits. Every set and
    restore goes through a controller narrowed to ``user_api``, so a BLAS context
    never touches OpenMP libraries and vice versa (threadpoolctl's
    ``restore_original_limits`` restores every library its controller holds).

    Raises:
        ValueError: If ``n_threads`` is below 1.
    """
    if isinstance(n_threads, bool) or not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError(f"n_threads must be an int >= 1, got {n_threads!r}")
    with _LIMIT_LOCK:
        state = _ACTIVE_LIMITS.get(user_api)
        narrowed = _controller(user_api).select(user_api=user_api)
        if state is None:
            state = _ACTIVE_LIMITS[user_api] = {
                "limits": [],
                "originals": narrowed.limit(limits=n_threads),
            }
        state["limits"].append(n_threads)
        try:
            narrowed.limit(limits=min(state["limits"]))
        except BaseException:
            # Roll back so a failed enter leaves no stale (stricter) cap behind.
            state["limits"].remove(n_threads)
            try:
                if state["limits"]:
                    narrowed.limit(limits=min(state["limits"]))
                else:
                    state["originals"].restore_original_limits()
            finally:
                if not state["limits"]:
                    del _ACTIVE_LIMITS[user_api]
            raise
    try:
        yield
    finally:
        with _LIMIT_LOCK:
            state["limits"].remove(n_threads)
            try:
                if state["limits"]:
                    narrowed.limit(limits=min(state["limits"]))
                else:
                    state["originals"].restore_original_limits()
            finally:
                if not state["limits"]:
                    del _ACTIVE_LIMITS[user_api]


def openmp_single_threaded() -> contextlib.AbstractContextManager:
    """Limit OpenMP runtimes (sklearn's neighbours/distances, boosters) to one thread."""
    return native_thread_limit(1, "openmp")


def openmp_single_threaded_call(func: _F) -> _F:
    """Decorator form of :func:`openmp_single_threaded`."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with openmp_single_threaded():
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
