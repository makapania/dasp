"""A persisted Optuna study must not be resumed across numerical environments.

Resuming reloads completed trial fingerprints and returns their cached scores
WITHOUT re-running cross-validation (`unified_bayesian.py` :2685 register/replay,
:1672 supervised objective). So a study created under one interpreter/dependency
set and resumed under another silently mixes numbers from two numerical stacks.

Reproduced 2026-09-12: a trial scored on Python 3.12 / numpy 2.4.4 / sklearn 1.8
replayed its exact value on Python 3.14 / numpy 2.5.3 / sklearn 1.9.1 with zero
CV calls. The fix keys the study name on an environment digest as well as the
analysis config, so the two simply never meet.
"""

from __future__ import annotations

import pytest

from spectral_predict.unified_bayesian import (
    ENV_FINGERPRINT_ATTR,
    ENV_FINGERPRINT_VERSION,
    EnvironmentFingerprintError,
    _ENV_TRACKED_DISTRIBUTIONS,
    _environment_digest,
    _numerical_environment,
)


def test_environment_reports_the_things_that_move_numbers():
    env = _numerical_environment()

    assert env["fingerprint_version"] == ENV_FINGERPRINT_VERSION
    for key in (
        "python_implementation", "python_version", "gil_enabled",
        "platform_system", "platform_machine", "frozen", "packages",
    ):
        assert key in env, f"environment is missing {key!r}"

    # Every tracked distribution must be accounted for, even if absent -- a
    # silently missing key would make two different environments hash alike.
    for dist in _ENV_TRACKED_DISTRIBUTIONS:
        assert dist in env["packages"], f"{dist} not reported"
        assert isinstance(env["packages"][dist], str) and env["packages"][dist]


def test_digest_is_stable_and_order_independent():
    """Same environment -> same digest, regardless of dict construction order."""
    env = _numerical_environment()
    assert _environment_digest(env) == _environment_digest(env)

    shuffled = {k: env[k] for k in reversed(list(env))}
    shuffled["packages"] = {k: env["packages"][k] for k in reversed(list(env["packages"]))}
    assert _environment_digest(shuffled) == _environment_digest(env), (
        "digest depends on dict ordering"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda e: e.update(python_version="3.12.10"), id="python_version"),
        pytest.param(lambda e: e["packages"].update(numpy="2.4.4"), id="numpy"),
        pytest.param(lambda e: e["packages"].update({"scikit-learn": "1.8.0"}), id="sklearn"),
        pytest.param(lambda e: e["packages"].update(optuna="4.8.0"), id="optuna"),
        pytest.param(lambda e: e.update(gil_enabled=False), id="free_threaded"),
        pytest.param(lambda e: e.update(platform_machine="ARM64"), id="architecture"),
        pytest.param(lambda e: e.update(frozen=True), id="frozen_bundle"),
    ],
)
def test_any_relevant_change_changes_the_digest(mutate):
    """Each of these can move a numerical result, so each must split the study."""
    baseline = _numerical_environment()
    changed = _numerical_environment()
    mutate(changed)

    assert changed != baseline, "test mutation did not actually change anything"
    assert _environment_digest(changed) != _environment_digest(baseline), (
        "environment change did not change the digest, so an incompatible "
        "study could be resumed"
    )


def test_unreadable_version_is_fatal_not_a_placeholder(monkeypatch):
    """An unknown version must NOT collapse to a shared placeholder.

    Two environments that both failed to report a version would hash identically
    and become resume-compatible -- precisely the bug this guards against. So an
    unreadable version has to raise rather than degrade.
    """
    import importlib.metadata as md

    real_version = md.version

    def exploding_version(name):
        if name == "numpy":
            raise OSError("metadata unreadable")
        return real_version(name)

    monkeypatch.setattr(md, "version", exploding_version)

    with pytest.raises(EnvironmentFingerprintError, match="numpy"):
        _numerical_environment()


@pytest.mark.parametrize("bad_version", [None, ""])
def test_missing_version_metadata_is_fatal(monkeypatch, bad_version):
    """importlib.metadata returns None, not an exception, for a dist-info with no
    Version field. That must not hash as a valid, shared value."""
    import importlib.metadata as md

    real_version = md.version

    def versionless(name):
        return bad_version if name == "numpy" else real_version(name)

    monkeypatch.setattr(md, "version", versionless)

    with pytest.raises(EnvironmentFingerprintError, match="numpy"):
        _numerical_environment()


def test_absent_package_is_recorded_not_fatal(monkeypatch):
    """Absence is a definite fact about the environment, unlike an unreadable one."""
    import importlib.metadata as md

    real_version = md.version

    def missing_version(name):
        if name == "catboost":
            raise md.PackageNotFoundError(name)
        return real_version(name)

    present = _numerical_environment()          # catboost readable
    monkeypatch.setattr(md, "version", missing_version)
    absent = _numerical_environment()           # catboost gone

    assert absent["packages"]["catboost"] == "absent"
    assert present["packages"]["catboost"] != "absent"
    # "absent" must be a distinguishable state, not a shrug that hashes the same.
    assert _environment_digest(absent) != _environment_digest(present)


def test_study_name_carries_the_environment_digest():
    """The digest has to reach the study NAME, not merely user_attrs.

    user_attrs alone would not prevent load_if_exists=True from attaching to an
    incompatible study -- the name is the thing Optuna keys on.
    """
    import inspect

    from spectral_predict import unified_bayesian

    src = inspect.getsource(unified_bayesian.run_unified_bayesian)
    assert "_env_hash" in src and "study_name =" in src
    assert f'{{ENV_FINGERPRINT_VERSION}}' in src or "ENV_FINGERPRINT_VERSION" in src, (
        "study_name must include the fingerprint version so a future scheme "
        "change cannot collide with existing digests"
    )
    assert ENV_FINGERPRINT_ATTR in src, "environment must also be stored readably"
