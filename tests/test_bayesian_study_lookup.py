"""Study-name discovery must preserve resume data and retirement notices."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
import pytest

from spectral_predict import run_state
from spectral_predict.unified_bayesian import run_unified_bayesian


@pytest.fixture
def search_options():
    rng = np.random.RandomState(11)
    X = rng.normal(size=(36, 120))
    return dict(
        X=X,
        y=X[:, :5].sum(axis=1) + rng.normal(size=36) * 0.1,
        wavelengths=np.linspace(1000.0, 2500.0, 120),
        model_name="PLS",
        n_trials=3,
        cv_folds=3,
        n_top_regions=2,
        enable_sqlite_persistence="always",
        verbose=False,
    )


def database_dump(path):
    with closing(sqlite3.connect(path)) as connection:
        return "\n".join(connection.iterdump())


def test_name_lookup_preserves_resume_data(tmp_path, monkeypatch, search_options):
    path = tmp_path / "resume.sqlite3"
    url = f"sqlite:///{path.as_posix()}"
    monkeypatch.setattr(run_state, "get_storage_url", lambda: url)
    initial_frame, initial = run_unified_bayesian(**search_options)
    assert not initial_frame.empty
    assert any(t.user_attrs.get("fingerprint") for t in initial.trials)
    initial.set_user_attr("preserve_arrays", {"indices": [1, 4], "values": [[0.2, 0.7]]})

    unrelated = optuna.create_study(study_name="unrelated", storage=url)
    unrelated.add_trial(
        optuna.trial.create_trial(
            value=1.25,
            params={"x": 2},
            distributions={"x": optuna.distributions.IntDistribution(1, 10)},
            user_attrs={"importances": [0.1, 0.8], "predictions": [0.3, 0.5]},
            system_attrs={"constraints": [0.0, -1.0]},
            intermediate_values={0: 2.5},
        )
    )
    snapshot = database_dump(path)
    prior_trials = initial.trials
    prior_attrs = initial.user_attrs
    names = Mock(wraps=optuna.study.get_all_study_names)
    summaries = Mock(wraps=optuna.get_all_study_summaries)
    monkeypatch.setattr(optuna.study, "get_all_study_names", names)
    monkeypatch.setattr(optuna, "get_all_study_summaries", summaries)

    resumed_frame, resumed = run_unified_bayesian(**search_options)

    names.assert_called_once_with(storage=url)
    summaries.assert_not_called()
    assert database_dump(path) == snapshot
    assert resumed.study_name == initial.study_name
    assert resumed.trials == prior_trials
    assert resumed.user_attrs == prior_attrs
    pd.testing.assert_frame_equal(resumed_frame, initial_frame, check_exact=True)

    # The discovery optimization must not change trial-budget accounting.
    _, continued = run_unified_bayesian(**dict(search_options, n_trials=4))
    assert len(continued.trials) == 4
    assert continued.trials[:3] == prior_trials
    assert continued.user_attrs == prior_attrs


@pytest.mark.parametrize(
    "prior_kinds",
    [("legacy",), ("different-environment",), ("legacy", "different-environment")],
    ids=["legacy", "different-environment", "both"],
)
def test_incompatible_study_notice_preserves_old_results(
    tmp_path, monkeypatch, search_options, prior_kinds
):
    path = tmp_path / "incompatible.sqlite3"
    url = f"sqlite:///{path.as_posix()}"
    monkeypatch.setattr(run_state, "get_storage_url", lambda: url)
    _, template = run_unified_bayesian(
        **dict(search_options, n_trials=0, enable_sqlite_persistence="never")
    )
    assert not path.exists()
    study_base = template.study_name.rsplit("_env1_", 1)[0]
    names = {"legacy": study_base, "different-environment": f"{study_base}_env1_other"}
    flags = {"legacy": "legacy_study_format", "different-environment": "environment_changed"}
    preserved_state = {}
    for kind in prior_kinds:
        prior = optuna.create_study(study_name=names[kind], storage=url)
        prior.set_user_attr("preserve_metadata", {"wavelengths": [1000.0, 1001.0]})
        prior.add_trial(
            optuna.trial.create_trial(value=12.5, user_attrs={"preserve_indices": [1, 3, 5]})
        )
        preserved_state[kind] = (prior.trials, prior.user_attrs)
    notes = []

    frame, current = run_unified_bayesian(
        **dict(search_options, n_trials=1, progress_callback=notes.append)
    )

    for kind, flag in flags.items():
        notices = [note for note in notes if note.get(flag)]
        if kind not in prior_kinds:
            assert notices == []
            continue
        assert len(notices) == 1
        assert names[kind] in notices[0]["message"]
        old_trials, old_attrs = preserved_state[kind]
        preserved = optuna.load_study(study_name=names[kind], storage=url)
        assert preserved.trials == old_trials
        assert preserved.user_attrs == old_attrs
    legacy_notices = [note for note in notes if note.get("legacy_study_format")]
    assert all("different numerical environment" not in n["message"] for n in legacy_notices)
    assert current.study_name == template.study_name
    assert current.study_name not in names.values()
    assert len(current.trials) == 1
    assert len(frame) == 1
