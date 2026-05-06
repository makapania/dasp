import numpy as np
import optuna
import pytest
from optuna.trial import FixedTrial, TrialState


class TestFingerprintConstructionTest:
    def test_fingerprint_tuple_includes_resolved_fields(self):
        from spectral_predict.unified_bayesian import _build_fit_fingerprint

        fingerprint = _build_fit_fingerprint(
            preprocess_config={
                'name': 'snv_deriv1',
                'deriv': 1,
                'window': 17,
                'polyorder': 2,
                'apply_baseline': True,
                'apply_smoothing': True,
                'apply_autoscale': True,
            },
            subset_type='importance',
            subset_tag='top10_importance',
            n_vars=10,
            top_indices=np.array([3, 1, 2]),
            model_name='PLS-DA',
            task_type='classification',
            model_params={'n_components': 3, 'lr_C': 1.5},
            imbalance_method='class_weight',
            imbalance_params={'k': 2},
            use_sample_weight_for_classification=True,
            resolved_class_weight=(('lr__class_weight', 'balanced'),),
            tail_lr_random_state=42,
            early_stopping_rounds=40,
            use_early_stopping=True,
            baseline_method='als',
            baseline_params={'lam': 1000.0, 'p': 0.01},
            smoothing_window=9,
            smoothing_polyorder=2,
        )

        fp = dict(fingerprint)
        assert fp['imbalance_method'] == 'class_weight'
        assert fp['imbalance_params'] == (('k', 2),)
        assert fp['use_sample_weight_for_classification'] is True
        assert fp['resolved_class_weight'] == (('lr__class_weight', 'balanced'),)
        assert fp['tail_lr_random_state'] == 42
        assert fp['early_stopping_rounds'] == 40
        assert fp['top_indices'] == (3, 1, 2)
        assert fp['baseline_params'] == (('lam', 1000.0), ('p', 0.01))


class TestDeduplicationHitTest:
    def test_second_identical_fingerprint_gets_pruned(self):
        from spectral_predict.unified_bayesian import _register_or_prune_fingerprint

        study = optuna.create_study(direction='minimize')
        first = study.ask()
        second = study.ask()
        seen = {}
        fingerprint = (('model_name', 'PLS'), ('n_vars', 10))

        _register_or_prune_fingerprint(first, fingerprint, seen)

        with pytest.raises(optuna.TrialPruned):
            _register_or_prune_fingerprint(second, fingerprint, seen)

        assert seen[fingerprint] == first.number
        assert second.user_attrs['duplicate_of_trial'] == first.number


class TestResumeRehydrationTest:
    def test_seen_fingerprints_populated_from_completed_trials(self):
        from spectral_predict.unified_bayesian import _rehydrate_seen_fingerprints

        study = optuna.create_study(direction='minimize')
        complete = study.ask()
        fingerprint = (('model_name', 'PLS'), ('n_vars', 10))
        complete.set_user_attr('fingerprint', repr(fingerprint))
        study.tell(complete, 1.0)

        pruned = study.ask()
        pruned.set_user_attr('fingerprint', repr((('model_name', 'PLS'), ('n_vars', 20))))
        study.tell(pruned, state=TrialState.PRUNED)

        seen = {}
        added = _rehydrate_seen_fingerprints(study, seen)

        assert added == 1
        assert seen == {fingerprint: complete.number}


class TestNoDuplicateFitsTest:
    def test_forced_identical_trials_execute_cv_once(self, monkeypatch):
        from spectral_predict import unified_bayesian as ub

        rng = np.random.default_rng(42)
        X = rng.normal(size=(18, 8))
        y = X[:, 0] - 0.25 * X[:, 1]
        wavelengths = np.linspace(1000, 1800, X.shape[1])
        seen = {}
        cv_calls = {'count': 0}

        original_cv = ub.cross_val_predict_pooled

        def counting_cv(*args, **kwargs):
            cv_calls['count'] += 1
            return original_cv(*args, **kwargs)

        monkeypatch.setattr(ub, 'cross_val_predict_pooled', counting_cv)
        objective = ub.create_unified_objective(
            X, y, wavelengths,
            model_name='PLS',
            task_type='regression',
            cv_folds=3,
            random_state=42,
            seen_fingerprints=seen,
        )

        params = {
            'preprocessing': 'raw',
            'subset_type': 'importance',
            'n_vars': 'full',
            'region_id': 0,
            'n_components': 2,
        }

        assert np.isfinite(objective(FixedTrial(params)))
        with pytest.raises(optuna.TrialPruned):
            objective(FixedTrial(params))
        with pytest.raises(optuna.TrialPruned):
            objective(FixedTrial(params))

        assert cv_calls['count'] == 1
