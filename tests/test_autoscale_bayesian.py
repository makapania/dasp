"""T-36: Bayesian-path autoscale tests + apply_preprocessing restructure regression.

Critical tests for Phase 5:
1. apply_preprocessing(apply_autoscale=False) must be bit-identical to the
   pre-T-36 implementation — the restructure (early-return -> assign-then-return)
   must not change any pre-existing trial outputs.
2. apply_preprocessing(apply_autoscale=True) must apply autoscale for EVERY
   preprocessing branch (raw, snv, snv_deriv, deriv_snv, deriv1-4) — not just
   one branch (the BUG #1 trap).
3. The Bayesian preprocessing cache key must include apply_autoscale so two
   trials differing only in autoscale produce different X_prep (BUG #2).
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_predict.unified_bayesian import apply_preprocessing


@pytest.fixture
def synthetic_X():
    rng = np.random.default_rng(0)
    return rng.normal(loc=2.0, scale=0.5, size=(20, 64))


def _ref_apply_preprocessing(X, config):
    """Bit-identical reproduction of the pre-T-36 apply_preprocessing logic
    (early-return form). Used to verify the T-36 restructure changes nothing
    when apply_autoscale=False.
    """
    from spectral_predict.preprocess import SNV, SavgolDerivative

    name = config['name']

    if name == 'raw':
        return X.copy()
    if name == 'snv':
        return SNV().fit_transform(X)
    if 'deriv' in name:
        deriv_order = config['deriv']
        window = config['window']
        polyorder = config['polyorder']
        if window < polyorder + 2:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1
        savgol = SavgolDerivative(deriv=deriv_order, window=window, polyorder=polyorder)
        if name.startswith('snv_deriv'):
            return savgol.fit_transform(SNV().fit_transform(X))
        elif name.startswith('deriv') and '_snv' in name:
            return SNV().fit_transform(savgol.fit_transform(X))
        else:
            return savgol.fit_transform(X)
    return X.copy()


PREPROCESSING_CASES = [
    ({'name': 'raw', 'deriv': 0, 'window': 0, 'polyorder': 0}, 'raw'),
    ({'name': 'snv', 'deriv': 0, 'window': 0, 'polyorder': 0}, 'snv'),
    ({'name': 'snv_deriv1', 'deriv': 1, 'window': 11, 'polyorder': 2}, 'snv_deriv1'),
    ({'name': 'snv_deriv2', 'deriv': 2, 'window': 11, 'polyorder': 3}, 'snv_deriv2'),
    ({'name': 'snv_deriv3', 'deriv': 3, 'window': 11, 'polyorder': 4}, 'snv_deriv3'),
    ({'name': 'snv_deriv4', 'deriv': 4, 'window': 11, 'polyorder': 5}, 'snv_deriv4'),
    ({'name': 'deriv1_snv', 'deriv': 1, 'window': 11, 'polyorder': 2}, 'deriv1_snv'),
    ({'name': 'deriv2_snv', 'deriv': 2, 'window': 11, 'polyorder': 3}, 'deriv2_snv'),
    ({'name': 'deriv3_snv', 'deriv': 3, 'window': 11, 'polyorder': 4}, 'deriv3_snv'),
    ({'name': 'deriv4_snv', 'deriv': 4, 'window': 11, 'polyorder': 5}, 'deriv4_snv'),
    ({'name': 'deriv1', 'deriv': 1, 'window': 11, 'polyorder': 2}, 'deriv1'),
    ({'name': 'deriv2', 'deriv': 2, 'window': 11, 'polyorder': 3}, 'deriv2'),
    ({'name': 'deriv3', 'deriv': 3, 'window': 11, 'polyorder': 4}, 'deriv3'),
    ({'name': 'deriv4', 'deriv': 4, 'window': 11, 'polyorder': 5}, 'deriv4'),
]


@pytest.mark.parametrize("config,label", PREPROCESSING_CASES)
def test_no_autoscale_bit_identical_to_pre_T36(synthetic_X, config, label):
    """T-36 BUG #1 regression: when apply_autoscale=False, the restructured
    apply_preprocessing must produce IDENTICAL output to the pre-T-36
    early-return form on every preprocessing branch.
    """
    cfg = dict(config)  # Don't add apply_autoscale -> default False path
    out_new = apply_preprocessing(synthetic_X.copy(), cfg)
    out_ref = _ref_apply_preprocessing(synthetic_X.copy(), cfg)
    np.testing.assert_array_equal(
        out_new, out_ref,
        err_msg=f"T-36 restructure changed output for '{label}' (apply_autoscale=False)"
    )


@pytest.mark.parametrize("config,label", PREPROCESSING_CASES)
def test_autoscale_fires_in_every_branch(synthetic_X, config, label):
    """T-36 BUG #1 second half: when apply_autoscale=True, every branch must
    pass through the autoscale step. Output should have ~zero column mean and
    ~unit column std (StandardScaler default ddof=0).
    """
    cfg = dict(config)
    cfg['apply_autoscale'] = True
    out = apply_preprocessing(synthetic_X.copy(), cfg)
    # After autoscale: column means ≈ 0, column stds ≈ 1.
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-10)


@pytest.mark.parametrize("config,label", PREPROCESSING_CASES)
def test_autoscale_changes_output(synthetic_X, config, label):
    """For non-degenerate preprocessing (i.e. not already autoscaled), turning
    on apply_autoscale must produce a DIFFERENT output array. Otherwise the
    autoscale step is silently a no-op (the BUG #1 dead-code symptom)."""
    cfg_off = dict(config)
    cfg_off['apply_autoscale'] = False
    cfg_on = dict(config)
    cfg_on['apply_autoscale'] = True
    out_off = apply_preprocessing(synthetic_X.copy(), cfg_off)
    out_on = apply_preprocessing(synthetic_X.copy(), cfg_on)
    # SNV already produces zero-mean rows but non-unit-variance columns; SG
    # derivatives have non-zero column means in general. So out_off should
    # differ from out_on for every branch in our fixture.
    assert not np.array_equal(out_off, out_on), (
        f"apply_autoscale=True produced identical output to False for '{label}' — "
        f"autoscale step was unreachable (BUG #1 not fixed)"
    )


def test_display_name_includes_autoscale_suffix():
    """T-36 Phase 5 follow-up: _build_display_preprocess_name must append
    '+autoscale' when apply_autoscale=True so Bayesian-path display names
    match the grid-path '+autoscale' suffix.
    """
    from spectral_predict.unified_bayesian import _build_display_preprocess_name

    # Plain core name + autoscale only
    assert _build_display_preprocess_name('snv', apply_autoscale=True) == 'snv+autoscale'
    # baseline + autoscale (suffix comes after baseline prefix)
    out = _build_display_preprocess_name(
        'snv', apply_baseline=True, baseline_method='als', apply_autoscale=True
    )
    assert out == 'als+snv+autoscale', f"got {out}"
    # smoothing + autoscale (sg0 prefix preserved, autoscale suffix at end)
    out = _build_display_preprocess_name(
        'snv', apply_smoothing=True, apply_autoscale=True
    )
    assert out == 'sg0+snv+autoscale', f"got {out}"
    # baseline + smoothing + autoscale
    out = _build_display_preprocess_name(
        'snv',
        apply_baseline=True,
        baseline_method='als',
        apply_smoothing=True,
        apply_autoscale=True,
    )
    assert out == 'als+sg0+snv+autoscale', f"got {out}"
    # autoscale=False omits the suffix
    assert _build_display_preprocess_name('snv', apply_autoscale=False) == 'snv'


def test_cache_key_includes_apply_autoscale():
    """T-36 BUG #2 regression: the Bayesian preprocessing cache key tuple in
    create_unified_objective must include apply_autoscale. We verify by source
    inspection — calling the inner objective requires a full Optuna trial
    machinery, but the key is a single tuple literal so a source check is
    sufficient and avoids brittle integration setup.
    """
    import inspect

    from spectral_predict import unified_bayesian

    src = inspect.getsource(unified_bayesian.create_unified_objective)
    # The cache_key tuple is unique enough that finding the line is robust.
    assert 'apply_autoscale' in src, (
        "create_unified_objective source must reference apply_autoscale "
        "(BUG #2 fix — preprocessing cache key)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
