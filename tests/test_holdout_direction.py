"""QW4: holdout direction for Kennard-Stone / SPXY / DUPLEX splits.

KS and SPXY pick the representative samples (extremes included). For a fixed
external holdout those samples are the CALIBRATION set and the rest validate;
picking the holdout with KS puts the extremes in validation and forces the model
to extrapolate. SPXY follows Galvão et al. 2005 (distance matrices divided by
their maxima). DUPLEX follows Snee 1977 (two interleaved max-min selections).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from spectral_predict.sample_selection import (
    duplex,
    kennard_stone,
    split_calibration_holdout,
    spxy,
    spxy_distance_matrix,
)

REPO = Path(__file__).resolve().parents[1]


def _line_with_extremes(n: int = 25, seed: int = 0):
    """Samples spread along one direction; rows 0 and 1 are the two ends."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([[-10.0, 10.0], rng.uniform(-8, 8, n - 2)])
    X = np.outer(t, np.linspace(0.5, 1.5, 40)) + rng.normal(0, 0.05, (n, 40))
    return X, t


@pytest.mark.parametrize("method", ["kennard-stone", "spxy"])
def test_calibration_holds_the_extremes_and_holdout_is_interior(method):
    X, t = _line_with_extremes()
    y = t.copy()
    cal, hold = split_calibration_holdout(X, 6, method=method, y=y)

    assert len(cal) == 19 and len(hold) == 6
    assert set(cal) | set(hold) == set(range(25)) and not set(cal) & set(hold)
    assert {0, 1} <= set(cal), "the extremes calibrate"
    # Every holdout sample lies inside the calibration range: no extrapolation.
    assert t[hold].min() > t[cal].min() and t[hold].max() < t[cal].max()
    assert y[hold].min() > y[cal].min() and y[hold].max() < y[cal].max()


def test_holdout_is_not_the_old_ks_pick():
    """Regression guard: the old code returned kennard_stone(X, n_holdout) as the holdout."""
    X, _ = _line_with_extremes()
    _, hold = split_calibration_holdout(X, 6, method="kennard-stone")
    old_holdout = set(kennard_stone(X, 6))
    assert {0, 1} <= old_holdout  # the old holdout held the extremes
    assert not {0, 1} & set(hold)
    assert set(hold) != old_holdout


def test_ks_calibration_is_the_ks_selection():
    X, _ = _line_with_extremes()
    cal, hold = split_calibration_holdout(X, 6, method="kennard-stone")
    np.testing.assert_array_equal(cal, np.sort(kennard_stone(X, 19)))
    np.testing.assert_array_equal(hold, np.setdiff1d(np.arange(25), cal))


def test_kennard_stone_order_is_unchanged():
    """CT standards and model_io representatives keep their KS order (R085 fixed)."""
    X = np.array([[0.0], [1.0], [2.0], [10.0], [6.0]])
    np.testing.assert_array_equal(kennard_stone(X, 4), [0, 3, 4, 2])


# --- SPXY: Galvão et al. 2005 normalisation ---------------------------------


def test_spxy_distance_divides_each_matrix_by_its_maximum():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 8)) * 50.0
    y = rng.normal(size=15) * 0.01
    dX = squareform(pdist(X))
    dY = np.abs(y[:, None] - y[None, :])
    expected = dX / dX.max() + dY / dY.max()
    np.testing.assert_allclose(spxy_distance_matrix(X, y), expected)


def test_spxy_does_not_rescale_spectral_columns():
    """A near-constant noise column must not get the weight of a real band.

    Column-wise [0, 1] scaling (the old GUI version) stretched a column that
    varies by 1e-6 to the same range as the informative column.
    """
    rng = np.random.default_rng(2)
    signal = rng.uniform(0, 10, 20)
    noise = rng.uniform(0, 1e-6, 20)
    X = np.column_stack([signal, noise])
    y = np.zeros(20)  # no y information: SPXY reduces to KS on X
    D = spxy_distance_matrix(X, y)
    d_signal = np.abs(signal[:, None] - signal[None, :])
    np.testing.assert_allclose(D, d_signal / d_signal.max(), atol=1e-6)
    np.testing.assert_array_equal(spxy(X, y, 8), kennard_stone(signal.reshape(-1, 1), 8))


def test_spxy_selection_ignores_units_of_x_and_y():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 12))
    y = rng.normal(size=30)
    base = spxy(X, y, 10)
    np.testing.assert_array_equal(spxy(X * 1000.0, y, 10), base)
    np.testing.assert_array_equal(spxy(X, y * 1e-3 + 5.0, 10), base)


def test_spxy_rejects_missing_y():
    X = np.random.default_rng(4).normal(size=(10, 3))
    y = np.arange(10, dtype=float)
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        spxy(X, y, 4)


# --- DUPLEX: Snee 1977 -------------------------------------------------------


def test_duplex_hand_worked_example():
    """1-D points 0..9, five each. Worked by hand:

    cal seed = (0, 9); val seed = farthest pair left = (1, 8); then cal takes 4
    (4 from {0, 9}), val takes 5 (3 from {1, 8}), cal takes 2, val takes 3, cal
    takes 6 and is full, and the last sample (7) goes to val.
    """
    X = np.arange(10, dtype=float).reshape(-1, 1)
    cal, val = duplex(X, n_cal=5)
    np.testing.assert_array_equal(cal, [0, 9, 4, 2, 6])
    np.testing.assert_array_equal(val, [1, 8, 5, 3, 7])

    # The old version alternated one KS order, which sent an extreme (9) to
    # validation; proper DUPLEX gives both ends to calibration.
    assert 9 not in val


def _replay_is_two_maxmin_selections(X, cal, val, n_cal, n_val):
    D = squareform(pdist(X))
    n = len(X)
    left = set(range(n))

    def farthest_pair(rows):
        rows = sorted(rows)
        sub = D[np.ix_(rows, rows)]
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        return {rows[a], rows[b]}

    assert set(cal[:2]) == farthest_pair(left)
    left -= set(cal[:2])
    assert set(val[:2]) == farthest_pair(left)
    left -= set(val[:2])
    ci, vi = 2, 2
    while left:
        if ci < n_cal:
            k = cal[ci]
            best = max(D[r, cal[:ci]].min() for r in left)
            assert D[k, cal[:ci]].min() == pytest.approx(best)
            left.remove(k)
            ci += 1
        if left and vi < n_val:
            k = val[vi]
            best = max(D[r, val[:vi]].min() for r in left)
            assert D[k, val[:vi]].min() == pytest.approx(best)
            left.remove(k)
            vi += 1
    assert ci == n_cal and vi == n_val


@pytest.mark.parametrize("n_cal", [30, 20, 15])
def test_duplex_alternates_two_independent_selections(n_cal):
    X = np.random.default_rng(5).normal(size=(40, 6))
    cal, val = duplex(X, n_cal=n_cal)
    assert len(cal) == n_cal and len(val) == 40 - n_cal
    assert set(cal) | set(val) == set(range(40)) and not set(cal) & set(val)
    _replay_is_two_maxmin_selections(X, list(cal), list(val), n_cal, 40 - n_cal)


def test_duplex_single_validation_sample():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    cal, val = duplex(X, n_cal=5)
    assert len(val) == 1 and len(cal) == 5
    assert {0, 5} <= set(cal)


def test_split_duplex_returns_the_duplex_calibration():
    X = np.random.default_rng(6).normal(size=(30, 4))
    cal, hold = split_calibration_holdout(X, 8, method="duplex")
    d_cal, d_val = duplex(X, n_cal=22)
    np.testing.assert_array_equal(cal, np.sort(d_cal))
    np.testing.assert_array_equal(hold, np.sort(d_val))


# --- split_calibration_holdout argument checks --------------------------------


@pytest.mark.parametrize("n_holdout", [0, 9, 10])
def test_split_rejects_impossible_sizes(n_holdout):
    X = np.random.default_rng(7).normal(size=(10, 3))
    with pytest.raises(ValueError, match="Cannot hold out"):
        split_calibration_holdout(X, n_holdout)


def test_split_rejects_unknown_method_and_spxy_without_y():
    X = np.random.default_rng(8).normal(size=(10, 3))
    with pytest.raises(ValueError, match="Unknown"):
        split_calibration_holdout(X, 3, method="kennard_stone")
    with pytest.raises(ValueError, match="needs y"):
        split_calibration_holdout(X, 3, method="spxy")


# --- Example data sanity (numbers documented in SESSION_LOG, not pinned) ------


@pytest.mark.skipif(
    not (REPO / "example" / "BoneCollagen.csv").exists(), reason="example data not present"
)
def test_example_data_holdout_direction_lowers_rmsep():
    """On the bundled bone-collagen set the corrected KS split stops extrapolating.

    Measured 2026-10-02 (PLS, 9 held out, LV chosen by 5-fold CV): old KS
    direction RMSEP 3.57 with 4 holdout samples outside the calibration y range,
    corrected 1.30 with none. This test checks only the direction of the effect.
    """
    from sklearn.cross_decomposition import PLSRegression

    from spectral_predict.io import align_xy, read_asd_dir, read_reference_csv

    X, _ = read_asd_dir(str(REPO / "example"))
    ref = read_reference_csv(str(REPO / "example" / "BoneCollagen.csv"), "File Number")
    X, y = align_xy(X, ref, "File Number", "%Collagen")
    Xa, ya = X.to_numpy(float), y.to_numpy(float)
    n_hold = 9

    def rmsep(hold):
        cal = np.setdiff1d(np.arange(len(ya)), hold)
        model = PLSRegression(8, scale=False).fit(Xa[cal], ya[cal])
        pred = model.predict(Xa[hold]).ravel()
        return float(np.sqrt(np.mean((pred - ya[hold]) ** 2))), cal

    old_hold = np.sort(kennard_stone(Xa, n_hold))
    _, new_hold = split_calibration_holdout(Xa, n_hold, method="kennard-stone")
    old_rmsep, _ = rmsep(old_hold)
    new_rmsep, cal = rmsep(new_hold)

    assert new_rmsep < old_rmsep
    assert ya[new_hold].min() >= ya[cal].min() and ya[new_hold].max() <= ya[cal].max()
