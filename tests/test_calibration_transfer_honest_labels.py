"""QW2: calibration-transfer methods are named for what the code does.

Covers the display-name table (stored keys stay stable), the default method, the
standards selector used by per-wavelength slope/bias ('tsr'), and the docstrings that
used to claim CTAI / NS-PFCE need no paired standards.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_predict import calibration_transfer as ct
from spectral_predict.sample_selection import kennard_stone


class TestMethodNames:
    def test_default_method_is_slope_bias(self) -> None:
        assert ct.DEFAULT_METHOD == "tsr"

    @pytest.mark.parametrize(
        ("key", "full", "short"),
        [
            ("tsr", "Per-wavelength slope/bias standardization", "Slope/bias per wavelength"),
            ("ctai", "Paired regression in satellite PCA space (PC-DS)", "PC-DS"),
            ("nspfce", "Iterative ridge DS (dasp heuristic)", "Iterative ridge DS"),
            ("ns-pfce", "Iterative ridge DS (dasp heuristic)", "Iterative ridge DS"),
        ],
    )
    def test_display_names(self, key: str, full: str, short: str) -> None:
        assert ct.method_display_name(key) == full
        assert ct.method_display_name(key, short=True) == short

    def test_no_label_claims_published_ctai_or_pfce(self) -> None:
        for table in (ct.METHOD_DISPLAY_NAMES, ct.METHOD_SHORT_LABELS):
            for label in table.values():
                assert "CTAI" not in label
                assert "PFCE" not in label

    def test_unknown_key_falls_back_to_upper(self) -> None:
        assert ct.method_display_name("weird") == "WEIRD"

    def test_stored_keys_unchanged_for_saved_models(self) -> None:
        """Saved models store these keys; the dispatcher must still accept them."""
        assert set(ct.MethodType.__args__) == {"ds", "pds", "tsr", "ctai", "nspfce", "jypls-inv"}

    def test_dispatch_still_applies_legacy_keys(self) -> None:
        rng = np.random.default_rng(0)
        Xp = rng.standard_normal((20, 30))
        Xs = 0.9 * Xp + 0.1
        wl = np.arange(30.0)
        for key, params in (
            ("tsr", ct.estimate_tsr(Xp, Xs, np.arange(20))),
            ("ctai", ct.estimate_ctai(Xp, Xs, n_components=5)),
            ("nspfce", ct.estimate_nspfce(Xp, Xs, wl, max_iterations=5)),
        ):
            tm = ct.TransferModel("p", "s", key, wl, params)
            assert ct.apply_transfer_dispatch(Xs, tm).shape == Xs.shape


class TestDocstringsAreHonest:
    def test_ctai_docstring_requires_pairs(self) -> None:
        doc = ct.estimate_ctai.__doc__
        assert "Need not be the same samples" not in doc
        assert "No transfer samples required" not in doc
        assert "not** the published CTAI" in doc

    def test_nspfce_docstring_admits_heuristic(self) -> None:
        doc = ct.estimate_nspfce.__doc__
        assert "Need not be the same samples" not in doc
        assert "Literature reference needed" not in doc
        assert "dasp heuristic" in doc

    def test_tsr_citation_corrected(self) -> None:
        doc = ct.estimate_tsr.__doc__
        assert "31(2), 469-474" not in doc  # the sample-selection paper, wrongly cited
        assert "31(6), 1694-1696" in doc
        assert "trimmed scores regression" in doc

    def test_ctai_rejects_unpaired_rows(self) -> None:
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="paired"):
            ct.estimate_ctai(rng.standard_normal((20, 10)), rng.standard_normal((15, 10)))


class TestSelectTransferStandards:
    def _X(self, n: int = 30, p: int = 40) -> np.ndarray:
        return np.random.default_rng(2).standard_normal((n, p))

    def test_none_uses_all_pairs(self) -> None:
        np.testing.assert_array_equal(ct.select_transfer_standards(self._X(), None), np.arange(30))

    def test_equal_count_uses_all_pairs(self) -> None:
        np.testing.assert_array_equal(ct.select_transfer_standards(self._X(), 30), np.arange(30))

    def test_subset_is_kennard_stone_not_first_rows(self) -> None:
        X = self._X()
        idx = ct.select_transfer_standards(X, 8)
        np.testing.assert_array_equal(idx, kennard_stone(X, n_samples=8))
        assert set(idx.tolist()) != set(range(8))

    @pytest.mark.parametrize("n", [1, 31])
    def test_bad_counts_raise(self, n: int) -> None:
        with pytest.raises(ValueError):
            ct.select_transfer_standards(self._X(), n)


class TestSlopeBiasFitsOnSelectedStandards:
    def test_fit_uses_only_the_selected_rows(self) -> None:
        """Rows outside transfer_indices must not influence the slope/bias fit."""
        rng = np.random.default_rng(3)
        Xp = rng.standard_normal((25, 12))
        Xs = 0.9 * Xp + 0.2
        idx = ct.select_transfer_standards(Xp, 10)
        rest = np.setdiff1d(np.arange(25), idx)

        Xs_corrupt = Xs.copy()
        Xs_corrupt[rest] += 50.0  # garbage on the unused rows

        a = ct.estimate_tsr(Xp, Xs, idx)
        b = ct.estimate_tsr(Xp, Xs_corrupt, idx)
        np.testing.assert_allclose(a["slope"], b["slope"])
        np.testing.assert_allclose(a["bias"], b["bias"])
        np.testing.assert_allclose(a["slope"], 1 / 0.9)
        np.testing.assert_array_equal(a["transfer_indices"], idx)
