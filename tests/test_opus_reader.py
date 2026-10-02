"""Tests for Bruker OPUS block selection (review finding R017).

The reader used to guard each fallback with ``if spectrum is None`` while never
assigning ``spectrum``, so every later block overwrote the earlier one and a
standard AB + ScSm + ScRf file returned the background reference (ScRf). The
io wrapper then merged the reader's metadata last, overwriting the normalised
data_type. No real OPUS file is in the repo, so ``brukeropus.read_opus`` is
replaced with fakes that mimic brukeropus 1.4.x: blocks are attributes named
by ``data_keys`` and unknown attributes resolve to None.
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from spectral_predict import io as sp_io
from spectral_predict.readers import opus_reader

N_POINTS = 60
X_DESC = np.linspace(4000.0, 3410.0, N_POINTS)  # OPUS stores wavenumbers descending

# Distinct, recognisable values per block
BLOCK_VALUES = {
    "a": 0.5,  # absorbance
    "t": 0.25,  # transmittance
    "r": 0.4,  # reflectance
    "logr": 0.45,  # log reflectance, already -log10(R)
    "km": 0.3,  # Kubelka-Munk
    "atr": 0.35,  # ATR-corrected absorbance
    "pas": 0.55,  # photoacoustic
    "ra": 500.0,  # Raman intensity
    "e": 600.0,  # emission
    "aria": 0.6,  # arithmetic result, absorbance-like
    "arit": 0.7,  # arithmetic result, transmittance-like
    "sm": 111.0,  # single-channel sample
    "rf": 999.0,  # single-channel background reference
}


class FakeData:
    def __init__(self, key: str, x=None, y=None):
        self.x = X_DESC.copy() if x is None else x
        if y is None:
            y = BLOCK_VALUES[key] + np.arange(N_POINTS) * 1e-3
        self.y = y
        self.label = f"label-{key}"


class FakeOpusFile:
    """Mimics brukeropus.OPUSFile: absent attributes resolve to None."""

    def __init__(self, blocks: dict[str, FakeData], is_opus: bool = True, with_keys: bool = True):
        self.is_opus = is_opus
        if with_keys:
            self.data_keys = list(blocks)
        for key, data in blocks.items():
            setattr(self, key, data)

    def __getattr__(self, name):
        return None


def _blocks(*keys: str) -> dict[str, FakeData]:
    return {k: FakeData(k) for k in keys}


@pytest.fixture
def fake_brukeropus(monkeypatch):
    """Install a fake brukeropus; returns a registry of path -> FakeOpusFile."""
    registry: dict[str, FakeOpusFile] = {}
    module = types.ModuleType("brukeropus")
    module.read_opus = lambda path: registry[str(Path(path))]
    monkeypatch.setitem(sys.modules, "brukeropus", module)
    return registry


def _make(tmp_path: Path, registry, name: str, opus_file: FakeOpusFile) -> Path:
    path = tmp_path / name
    path.write_bytes(b"placeholder")
    registry[str(path)] = opus_file
    return path


def _expected(key: str) -> np.ndarray:
    # Ascending x after the reader reverses the stored descending order
    return (BLOCK_VALUES[key] + np.arange(N_POINTS) * 1e-3)[::-1]


@pytest.mark.parametrize(
    "keys, chosen, data_type",
    [
        (("a",), "a", "absorbance"),
        (("a", "sm", "rf"), "a", "absorbance"),
        (("sm", "rf", "a"), "a", "absorbance"),  # data_keys order must not matter
        (("a", "t"), "a", "absorbance"),
        (("t", "a", "sm", "rf"), "a", "absorbance"),
        (("t", "sm", "rf"), "t", "transmittance"),
        (("r", "sm", "rf"), "r", "reflectance"),
    ],
)
def test_processed_block_wins(tmp_path, fake_brukeropus, keys, chosen, data_type):
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(_blocks(*keys)))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no single-channel warning expected
        spectrum, meta = opus_reader.read_opus_file(path)

    np.testing.assert_array_equal(spectrum.to_numpy(), _expected(chosen))
    np.testing.assert_array_equal(spectrum.index.to_numpy(), X_DESC[::-1])
    assert meta["data_type"] == data_type
    assert meta["opus_block"] == chosen
    assert meta["opus_block_label"] == f"label-{chosen}"


@pytest.mark.parametrize(
    "keys, chosen, data_type",
    [
        (("sm", "rf"), "sm", "sample"),
        (("rf", "sm"), "sm", "sample"),
        (("rf",), "rf", "reference"),
    ],
)
def test_single_channel_only_as_fallback_with_warning(
    tmp_path, fake_brukeropus, keys, chosen, data_type
):
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(_blocks(*keys)))

    with pytest.warns(UserWarning, match="single-channel"):
        spectrum, meta = opus_reader.read_opus_file(path)

    np.testing.assert_array_equal(spectrum.to_numpy(), _expected(chosen))
    assert meta["data_type"] == data_type
    assert meta["opus_block"] == chosen


def test_unusable_absorbance_falls_through(tmp_path, fake_brukeropus, capsys):
    blocks = _blocks("t", "sm")
    blocks["a"] = FakeData("a", y=np.ones(N_POINTS - 5))  # length mismatch
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(blocks))

    spectrum, meta = opus_reader.read_opus_file(path)

    assert meta["opus_block"] == "t"
    np.testing.assert_array_equal(spectrum.to_numpy(), _expected("t"))
    assert "skipped unusable OPUS blocks" in capsys.readouterr().out


def test_two_dimensional_series_block_is_skipped(tmp_path, fake_brukeropus):
    blocks = _blocks("t")
    blocks["a"] = FakeData("a", y=np.ones((3, N_POINTS)))  # a DataSeries-like block
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(blocks))

    _, meta = opus_reader.read_opus_file(path)

    assert meta["opus_block"] == "t"


def test_without_data_keys_attribute_uses_attribute_lookup(tmp_path, fake_brukeropus):
    opus = FakeOpusFile(_blocks("sm", "rf", "a"), with_keys=False)
    path = _make(tmp_path, fake_brukeropus, "s.0", opus)

    spectrum, meta = opus_reader.read_opus_file(path)

    assert meta["opus_block"] == "a"
    np.testing.assert_array_equal(spectrum.to_numpy(), _expected("a"))


def test_no_usable_block_raises(tmp_path, fake_brukeropus):
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile({}))
    with pytest.raises(ValueError, match="No spectral data"):
        opus_reader.read_opus_file(path)


def test_non_opus_file_raises_value_error(tmp_path, fake_brukeropus):
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(_blocks("a"), is_opus=False))
    with pytest.raises(ValueError, match="Not a valid Bruker OPUS file"):
        opus_reader.read_opus_file(path)


# ---------------------------------------------------------------------------
# io.read_opus_file wrapper: normalised keys must not be overwritten
# ---------------------------------------------------------------------------


def test_wrapper_absorbance_with_single_channels(tmp_path, fake_brukeropus):
    path = _make(tmp_path, fake_brukeropus, "s1.0", FakeOpusFile(_blocks("a", "sm", "rf")))

    df, meta = sp_io.read_opus_file(path)

    assert df.shape == (1, N_POINTS)
    assert df.index[0] == "s1"
    np.testing.assert_array_equal(df.iloc[0].to_numpy(), _expected("a"))
    assert meta["data_type"] == "absorbance"
    assert meta["source_data_type"] == "absorbance"
    assert meta["opus_block"] == "a"
    assert meta["file_format"] == "opus"
    assert meta["x_unit"] == "cm-1"


@pytest.mark.parametrize("key, source", [("t", "transmittance"), ("r", "reflectance")])
def test_wrapper_keeps_mapped_data_type(tmp_path, fake_brukeropus, key, source):
    path = _make(tmp_path, fake_brukeropus, "s1.0", FakeOpusFile(_blocks(key, "sm", "rf")))

    _, meta = sp_io.read_opus_file(path)

    # Before the fix '**file_metadata' came last and data_type became the raw string
    assert meta["data_type"] == "reflectance"
    assert meta["source_data_type"] == source
    assert meta["detection_method"] == f"opus_metadata({source})"


def test_wrapper_single_channel_is_not_labelled_absorbance_by_metadata(tmp_path, fake_brukeropus):
    path = _make(tmp_path, fake_brukeropus, "s1.0", FakeOpusFile(_blocks("sm", "rf")))

    with pytest.warns(UserWarning, match="single-channel"):
        _, meta = sp_io.read_opus_file(path)

    assert meta["source_data_type"] == "sample"
    # Raw intensities: neither absorbance nor reflectance, so no conversion applies
    assert meta["data_type"] == "other"
    assert meta["detection_method"] == "opus_metadata(sample)"


# ---------------------------------------------------------------------------
# Directory reader (the GUI's OPUS import path)
# ---------------------------------------------------------------------------


def test_read_opus_dir_returns_absorbance_for_every_file(tmp_path, fake_brukeropus):
    for i in range(3):
        _make(tmp_path, fake_brukeropus, f"sample{i}.0", FakeOpusFile(_blocks("a", "sm", "rf")))

    df, meta = sp_io.read_opus_dir(tmp_path)

    assert df.shape == (3, N_POINTS)
    for i in range(3):
        np.testing.assert_array_equal(df.loc[f"sample{i}"].to_numpy(), _expected("a"))
    assert meta["data_type"] == "absorbance"
    assert meta["dominant_data_type"] == "absorbance"
    assert meta["opus_blocks"] == {"a": 3}


def test_read_opus_dir_warns_on_mixed_blocks(tmp_path, fake_brukeropus):
    _make(tmp_path, fake_brukeropus, "s0.0", FakeOpusFile(_blocks("a", "sm", "rf")))
    _make(tmp_path, fake_brukeropus, "s1.0", FakeOpusFile(_blocks("a", "sm", "rf")))
    _make(tmp_path, fake_brukeropus, "s2.0", FakeOpusFile(_blocks("sm", "rf")))

    with pytest.warns(UserWarning, match="different data types"):
        _, meta = opus_reader.read_opus_dir(tmp_path)

    assert meta["data_types"] == {"absorbance": 2, "sample": 1}
    assert meta["opus_blocks"] == {"a": 2, "sm": 1}
    assert any("different data types" in m for m in meta["import_warnings"])
    assert any("1 of 3 OPUS files" in m for m in meta["import_warnings"])


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, source, data_type",
    [
        # Absorbance-equivalent: -log10(R) and ATR-corrected absorbance
        ("logr", "log_reflectance", "absorbance"),
        ("atr", "atr", "absorbance"),
        ("aria", "absorbance", "absorbance"),
        # Linear, but neither absorbance nor reflectance: no conversion may apply
        ("km", "kubelka_munk", "other"),
        ("pas", "photoacoustic", "other"),
        ("ra", "raman", "other"),
        ("e", "emission", "other"),
        ("sm", "sample", "other"),
    ],
)
def test_block_types_map_to_physical_pipeline_types(
    tmp_path, fake_brukeropus, key, source, data_type
):
    """Logging logr again, or 10**-x of KM/Raman data, would both be wrong."""
    path = _make(tmp_path, fake_brukeropus, "s1.0", FakeOpusFile(_blocks(key, "rf")))

    df, meta = sp_io.read_opus_file(path)

    np.testing.assert_array_equal(df.iloc[0].to_numpy(), _expected(key))
    assert meta["opus_block"] == key
    assert meta["source_data_type"] == source
    assert meta["data_type"] == data_type
    assert meta["detection_method"] == f"opus_metadata({source})"


@pytest.mark.parametrize(
    "keys, chosen, data_type",
    [
        (("aria", "sm", "rf"), "aria", "absorbance"),
        (("arit", "sm", "rf"), "arit", "transmittance"),
        (("a", "aria"), "a", "absorbance"),  # native AB outranks the arithmetic result
        (("t", "aria"), "t", "transmittance"),
        (("aria", "r"), "aria", "absorbance"),
    ],
)
def test_arithmetic_result_blocks(tmp_path, fake_brukeropus, keys, chosen, data_type):
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(_blocks(*keys)))

    spectrum, meta = opus_reader.read_opus_file(path)

    assert meta["opus_block"] == chosen
    assert meta["data_type"] == data_type
    np.testing.assert_array_equal(spectrum.to_numpy(), _expected(chosen))


@pytest.mark.parametrize(
    "bad_x",
    [
        np.full(N_POINTS, np.nan),  # FXV=NaN: linspace gives all NaN
        np.where(np.arange(N_POINTS) == 0, np.nan, X_DESC),  # one NaN coordinate
        np.full(N_POINTS, 4000.0),  # FXV == LXV: collapsed axis
        np.where(np.arange(N_POINTS) == 5, np.inf, X_DESC),
    ],
)
def test_bad_x_axis_falls_back_to_next_block(tmp_path, fake_brukeropus, capsys, bad_x):
    blocks = _blocks("t", "sm", "rf")
    blocks["a"] = FakeData("a", x=bad_x)
    path = _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(blocks))

    spectrum, meta = opus_reader.read_opus_file(path)

    assert meta["opus_block"] == "t"
    np.testing.assert_array_equal(spectrum.to_numpy(), _expected("t"))
    assert np.isfinite(spectrum.index.to_numpy()).all()
    assert "a: " in capsys.readouterr().out


def test_all_single_channel_folder_reports_import_warning(tmp_path, fake_brukeropus):
    for i in range(3):
        _make(tmp_path, fake_brukeropus, f"s{i}.0", FakeOpusFile(_blocks("sm", "rf")))

    with pytest.warns(UserWarning, match="3 of 3 OPUS files have no processed spectrum"):
        df, meta = sp_io.read_opus_dir(tmp_path)

    assert df.shape == (3, N_POINTS)
    assert meta["dominant_data_type"] == "sample"
    assert meta["source_data_type"] == "sample"
    # The GUI shows import_warnings in a dialog; this one must be there
    assert len(meta["import_warnings"]) == 1
    assert "single-channel" in meta["import_warnings"][0]


def test_absorbance_folder_has_no_import_warnings(tmp_path, fake_brukeropus):
    for i in range(2):
        _make(tmp_path, fake_brukeropus, f"s{i}.0", FakeOpusFile(_blocks("a", "sm", "rf")))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, meta = sp_io.read_opus_dir(tmp_path)

    assert meta["import_warnings"] == []


def test_overwritten_duplicate_stem_not_counted(tmp_path, fake_brukeropus):
    _make(tmp_path, fake_brukeropus, "s.0", FakeOpusFile(_blocks("sm", "rf")))
    _make(tmp_path, fake_brukeropus, "s.1", FakeOpusFile(_blocks("a")))

    _, meta = opus_reader.read_opus_dir(tmp_path)

    assert meta["data_types"] == {"absorbance": 1}
    assert meta["import_warnings"] == []


# ---------------------------------------------------------------------------
# Real brukeropus on bytes that are not an OPUS file
# ---------------------------------------------------------------------------


def test_real_brukeropus_rejects_garbage_with_value_error(tmp_path):
    pytest.importorskip("brukeropus")
    path = tmp_path / "garbage.0"
    path.write_bytes(b"OPUS" + b"\x00" * 100)

    with pytest.raises(ValueError):
        opus_reader.read_opus_file(path)
