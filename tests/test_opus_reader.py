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
    assert meta["data_type"] in ("absorbance", "reflectance")  # heuristic, not metadata
    assert not str(meta["detection_method"]).startswith("opus_metadata")


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


# ---------------------------------------------------------------------------
# Real brukeropus on bytes that are not an OPUS file
# ---------------------------------------------------------------------------


def test_real_brukeropus_rejects_garbage_with_value_error(tmp_path):
    pytest.importorskip("brukeropus")
    path = tmp_path / "garbage.0"
    path.write_bytes(b"OPUS" + b"\x00" * 100)

    with pytest.raises(ValueError):
        opus_reader.read_opus_file(path)
