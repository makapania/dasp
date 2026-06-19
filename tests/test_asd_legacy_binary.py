"""Tests for the legacy float32 ASD-v1 binary reader (e.g. .sco files)."""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from spectral_predict.readers.asd_native import read_legacy_asd, read_binary_asd


def _make_legacy_asd(
    values, ch1=350.0, step=1.0, data_type=1, magic=b"ASD\x00", dtype="<f4"
):
    """Build an in-memory legacy ASD-v1 byte buffer for the given spectrum."""
    values = np.asarray(values, dtype="float64")
    channels = len(values)
    width = 8 if dtype == "<f8" else 4
    raw = bytearray(484 + channels * width)
    raw[0:4] = magic
    raw[186] = data_type
    struct.pack_into("<f", raw, 191, ch1)
    struct.pack_into("<f", raw, 195, step)
    struct.pack_into("<H", raw, 204, channels)
    raw[484:] = np.asarray(values, dtype=dtype).tobytes()
    return bytes(raw)


def test_decodes_float32_spectrum(tmp_path):
    vals = np.linspace(0.13, 0.68, 2151)
    p = tmp_path / "italy.000.sco"
    p.write_bytes(_make_legacy_asd(vals))

    s = read_legacy_asd(p)

    assert isinstance(s, pd.Series)
    assert len(s) == 2151
    assert not s.isna().any()
    assert s.index.min() == 350.0
    assert s.index.max() == 2500.0
    np.testing.assert_allclose(s.values, vals, rtol=1e-6)


def test_modern_float64_returns_none(tmp_path):
    # size == 484 + channels*8 -> caller should fall back to SpecDAL.
    p = tmp_path / "modern.asd"
    p.write_bytes(_make_legacy_asd(np.zeros(2151), dtype="<f8"))

    assert read_legacy_asd(p) is None


def test_non_asd_returns_none(tmp_path):
    p = tmp_path / "plain.asd"
    p.write_bytes(b"ASD Field Spec Pro\n350 0.1\n351 0.2\n")  # ASCII, not binary

    assert read_legacy_asd(p) is None


def test_truncated_legacy_raises(tmp_path):
    # Valid magic + header but the body is short -> corrupt, must not silently pass.
    raw = bytearray(_make_legacy_asd(np.zeros(2151)))
    p = tmp_path / "trunc.sco"
    p.write_bytes(bytes(raw[: 484 + 100 * 4]))  # claims 2151 channels, has 100

    with pytest.raises(ValueError, match="does not match header"):
        read_legacy_asd(p)


def test_implausible_channel_count_raises(tmp_path):
    raw = bytearray(_make_legacy_asd(np.zeros(10)))
    struct.pack_into("<H", raw, 204, 0)  # zero channels
    p = tmp_path / "zero.sco"
    p.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="implausible channel count"):
        read_legacy_asd(p)


def test_non_reflectance_data_type_still_decodes(tmp_path, capsys):
    vals = np.linspace(0.1, 0.5, 100)
    p = tmp_path / "raw.sco"
    p.write_bytes(_make_legacy_asd(vals, data_type=0))  # 0 = RAW

    s = read_legacy_asd(p)

    assert s is not None and len(s) == 100
    assert "dataType=0" in capsys.readouterr().out


def test_read_binary_asd_delegates(tmp_path):
    vals = np.linspace(0.2, 0.4, 512)
    p = tmp_path / "x.sco"
    p.write_bytes(_make_legacy_asd(vals, ch1=350.0, step=1.0))

    s = read_binary_asd(p)
    assert len(s) == 512
    np.testing.assert_allclose(s.values, vals, rtol=1e-6)


def test_read_asd_dir_end_to_end_sco(tmp_path):
    """A directory of .sco files loads through the full read_asd_dir path."""
    from spectral_predict.io import read_asd_dir

    for i in range(3):
        vals = np.linspace(0.13 + i * 0.01, 0.68, 2151)
        (tmp_path / f"sample.00{i}.sco").write_bytes(_make_legacy_asd(vals))

    df, meta = read_asd_dir(tmp_path)

    assert df.shape == (3, 2151)
    assert not np.isnan(df.values).any()
    assert set(df.index) == {"sample.000", "sample.001", "sample.002"}
    assert meta["data_type"] == "reflectance"
    assert df.columns.min() == 350.0 and df.columns.max() == 2500.0


def test_nan_payload_does_not_crash(tmp_path):
    """A corrupt float32 payload (NaN/inf) decodes without raising."""
    vals = np.linspace(0.1, 0.5, 100)
    vals[0] = np.nan
    vals[1] = np.inf
    p = tmp_path / "nan.sco"
    p.write_bytes(_make_legacy_asd(vals))

    s = read_legacy_asd(p)
    assert len(s) == 100
    assert np.isnan(s.iloc[0])
