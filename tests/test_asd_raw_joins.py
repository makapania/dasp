"""Detector joins from the raw readings (DN) stored in as7 / as8 ASD files."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from spectral_predict import spectral_diagnostics as sd
from spectral_predict.io import read_asd_dir
from spectral_predict.readers.asd_native import read_asd_white_reference

N = 2151
WL = 350.0 + np.arange(N)


def _white(joins=(1000.0, 1800.0), step=0.12, seed=0, smooth_only=False):
    """A smooth lamp-like white reference with a level step after each join."""
    rng = np.random.default_rng(seed)
    base = 20000.0 * np.exp(-0.5 * ((WL - 900.0) / 700.0) ** 2) + 3000.0
    base = base * (1.0 + 0.0005 * rng.standard_normal(N))
    if not smooth_only:
        for j in joins:
            base = np.where(WL > j, base * (1.0 + step), base)
    return base


def _raw_asd(
    path: Path,
    white: np.ndarray,
    *,
    magic: bytes = b"as8",
    header_joins=(0.0, 0.0),
    sample_scale: float = 0.5,
    description: bytes = b"white ref",
    data_type: int = 0,
    ref_flag: bool = True,
) -> Path:
    hdr = bytearray(484)
    hdr[:3] = magic
    hdr[3:19] = b"Trek field scan "  # comment text follows the magic; no NUL after it
    hdr[186] = data_type
    hdr[199] = 2
    struct.pack_into("<f", hdr, 191, 350.0)
    struct.pack_into("<f", hdr, 195, 1.0)
    struct.pack_into("<H", hdr, 204, N)
    struct.pack_into("<f", hdr, 444, header_joins[0])
    struct.pack_into("<f", hdr, 448, header_joins[1])
    sample = (white * sample_scale).astype("<f8").tobytes()
    if magic == b"as7":
        ref_hdr = bytearray(20)
        ref_hdr[0] = 1 if ref_flag else 0
        struct.pack_into("<H", ref_hdr, 17, len(description))
    else:
        ref_hdr = bytearray(20)
        struct.pack_into("<H", ref_hdr, 0, 0xFFFF if ref_flag else 0)
        struct.pack_into("<H", ref_hdr, 18, len(description))
    path.write_bytes(
        bytes(hdr) + sample + bytes(ref_hdr) + description + white.astype("<f8").tobytes()
    )
    return path


@pytest.mark.parametrize("magic", [b"as7", b"as8"])
def test_white_reference_is_read_back(tmp_path, magic):
    w = _white()
    f = _raw_asd(tmp_path / "a.asd", w, magic=magic, description=b"x" * 13)
    wl, white = read_asd_white_reference(f)
    np.testing.assert_array_equal(wl, WL)
    np.testing.assert_array_equal(white, w)


def test_files_without_raw_readings_give_none(tmp_path):
    w = _white()
    assert read_asd_white_reference(_raw_asd(tmp_path / "a.asd", w, data_type=1)) is None
    assert read_asd_white_reference(_raw_asd(tmp_path / "b.asd", w, ref_flag=False)) is None
    assert read_asd_white_reference(_raw_asd(tmp_path / "c.asd", w, magic=b"as6")) is None
    truncated = _raw_asd(tmp_path / "d.asd", w).read_bytes()[:-100]
    (tmp_path / "d.asd").write_bytes(truncated)
    assert read_asd_white_reference(tmp_path / "d.asd") is None
    assert read_asd_white_reference(tmp_path / "missing.asd") is None
    (tmp_path / "t.asd").write_text("ASD text export\n" * 50)
    assert read_asd_white_reference(tmp_path / "t.asd") is None


def test_detect_joins_from_white_reference_finds_the_steps():
    assert sd.detect_joins_from_white_reference(WL, _white((1000.0, 1785.0))) == (1000.0, 1785.0)
    # a join anywhere in the window, not only at the nominal wavelengths
    assert sd.detect_joins_from_white_reference(WL, _white((972.0, 1733.0))) == (972.0, 1733.0)


def test_detect_joins_needs_a_clear_jump_in_every_window():
    assert sd.detect_joins_from_white_reference(WL, _white(smooth_only=True)) is None
    one = _white((1000.0, 1800.0))
    one[WL > 1800] /= 1.12  # undo the second step
    assert sd.detect_joins_from_white_reference(WL, one) is None
    small = _white((1000.0, 1800.0), step=0.004)  # below the absolute floor
    assert sd.detect_joins_from_white_reference(WL, small) is None


def test_detect_joins_handles_dead_channels_and_short_grids():
    w = _white()
    w[WL == 1200] = 0.0  # zero DN: infinite relative jump around it must not win
    w[WL == 1201] = np.nan
    assert sd.detect_joins_from_white_reference(WL, w) == (1000.0, 1800.0)
    assert sd.detect_joins_from_white_reference(WL[:300], w[:300]) is None  # misses a window
    assert sd.detect_joins_from_white_reference(WL, w[:-1]) is None  # shape mismatch


def test_read_asd_dn_joins_on_a_file(tmp_path):
    f = _raw_asd(tmp_path / "t.asd", _white((1000.0, 1785.0)), magic=b"as7")
    assert sd.read_asd_dn_joins(f) == (1000.0, 1785.0)
    assert sd.read_asd_dn_joins(_raw_asd(tmp_path / "s.asd", _white(smooth_only=True))) is None


def test_resolve_joins_order_user_header_dn_detected():
    wl = WL
    dn = (1000.0, 1785.0)
    head = (1000.0, 1800.0)
    assert (
        sd.resolve_joins(wl, user_joins=(990.0,), header_joins=head, dn_joins=dn).source == "user"
    )
    res = sd.resolve_joins(wl, header_joins=head, dn_joins=dn)
    assert (res.source, res.joins) == ("header", head)
    assert res.warnings and "raw readings" in res.warnings[0]
    res = sd.resolve_joins(wl, header_joins=(0.0, 0.0), dn_joins=dn)
    assert (res.source, res.joins) == ("dn", dn)
    res = sd.resolve_joins(wl, header_joins=head, dn_joins=(1000.5, 1800.0))
    assert res.source == "header" and not res.warnings  # agrees within the tolerance
    # a DN join outside the grid's usable range is ignored, not trusted
    assert sd.resolve_joins(wl, dn_joins=(100.0, 200.0)).source == "none"


def test_union_joins():
    assert sd.union_joins((1000.0, 1800.0), (1000.0, 1785.0)) == (1000.0, 1785.0, 1800.0)
    assert sd.union_joins((1000.0, 1800.0), (1002.0, 1802.0)) == (1000.0, 1800.0)
    assert sd.union_joins(None, (), (1500.0,)) == (1500.0,)
    assert sd.union_joins() == ()


def _folder(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    for k, (white, kw) in enumerate(files):
        _raw_asd(d / f"s{k:02d}.asd", white, **kw)
    return d


@pytest.fixture
def specdal_reads_synthetic_files(tmp_path):
    """read_asd_dir decodes modern files through SpecDAL; skip if it cannot read ours."""
    d = _folder(tmp_path, "probe", [(_white(), {})])
    try:
        read_asd_dir(str(d))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SpecDAL cannot read the synthetic raw file here: {exc}")


def test_read_asd_dir_reports_joins_from_dn_when_the_header_has_none(
    tmp_path, specdal_reads_synthetic_files
):
    d = _folder(
        tmp_path,
        "trek",
        [(_white((1000.0, 1785.0), seed=k), {"magic": b"as7"}) for k in range(3)],
    )
    _, meta = read_asd_dir(str(d))
    assert meta["detector_joins"] == (1000.0, 1785.0)
    assert meta["detector_joins_source"] == "dn"


def test_read_asd_dir_prefers_a_plausible_header(tmp_path, specdal_reads_synthetic_files):
    d = _folder(
        tmp_path,
        "lab",
        [(_white((1000.0, 1785.0), seed=k), {"header_joins": (1000.0, 1800.0)}) for k in range(3)],
    )
    _, meta = read_asd_dir(str(d))
    assert meta["detector_joins"] == (1000.0, 1800.0)
    assert meta["detector_joins_source"] == "header"


def test_read_asd_dir_ignores_an_implausible_header(tmp_path, specdal_reads_synthetic_files):
    """Header joins outside the ASD join windows are not believed; the DN decide."""
    d = _folder(
        tmp_path,
        "odd",
        [(_white((1000.0, 1785.0), seed=k), {"header_joins": (1200.0, 1500.0)}) for k in range(2)],
    )
    _, meta = read_asd_dir(str(d))
    assert (meta["detector_joins"], meta["detector_joins_source"]) == ((1000.0, 1785.0), "dn")


def test_read_asd_dir_records_nothing_when_files_disagree(tmp_path, specdal_reads_synthetic_files):
    d = _folder(
        tmp_path,
        "mixed",
        [
            (_white((1000.0, 1785.0)), {}),
            (_white((1000.0, 1830.0), seed=1), {}),
        ],
    )
    _, meta = read_asd_dir(str(d))
    assert meta["detector_joins"] is None and meta["detector_joins_source"] is None


def test_read_asd_dir_no_clear_jump_means_no_joins(tmp_path, specdal_reads_synthetic_files):
    d = _folder(tmp_path, "flat", [(_white(smooth_only=True, seed=k), {}) for k in range(2)])
    _, meta = read_asd_dir(str(d))
    assert meta["detector_joins"] is None


def test_example_files_keep_their_header_joins():
    _, meta = read_asd_dir(str(Path(__file__).resolve().parents[1] / "example"))
    assert meta["detector_joins"] == (1000.0, 1800.0)
    assert meta["detector_joins_source"] == "header"


@pytest.mark.parametrize("dead", [0.0, np.nan, -5.0])
def test_a_dead_white_reference_channel_is_not_a_join(dead):
    white = _white()
    white[WL == 980.0] = dead
    assert sd.detect_joins_from_white_reference(WL, white) == (1000.0, 1800.0)
