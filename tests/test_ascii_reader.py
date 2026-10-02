"""Tests for the consolidated ASCII spectrum reader (review finding R062).

io.py used to define read_ascii_spectra twice; the later, effective definition
called pd.read_csv with header=0 and no directory handling, so folder imports
raised and a headerless file lost its first point. These tests pin the single
implementation: folders, headed and headerless files, exact endpoints and values.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_predict import io as sp_io
from spectral_predict.io import read_ascii_spectra, read_spectra

WAVENUMBERS = np.arange(4000.0, 3800.0, -1.0)  # 200 points, descending like Bruker .dpt


def _values(seed: int, n: int = len(WAVENUMBERS)) -> np.ndarray:
    return np.round(np.random.default_rng(seed).uniform(0.1, 1.5, n), 6)


def _write_xy(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    delimiter: str = ",",
    header: list[str] | None = None,
    encoding: str = "utf-8",
) -> Path:
    lines = list(header or [])
    lines += [f"{xi:.6f}{delimiter}{yi:.6f}" for xi, yi in zip(x, y)]
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


def test_only_one_read_ascii_spectra_definition():
    """The duplicate definition is gone: the directory-aware reader is the effective one."""
    source = Path(sp_io.__file__).read_text(encoding="utf-8")
    assert source.count("def read_ascii_spectra(") == 1


@pytest.mark.parametrize("delimiter", [",", "\t", " ", ";"])
def test_headerless_file_keeps_first_and_last_point(tmp_path, delimiter):
    y = _values(1)
    path = _write_xy(tmp_path / "s1.dpt", WAVENUMBERS, y, delimiter=delimiter)

    df, meta = read_ascii_spectra(path)

    assert df.shape == (1, 200)
    # Columns ascend; the file's first row (4000) is the last column
    assert df.columns[-1] == 4000.0
    assert df.columns[0] == 3801.0
    assert df.loc["s1", 4000.0] == y[0]
    assert df.loc["s1", 3801.0] == y[-1]
    np.testing.assert_array_equal(df.iloc[0].to_numpy(), y[::-1])
    assert meta["header_lines"] == []
    assert meta["n_points"] == 200


def test_headed_file_drops_only_the_heading(tmp_path):
    y = _values(2)
    path = _write_xy(
        tmp_path / "headed.txt",
        WAVENUMBERS,
        y,
        delimiter="\t",
        header=["Wavenumber (cm-1)\tAbsorbance"],
    )

    df, meta = read_ascii_spectra(path)

    assert df.shape == (1, 200)
    assert df.loc["headed", 4000.0] == y[0]
    assert meta["column_names"] == ["Wavenumber (cm-1)", "Absorbance"]
    assert meta["x_unit"] == "cm-1"
    assert meta["x_unit_detection_method"] == "ascii_header"


def test_multi_line_instrument_header(tmp_path):
    y = _values(3)
    header = ["Instrument: Example FTIR", "Operator: test", "Date 2026-10-02", "X Y"]
    path = _write_xy(tmp_path / "inst.dat", WAVENUMBERS, y, delimiter=" ", header=header)

    df, meta = read_ascii_spectra(path)

    assert df.shape == (1, 200)
    assert meta["header_lines"] == header
    assert meta["column_names"] == ["X", "Y"]


def test_header_that_splits_differently_from_data(tmp_path):
    """A heading with more spaces than tabs must not choose the space delimiter.

    The old folder parser chose the delimiter from the first line, so this file
    split on ' ' and no data row parsed.
    """
    y = _values(4)
    heading = "Wavelength (nm)\tReflectance (%)"
    path = _write_xy(tmp_path / "spaced.dpt", WAVENUMBERS, y, delimiter="\t", header=[heading])

    df, meta = read_ascii_spectra(path)

    assert df.shape == (1, 200)
    assert df.loc["spaced", 4000.0] == y[0]
    assert meta["delimiter"] == "\t"
    assert meta["column_names"] == ["Wavelength (nm)", "Reflectance (%)"]
    assert meta["x_unit"] == "nm"


def test_byte_order_mark_does_not_eat_first_row(tmp_path):
    y = _values(5)
    path = _write_xy(tmp_path / "bom.dpt", WAVENUMBERS, y, encoding="utf-8-sig")

    df, meta = read_ascii_spectra(path)

    assert df.shape == (1, 200)
    assert df.loc["bom", 4000.0] == y[0]
    assert meta["header_lines"] == []


def test_explicit_delimiter(tmp_path):
    y = _values(6)
    path = _write_xy(tmp_path / "semi.txt", WAVENUMBERS, y, delimiter=";")

    df, meta = read_ascii_spectra(path, delimiter=";")

    assert df.shape == (1, 200)
    assert meta["delimiter"] == ";"
    with pytest.raises(ValueError, match="no rows of two or more numeric columns"):
        read_ascii_spectra(path, delimiter="\t")


def test_trailing_delimiter_is_not_a_column(tmp_path):
    path = tmp_path / "trail.dat"
    path.write_text("1,0.5,\n2,0.6,\n3,0.7,\n")

    df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1.0, 2.0, 3.0]
    assert df.iloc[0].tolist() == [0.5, 0.6, 0.7]
    assert meta["n_columns"] == 2


def test_extra_columns_warn_and_use_second_column(tmp_path):
    path = tmp_path / "three.dat"
    path.write_text("1\t0.5\t9\n2\t0.6\t9\n3\t0.7\t9\n")

    with pytest.warns(UserWarning, match="column 2 as y"):
        df, _ = read_ascii_spectra(path)

    assert df.iloc[0].tolist() == [0.5, 0.6, 0.7]


def test_non_numeric_line_inside_data_is_reported(tmp_path):
    path = tmp_path / "gap.dat"
    path.write_text("1,0.5\n2,0.6\nERROR\n3,0.7\n")

    with pytest.warns(UserWarning, match="skipped 1 non-numeric"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1.0, 2.0, 3.0]
    assert meta["n_skipped_lines"] == 1


def test_duplicate_x_values_warn(tmp_path):
    path = tmp_path / "dupe.dat"
    path.write_text("1,0.5\n2,0.6\n2,0.9\n3,0.7\n")

    with pytest.warns(UserWarning, match="duplicate x"):
        df, _ = read_ascii_spectra(path)

    assert df.iloc[0].tolist() == [0.5, 0.6, 0.7]


def test_file_without_numeric_data_raises(tmp_path):
    path = tmp_path / "text.dat"
    path.write_text("just\nsome words\n")
    with pytest.raises(ValueError):
        read_ascii_spectra(path)


def test_missing_path_raises(tmp_path):
    with pytest.raises(ValueError, match="File not found"):
        read_ascii_spectra(tmp_path / "nope.dpt")


def test_unsupported_kwargs_raise(tmp_path):
    path = _write_xy(tmp_path / "s.dpt", WAVENUMBERS, _values(7))
    with pytest.raises(TypeError, match="skiprows"):
        read_ascii_spectra(path, skiprows=1)


# ---------------------------------------------------------------------------
# Folders (the GUI passes the folder path for .dpt/.dat/.asc imports)
# ---------------------------------------------------------------------------


def test_folder_of_headerless_dpt_files(tmp_path):
    ys = {f"sample_{i}": _values(10 + i) for i in range(3)}
    for name, y in ys.items():
        _write_xy(tmp_path / f"{name}.dpt", WAVENUMBERS, y)

    df, meta = read_ascii_spectra(tmp_path)

    assert df.shape == (3, 200)
    assert list(df.index) == ["sample_0", "sample_1", "sample_2"]
    for name, y in ys.items():
        np.testing.assert_array_equal(df.loc[name].to_numpy(), y[::-1])
    assert meta["n_spectra"] == 3
    assert meta["file_format"] == "ascii"
    assert meta["wavelength_range"] == (3801.0, 4000.0)
    assert meta["n_failed"] == 0


def test_folder_mixes_headed_headerless_and_extensions(tmp_path):
    y0, y1, y2 = _values(20), _values(21), _values(22)
    _write_xy(tmp_path / "a.dpt", WAVENUMBERS, y0)
    _write_xy(tmp_path / "b.DAT", WAVENUMBERS, y1, delimiter="\t", header=["x\ty"])
    _write_xy(tmp_path / "c.asc", WAVENUMBERS, y2, delimiter=" ", header=["# comment"])
    (tmp_path / "ignored.csv").write_text("not,an,ascii,spectrum\n")

    df, meta = read_ascii_spectra(tmp_path, delimiter=None)

    assert list(df.index) == ["a", "b", "c"]
    assert df.loc["a", 4000.0] == y0[0]
    assert df.loc["b", 4000.0] == y1[0]
    assert df.loc["c", 4000.0] == y2[0]
    assert meta["n_spectra"] == 3


def test_folder_reports_unreadable_files(tmp_path):
    _write_xy(tmp_path / "good.dpt", WAVENUMBERS, _values(30))
    (tmp_path / "bad.dpt").write_text("garbage only\n")

    with pytest.warns(UserWarning, match="Could not read 1 of 2"):
        df, meta = read_ascii_spectra(tmp_path)

    assert list(df.index) == ["good"]
    assert meta["n_failed"] == 1
    assert meta["failed_files"][0].startswith("bad.dpt")


def test_folder_with_different_x_grids_warns(tmp_path):
    _write_xy(tmp_path / "a.dpt", WAVENUMBERS, _values(40))
    _write_xy(tmp_path / "b.dpt", WAVENUMBERS[:-10], _values(41, 190))

    with pytest.warns(UserWarning, match="do not share one x grid"):
        df, _ = read_ascii_spectra(tmp_path)

    assert df.shape == (2, 200)
    assert df.loc["b"].isna().sum() == 10


def test_folder_without_ascii_files_raises(tmp_path):
    (tmp_path / "x.csv").write_text("a,b\n")
    with pytest.raises(ValueError, match="No .dpt, .dat, or .asc files"):
        read_ascii_spectra(tmp_path)


def test_folder_unit_from_agreeing_headers(tmp_path):
    for i in range(2):
        _write_xy(tmp_path / f"s{i}.dpt", WAVENUMBERS, _values(50 + i), header=["Wavenumber,Abs"])
    _write_xy(tmp_path / "s9.dpt", WAVENUMBERS, _values(59))  # states no unit

    _, meta = read_ascii_spectra(tmp_path)

    assert meta["x_unit"] == "cm-1"


def test_read_spectra_ascii_dispatch_handles_file_and_folder(tmp_path):
    y = _values(60)
    folder = tmp_path / "folder"
    folder.mkdir()
    path = _write_xy(folder / "only.dpt", WAVENUMBERS, y)

    df_file, _ = read_spectra(path, format="ascii")
    df_dir, _ = read_spectra(folder, format="ascii")

    pd.testing.assert_frame_equal(df_file, df_dir)
    assert df_file.loc["only", 4000.0] == y[0]


def test_write_read_round_trip_with_header(tmp_path):
    from spectral_predict.io import write_ascii_spectra

    x = np.linspace(400, 2500, 211)
    original = pd.DataFrame([_values(70, 211)], index=["rt"], columns=x)
    path = tmp_path / "rt.txt"
    write_ascii_spectra(original, path, include_header=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean file must not warn
        df, meta = read_ascii_spectra(path)

    np.testing.assert_allclose(df.columns.to_numpy(dtype=float), x)
    np.testing.assert_allclose(df.iloc[0].to_numpy(), original.iloc[0].to_numpy())
    assert meta["column_names"] == ["Wavelength", "Intensity"]
    # The writer labels x "Wavelength" whatever its unit, so that word sets no unit
    assert meta["x_unit_detection_method"] == "default"
