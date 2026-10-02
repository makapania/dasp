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
    with pytest.raises(ValueError, match="no rows with numeric values"):
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

    with pytest.warns(UserWarning, match="skipped 1 line"):
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

    _, meta = read_ascii_spectra(tmp_path)

    assert meta["x_unit"] == "cm-1"
    assert meta["x_unit_confidence"] == 90.0
    assert meta["import_warnings"] == []


def test_folder_unit_stated_by_some_files_only_is_low_confidence(tmp_path):
    _write_xy(tmp_path / "s0.dpt", WAVENUMBERS, _values(50), header=["Wavenumber,Abs"])
    for i in range(1, 4):
        _write_xy(tmp_path / f"s{i}.dpt", WAVENUMBERS, _values(50 + i))  # no unit stated

    with pytest.warns(UserWarning, match="Only 1 of 4 ASCII files state an x unit"):
        _, meta = read_ascii_spectra(tmp_path)

    assert meta["x_unit"] == "cm-1"
    assert meta["x_unit_confidence"] == 60.0
    assert meta["x_unit_detection_method"] == "ascii_header_partial"
    assert any("Only 1 of 4" in m for m in meta["import_warnings"])


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


# ---------------------------------------------------------------------------
# Review round 1 (Codex / GLM) regressions and gaps
# ---------------------------------------------------------------------------


def test_decimal_comma_with_comma_delimiter_reads_points_with_warning(tmp_path):
    """Decimal commas in a comma-delimited file are not valid CSV: read as points, warn.

    '4000,5,0,123' reads as x=4000, y=5 (as pd.read_csv would), and the import
    warning names the alternative so the user sees it in the GUI dialog.
    """
    path = tmp_path / "de.dat"
    path.write_text("4000,5,0,123\n3999,5,0,124\n3998,5,0,125\n")

    with pytest.warns(UserWarning, match="decimal-comma"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [3998.0, 3999.0, 4000.0]
    assert df.iloc[0].tolist() == [5.0, 5.0, 5.0]
    assert any("decimal=','" in m and "';'" in m for m in meta["import_warnings"])


def test_integer_comma_columns_accepted_with_explicit_decimal_point(tmp_path):
    path = tmp_path / "ints.dat"
    path.write_text("400,12,7\n401,13,7\n402,14,7\n")

    with pytest.warns(UserWarning, match="columns 3\\+ ignored"):
        df, meta = read_ascii_spectra(path, decimal=".")

    assert df.iloc[0].tolist() == [12.0, 13.0, 14.0]
    assert meta["decimal"] == "."


def test_semicolon_decimal_comma_explicit(tmp_path):
    """The delimiter=';', decimal=',' route that pd.read_csv used to provide."""
    path = tmp_path / "de.dat"
    path.write_text("Wellenzahl;Absorbanz\n1000,5;0,123\n1001,5;0,124\n")

    df, meta = read_ascii_spectra(path, delimiter=";", decimal=",")

    assert df.columns.tolist() == [1000.5, 1001.5]
    assert df.iloc[0].tolist() == [0.123, 0.124]
    assert meta["decimal"] == ","


@pytest.mark.parametrize("sep", [";", "\t", " "])
def test_decimal_comma_auto_fallback_warns(tmp_path, sep):
    path = tmp_path / "de.dat"
    path.write_text(f"1000,5{sep}0,123\n1001,5{sep}0,124\n1002{sep}0,125\n")

    with pytest.warns(UserWarning, match="decimal comma"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.5, 1001.5, 1002.0]
    assert df.iloc[0].tolist() == [0.123, 0.124, 0.125]
    assert meta["decimal"] == ","


def test_decimal_comma_conflicts_with_comma_delimiter(tmp_path):
    path = _write_xy(tmp_path / "s.dat", WAVENUMBERS, _values(80))
    with pytest.raises(ValueError, match="cannot be combined"):
        read_ascii_spectra(path, delimiter=",", decimal=",")


def test_unknown_kwargs_still_raise_alongside_decimal(tmp_path):
    path = _write_xy(tmp_path / "s.dpt", WAVENUMBERS, _values(81))
    with pytest.raises(TypeError, match="encoding"):
        read_ascii_spectra(path, decimal=".", encoding="latin-1")


def test_read_spectra_forwards_decimal(tmp_path):
    path = tmp_path / "de.txt"
    path.write_text("1000,5;0,123\n1001,5;0,124\n")

    df, _ = read_spectra(path, format="ascii", delimiter=";", decimal=",")

    assert df.columns.tolist() == [1000.5, 1001.5]


def test_unit_comes_from_x_column_heading_only(tmp_path):
    path = tmp_path / "multi.dat"
    path.write_text("Wavelength (nm),Intensity,Wavenumber (cm-1)\n1000,0.2,10000\n1001,0.3,9990\n")

    with pytest.warns(UserWarning, match="columns 3\\+ ignored"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0]
    assert meta["x_unit"] == "nm"


def test_unit_from_other_header_lines_when_unambiguous(tmp_path):
    path = tmp_path / "inst.dat"
    path.write_text("XUNITS=1/CM\nx y\n1000 0.2\n1001 0.3\n")

    _, meta = read_ascii_spectra(path)

    assert meta["x_unit"] == "cm-1"


def test_header_naming_both_units_outside_columns_sets_none(tmp_path):
    path = tmp_path / "both.dat"
    path.write_text("converted from nm to cm-1\n1000 0.2\n1001 0.3\n")

    _, meta = read_ascii_spectra(path)

    assert meta["x_unit_detection_method"] == "default"


def test_folder_with_conflicting_explicit_units_is_refused(tmp_path):
    x = np.array([1000.0, 1001.0, 1002.0])
    _write_xy(tmp_path / "a.dat", x, _values(90, 3), header=["Wavenumber (cm-1),A"])
    _write_xy(tmp_path / "b.dat", x, _values(91, 3), header=["Wavelength (nm),A"])

    with pytest.raises(ValueError, match=r"different x units.*a\.dat.*b\.dat"):
        read_ascii_spectra(tmp_path)


def test_ignored_text_column_does_not_reject_rows(tmp_path):
    path = tmp_path / "flags.dat"
    path.write_text("Wavelength,Intensity,Quality\n1000,0.2,OK\n1001,0.3,BAD\n1002,0.4,OK\n")

    with pytest.warns(UserWarning, match="3 non-numeric"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0, 1002.0]
    assert df.iloc[0].tolist() == [0.2, 0.3, 0.4]
    assert meta["column_names"] == ["Wavelength", "Intensity", "Quality"]
    assert meta["n_skipped_lines"] == 0


def test_inline_comments_are_stripped(tmp_path):
    path = tmp_path / "notes.dat"
    path.write_text("# header comment\n1000,0.1 # good\n1001,0.2#ok\n1002,0.3\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0, 1002.0]
    assert df.iloc[0].tolist() == [0.1, 0.2, 0.3]


def test_single_point_file_fails_in_folder(tmp_path):
    _write_xy(tmp_path / "good.dpt", WAVENUMBERS, _values(100))
    (tmp_path / "one.dpt").write_text("1,0.5\n")

    with pytest.warns(UserWarning, match="Could not read 1 of 2"):
        df, meta = read_ascii_spectra(tmp_path)

    assert list(df.index) == ["good"]
    assert meta["n_failed"] == 1
    assert "at least 2" in meta["failed_files"][0]


def test_folder_of_only_one_point_files_raises(tmp_path):
    (tmp_path / "one.dpt").write_text("1,0.5\n")
    with pytest.raises(ValueError, match="No valid ASCII spectra"):
        read_ascii_spectra(tmp_path)


def test_all_nan_x_file_fails_in_folder(tmp_path):
    _write_xy(tmp_path / "good.dpt", WAVENUMBERS, _values(101))
    (tmp_path / "nan.dpt").write_text("nan,0.5\nnan,0.6\nnan,0.7\n")

    with pytest.warns(UserWarning, match="Could not read 1 of 2"):
        df, meta = read_ascii_spectra(tmp_path)

    assert list(df.index) == ["good"]
    assert meta["n_failed"] == 1


def test_no_finite_y_is_rejected(tmp_path):
    path = tmp_path / "nany.dat"
    path.write_text("1,nan\n2,nan\n3,nan\n")
    with pytest.raises(ValueError, match="no finite y"):
        read_ascii_spectra(path)


def test_delimiter_that_parses_most_lines_wins(tmp_path):
    """A lone '2,0' metadata line must not pin the comma delimiter for tab data."""
    y = _values(102)
    path = _write_xy(tmp_path / "tabs.dat", WAVENUMBERS, y, delimiter="\t", header=["2,0"])

    df, meta = read_ascii_spectra(path)

    assert meta["delimiter"] == "\t"
    assert df.shape == (1, 200)
    assert meta["header_lines"] == ["2,0"]


def test_folder_does_not_pick_up_txt_files(tmp_path):
    from spectral_predict.io import _detect_directory_format

    _write_xy(tmp_path / "s.dpt", WAVENUMBERS, _values(103))
    (tmp_path / "README.txt").write_text("Notes about this folder\n")

    df, _ = read_ascii_spectra(tmp_path)

    assert list(df.index) == ["s"]
    assert _detect_directory_format(tmp_path) == "ascii"


def test_txt_only_folder_is_not_ascii(tmp_path):
    from spectral_predict.io import _detect_directory_format

    _write_xy(tmp_path / "s.txt", WAVENUMBERS, _values(104))

    assert _detect_directory_format(tmp_path) == "unknown"
    with pytest.raises(ValueError, match="No .dpt, .dat, or .asc files"):
        read_ascii_spectra(tmp_path)
    # A single .txt file is still readable
    df, _ = read_ascii_spectra(tmp_path / "s.txt")
    assert df.shape == (1, 200)


def test_read_spectra_auto_detects_ascii_folder(tmp_path):
    _write_xy(tmp_path / "a.dpt", WAVENUMBERS, _values(105))
    _write_xy(tmp_path / "b.dpt", WAVENUMBERS, _values(106))
    (tmp_path / "reference.csv").write_text("id,y\na,1\nb,2\n")

    df, meta = read_spectra(tmp_path)

    assert list(df.index) == ["a", "b"]
    assert meta["file_format"] == "ascii"


# ---------------------------------------------------------------------------
# Review round 2: number-format interpretation, quoting, header units
# ---------------------------------------------------------------------------


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_one_short_row_does_not_defeat_decimal_comma_refusal(tmp_path):
    path = _write(tmp_path, "de.dat", "4000,5,0,123\n3999,5\n3998,5,0,125\n")
    with pytest.raises(ValueError, match="decimal-comma"):
        read_ascii_spectra(path)


def test_dot_in_ignored_note_does_not_hide_decimal_comma(tmp_path):
    path = _write(tmp_path, "note.dat", "1000,5,0,123,note.v1\n1001,5,0,124,note.v1\n")

    with pytest.warns(UserWarning, match="decimal-comma"):
        _, meta = read_ascii_spectra(path)

    assert any("decimal-comma" in m for m in meta["import_warnings"])


def test_integer_rows_with_text_column_are_accepted(tmp_path):
    path = _write(tmp_path, "flags.dat", "1000,5,OK\n1001,6,OK\n1002,7,BAD\n")

    with pytest.warns(UserWarning, match="columns 3\\+ ignored"):
        df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0, 1002.0]
    assert df.iloc[0].tolist() == [5.0, 6.0, 7.0]


@pytest.mark.parametrize("text", ["1,000,0.123\n1,001,0.124\n", "1,000.5,0.123\n1,001.5,0.124\n"])
def test_thousands_separators_are_refused(tmp_path, text):
    path = _write(tmp_path, "thousands.dat", text)
    with pytest.raises(ValueError, match="leading zero"):
        read_ascii_spectra(path)


def test_integer_rows_do_not_block_decimal_comma_reading(tmp_path):
    """'1001,5;0,123' must not be dropped because the other rows parse either way."""
    path = _write(tmp_path, "mixed.dat", "1000;0\n1001,5;0,123\n1002;1\n")

    with pytest.warns(UserWarning, match="decimal comma"):
        df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.5, 1002.0]
    assert df.iloc[0].tolist() == [0.0, 0.123, 1.0]
    assert meta["decimal"] == ","
    assert meta["n_skipped_lines"] == 0


def test_tied_interpretations_with_different_values_are_refused(tmp_path):
    text = "1000;0,5\n1001;0,6\n1002,0.7\n1003,0.8\n"
    path = _write(tmp_path, "tie.dat", text)

    with pytest.raises(ValueError, match="ambiguous number format"):
        read_ascii_spectra(path)
    # An explicit choice resolves it (the other rows are then skipped, with a warning)
    with pytest.warns(UserWarning, match="skipped 2 line"):
        df, _ = read_ascii_spectra(path, delimiter=";", decimal=",")
    assert df.columns.tolist() == [1000.0, 1001.0]


def test_identical_ties_are_not_ambiguous(tmp_path):
    # Tab and whitespace, '.' and ',' decimals all read these rows identically
    path = _write(tmp_path, "ints.dat", "1000\t5\n1001\t6\n")

    df, meta = read_ascii_spectra(path)

    assert df.iloc[0].tolist() == [5.0, 6.0]
    assert meta["delimiter"] == "\t"
    assert meta["decimal"] == "."


def test_two_integer_comma_fields_warn(tmp_path):
    path = _write(tmp_path, "pairs.dat", "4000,5\n3999,6\n3998,7\n")

    with pytest.warns(UserWarning, match="two comma-separated integers"):
        df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [3998.0, 3999.0, 4000.0]


def test_quoted_fields_are_read(tmp_path):
    text = '"Wavelength","Intensity"\n"1000","0.1"\n"1001","0.2"\n"1002","0.3"\n'
    path = _write(tmp_path, "quoted.csv", text)

    df, meta = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0, 1002.0]
    assert df.iloc[0].tolist() == [0.1, 0.2, 0.3]
    assert meta["column_names"] == ["Wavelength", "Intensity"]


def test_quoted_decimal_comma_fields(tmp_path):
    path = _write(tmp_path, "quoted_de.dat", '"1000,5";"0,1"\n"1001,5";"0,2"\n')

    with pytest.warns(UserWarning, match="decimal comma"):
        df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.5, 1001.5]
    assert df.iloc[0].tolist() == [0.1, 0.2]


def test_quoted_whitespace_fields(tmp_path):
    path = _write(tmp_path, "quoted_ws.dat", '"1000" "0.1"\n"1001" "0.2"\n')

    df, _ = read_ascii_spectra(path)

    assert df.iloc[0].tolist() == [0.1, 0.2]


def test_unsplittable_whitespace_heading_does_not_promote_other_column_unit(tmp_path):
    text = "X Intensity Wavenumber (cm-1)\n1000 0.2 10000\n1001 0.3 9990\n"
    path = _write(tmp_path, "ws.dat", text)

    with pytest.warns(UserWarning, match="not for the x column"):
        _, meta = read_ascii_spectra(path)

    assert meta["x_unit"] == "nm"
    assert meta["x_unit_detection_method"] == "default"
    assert meta["x_unit_confidence"] == 50.0
    assert any("not for the x column" in m for m in meta["import_warnings"])


def test_two_column_whitespace_heading_with_one_unit(tmp_path):
    text = "Wavenumber (cm-1) Absorbance\n1000 0.2\n1001 0.3\n"
    path = _write(tmp_path, "ws2.dat", text)

    _, meta = read_ascii_spectra(path)

    assert meta["x_unit"] == "cm-1"


def test_semicolon_inline_note_rows_are_skipped_not_misread(tmp_path):
    path = _write(tmp_path, "semi_note.dat", "1000,0.1\n1001,0.2 ; check\n1002,0.3\n")

    with pytest.warns(UserWarning, match="skipped 1 line"):
        df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1002.0]


def test_folder_failure_message_names_file_once(tmp_path):
    _write_xy(tmp_path / "good.dpt", WAVENUMBERS, _values(200))
    (tmp_path / "one.dpt").write_text("1,0.5\n")

    with pytest.warns(UserWarning):
        _, meta = read_ascii_spectra(tmp_path)

    assert meta["failed_files"][0].startswith("one.dpt: need at least 2")
    assert "one.dpt: one.dpt" not in meta["failed_files"][0]


# ---------------------------------------------------------------------------
# Review round 3
# ---------------------------------------------------------------------------


def test_nan_values_do_not_make_identical_readings_ambiguous(tmp_path):
    path = _write(tmp_path, "nan.dat", "1000\t0.1\n1001\tnan\n1002\t0.3\n")

    df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == [1000.0, 1001.0, 1002.0]
    assert df.iloc[0, 0] == 0.1 and np.isnan(df.iloc[0, 1]) and df.iloc[0, 2] == 0.3


@pytest.mark.parametrize(
    "text",
    [
        # Field counts differ, and a decimal-comma split explains it
        "1000,5,0,123\n1001.5,0.124\n",
        "4000,5,0,123\n3999,5\n3998,5,0,125\n",
        # The decimal-point reading repeats x; a thousands split explains it
        "1,234,0.1\n1,236,0.2\n1,238,0.3\n",
        "1,234.5,0.1\n1,236.5,0.2\n",
        # ...or a decimal-comma split does
        "4000,5,0,123\n4000,7,0,124\n",
    ],
)
def test_inconsistent_decimal_point_reading_is_refused(tmp_path, text):
    path = _write(tmp_path, "amb.dat", text)
    with pytest.raises(ValueError, match="comma-separated values may be"):
        read_ascii_spectra(path)


@pytest.mark.parametrize(
    "rows, warns",
    [
        # x, integer y, point-decimal third field: no decimal-comma reading exists
        ([f"{4000 - i},{1523 + i},0.5" for i in range(5)], False),
        ([f"{1100 + i},{250 + i},1.0e-3" for i in range(5)], False),
        # Integer Raman shift and counts plus float columns
        ([f"{200 + i},{3000 + 7 * i},0.{i}1,1.5" for i in range(5)], False),
        # Integer nm 350-2500 with integer counts and a float column: the 3-digit
        # rows also fit a thousands split, so these load with a warning
        ([f"{nm},{100 + nm % 7},0.25" for nm in range(350, 2501, 50)], True),
        (["400,100,0.25", "401,102,0.26", "402,104,0.27"], True),
        (["1,234,0.123", "2,345,0.124"], True),
    ],
)
def test_legitimate_comma_files_load(tmp_path, rows, warns):
    path = _write(tmp_path, "ok.dat", "\n".join(rows) + "\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df, meta = read_ascii_spectra(path)

    expected_x = sorted(float(r.split(",")[0]) for r in rows)
    assert df.columns.tolist() == expected_x
    comma_notes = [m for m in meta["import_warnings"] if "decimal-comma" in m]
    assert bool(comma_notes) == warns
    assert bool([w for w in caught if "decimal-comma" in str(w.message)]) == warns


def test_competing_rows_accepted_with_explicit_decimal(tmp_path):
    path = _write(tmp_path, "amb.dat", "1,234,0.123\n2,345,0.124\n")

    with pytest.warns(UserWarning, match="columns 3\\+ ignored"):
        df, _ = read_ascii_spectra(path, decimal=".")

    assert df.columns.tolist() == [1.0, 2.0]


def test_unit_in_trailing_heading_field_of_two_column_data_is_not_used(tmp_path):
    path = _write(tmp_path, "trail.dat", "X Y Wavenumber_cm-1_ref\n1000 0.2\n1001 0.3\n")

    with pytest.warns(UserWarning, match="not for the x column"):
        _, meta = read_ascii_spectra(path)

    assert meta["x_unit"] == "nm"
    assert meta["x_unit_detection_method"] == "default"


def test_unit_in_leading_heading_fields_of_two_column_data_is_used(tmp_path):
    path = _write(tmp_path, "lead.dat", "Wavelength (nm) Reflectance\n1000 0.2\n1001 0.3\n")

    _, meta = read_ascii_spectra(path)

    assert meta["x_unit"] == "nm"
    assert meta["x_unit_detection_method"] == "ascii_header"


# ---------------------------------------------------------------------------
# Review round 5
# ---------------------------------------------------------------------------


def test_folder_summary_never_hides_number_format_warnings(tmp_path):
    """a/b/c carry harmless extra-column notes; z.dat's decimal-comma note must survive."""
    for name in ("a", "b", "c"):
        _write(tmp_path, f"{name}.dat", "1000,0.1,x\n1001,0.2,x\n1002,0.3,x\n")
    _write(tmp_path, "z.dat", "1000,5,0,123\n1001,5,0,124\n1002,5,0,125\n")

    with pytest.warns(UserWarning):
        _, meta = read_ascii_spectra(tmp_path)

    warnings_text = meta["import_warnings"]
    comma = [m for m in warnings_text if "decimal-comma" in m]
    assert len(comma) == 1 and comma[0].startswith("z.dat:")
    extra = [m for m in warnings_text if "ignored extra columns" in m]
    assert len(extra) == 1
    for name in ("a.dat", "b.dat", "c.dat", "z.dat"):
        assert name in extra[0]


def test_folder_lists_every_file_with_number_format_warning(tmp_path):
    names = [f"f{i}.dat" for i in range(6)]
    for name in names:
        _write(tmp_path, name, "1000,5,0,123\n1001,5,0,124\n")

    with pytest.warns(UserWarning):
        _, meta = read_ascii_spectra(tmp_path)

    comma = [m for m in meta["import_warnings"] if "decimal-comma" in m]
    assert sorted(m.split(":", 1)[0] for m in comma) == names


def test_exponent_fragments_raise_the_ambiguity_warning(tmp_path):
    path = _write(tmp_path, "exp.dat", "4,123E+3,0,123\n3,999E+3,0,124\n")

    with pytest.warns(UserWarning, match="decimal-comma"):
        df, meta = read_ascii_spectra(path)

    # The default decimal-point reading is unchanged
    assert df.columns.tolist() == [3.0, 4.0]
    assert any("decimal-comma" in m for m in meta["import_warnings"])


def test_exponent_with_point_decimal_third_field_is_no_evidence(tmp_path):
    path = _write(tmp_path, "exp_ok.dat", "4,123E+3,0.5\n3,999E+3,0.6\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, meta = read_ascii_spectra(path)

    assert not [m for m in meta["import_warnings"] if "decimal-comma" in m]
    assert not [w for w in caught if "decimal-comma" in str(w.message)]


@pytest.mark.parametrize(
    "text, x",
    [
        ("0400,0.5\n0401,0.6\n0402,0.7\n", [400.0, 401.0, 402.0]),  # zero-padded x
        ("1000,05\n1001,06\n1002,07\n", [1000.0, 1001.0, 1002.0]),  # zero-padded y
    ],
)
def test_zero_padding_without_a_competing_split_loads(tmp_path, text, x):
    path = _write(tmp_path, "pad.dat", text)

    df, _ = read_ascii_spectra(path)

    assert df.columns.tolist() == x


def test_leading_zero_with_a_thousands_split_still_refuses(tmp_path):
    path = _write(tmp_path, "thousands.dat", "1,000,0.123\n2,000,0.124\n")
    with pytest.raises(ValueError, match="leading zero"):
        read_ascii_spectra(path)
