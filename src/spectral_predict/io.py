"""I/O functions for reading spectral data and reference files."""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union


def _heuristic_x_unit(columns) -> str:
    """Guess x-axis unit from column value range.

    NIR/VIS wavelengths are typically 350-2500 nm.
    FTIR wavenumbers are typically 400-4000 cm⁻¹ (or up to ~15000 cm⁻¹).
    Overlap zone: 400-2500 could be either, but >4000 is almost certainly nm,
    and values in 400-4000 with nothing above 4000 is likely cm⁻¹.
    """
    vals = np.asarray(columns, dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    # Strong nm indicators: max > 4000 (no wavenumber instrument goes that high commonly)
    if vmax > 5000:
        return 'nm'
    # Strong cm-1 indicator: typical mid-IR range
    if vmin >= 400 and vmax <= 4000:
        # Could be either — but if spacing is uniform and small, likely cm-1
        # Default to nm for ambiguous cases (backwards compatible)
        return 'nm'
    return 'nm'


def convert_x_axis(columns: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert x-axis values between nm and cm⁻¹.

    Parameters
    ----------
    columns : array-like
        X-axis values to convert.
    from_unit : str
        Source unit ('nm' or 'cm-1').
    to_unit : str
        Target unit ('nm' or 'cm-1').

    Returns
    -------
    np.ndarray
        Converted values. Note: nm ↔ cm⁻¹ inverts sort order.
    """
    columns = np.asarray(columns, dtype=float)
    if from_unit == to_unit:
        return columns
    if {from_unit, to_unit} == {'nm', 'cm-1'}:
        return 1e7 / columns
    raise ValueError(f"Unsupported conversion: {from_unit} -> {to_unit}")


def read_csv_spectra(path):
    """
    Read spectral data from CSV file.

    Supports two formats:
    - Wide format: first column = id, remaining columns = numeric wavelengths (nm)
    - Long format (single spectrum): wavelength[_nm], value columns → pivoted to wide

    Parameters
    ----------
    path : str or Path
        Path to CSV file

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = id, columns = float wavelengths (nm)
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if df.shape[0] == 0:
        raise ValueError(f"Empty CSV file: {path}")

    # Detect long format: look for wavelength and value columns
    wl_cols = [c for c in df.columns if c.lower() in ["wavelength", "wavelength_nm"]]
    val_cols = [
        c for c in df.columns if c.lower() in ["value", "intensity", "reflectance", "pct_reflect"]
    ]

    if wl_cols and val_cols:
        # Long format - single spectrum
        wl_col = wl_cols[0]
        val_col = val_cols[0]

        # Use filename (without extension) as ID
        sample_id = path.stem

        # Pivot to wide format
        df_wide = df[[wl_col, val_col]].copy()
        df_wide = df_wide.dropna()

        # Convert to wide: single row with wavelengths as columns (vectorized)
        # Create dict from wavelength -> value without iterrows() for better performance
        wavelengths = df_wide[wl_col].astype(float).values
        values = df_wide[val_col].values
        result = pd.DataFrame([values], columns=wavelengths, index=[sample_id])

        # Sort columns by wavelength
        result = result[sorted(result.columns)]

    else:
        # Wide format
        # First column is ID, rest should be numeric wavelengths
        # BUT: may also have non-numeric columns (like target variables)
        id_col = df.columns[0]
        df = df.set_index(id_col)

        # Parse column names as wavelengths, filtering out non-numeric columns
        wl_cols = {}
        non_wl_cols = []
        for col in df.columns:
            try:
                wl_cols[col] = float(col)
            except ValueError:
                # Non-numeric column (e.g., target variable like 'N', 'protein', etc.)
                non_wl_cols.append(col)

        if not wl_cols:
            raise ValueError(f"No numeric wavelength columns found. Columns: {list(df.columns)}")

        # Drop non-wavelength columns for spectral data
        if non_wl_cols:
            print(f"Note: Ignoring non-wavelength columns: {non_wl_cols}")
            df = df.drop(columns=non_wl_cols)

        # Rename columns to floats and sort
        df = df.rename(columns=wl_cols)
        df = df[sorted(df.columns)]
        result = df

    # Validate
    if result.shape[1] < 100:
        raise ValueError(f"Expected at least 100 wavelengths, got {result.shape[1]}")

    # Check wavelengths are strictly increasing (allowing for floating point tolerance)
    wls = np.array(result.columns)
    if not np.all(wls[1:] > wls[:-1]):
        raise ValueError("Wavelengths must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(result)
    value_scale = infer_reflectance_scale(result) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Compile metadata
    metadata = {
        'n_spectra': len(result),
        'wavelength_range': (result.columns.min(), result.columns.max()),
        'file_format': 'csv',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return result, metadata


def _is_likely_reference_csv(path: Union[str, Path]) -> bool:
    """
    Heuristic to determine if a CSV file is a reference/metadata file vs spectral data.

    Reference CSVs typically have few columns with non-numeric header names (e.g.,
    'SampleID', 'Nitrogen', 'Moisture'). Spectral CSVs have many numeric column
    names (wavelengths) or wavelength/value columns (long format).

    Only reads the header row for efficiency.

    Parameters
    ----------
    path : str or Path
        Path to CSV file

    Returns
    -------
    bool
        True if the file looks like a reference/metadata CSV, False if it looks spectral.
    """
    path = Path(path)
    try:
        # Read only the header row
        df_header = pd.read_csv(path, nrows=0)
        columns = list(df_header.columns)

        if len(columns) < 2:
            return False

        # Check for long-format spectral CSV (wavelength/value columns)
        lower_cols = [str(c).lower() for c in columns]
        wl_names = {"wavelength", "wavelength_nm"}
        val_names = {"value", "intensity", "reflectance", "pct_reflect"}
        if wl_names.intersection(lower_cols) and val_names.intersection(lower_cols):
            return False  # This is a spectral file (long format)

        # Count numeric column names (wavelengths in wide format)
        n_numeric = 0
        for col in columns:
            try:
                float(col)
                n_numeric += 1
            except (ValueError, TypeError):
                pass

        # Spectral wide-format CSVs have many numeric columns (wavelengths)
        # Reference CSVs have mostly text column names
        if n_numeric >= 50:
            return False  # Likely spectral data (wide format)

        # Few or no numeric columns -> likely a reference/metadata file
        return True

    except Exception:
        # If we can't read it, assume it's not a reference file
        return False


def read_csv_dir(
    csv_dir: Union[str, Path],
    exclude_files: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read individual CSV spectrum files from a directory.

    Each CSV file should contain a single spectrum in long format
    (wavelength, value columns) or wide format. Files are combined into
    a single wide DataFrame.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing CSV files
    exclude_files : list of str, optional
        Filenames to exclude (e.g., reference CSV)

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = filename, columns = wavelengths (nm)
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.
    """
    csv_dir = Path(csv_dir)

    if not csv_dir.exists():
        raise ValueError(f"Directory not found: {csv_dir}")

    if not csv_dir.is_dir():
        raise ValueError(f"Not a directory: {csv_dir}")

    exclude_set = {f.lower() for f in (exclude_files or [])}

    # Find CSV files, excluding reference files
    csv_files = sorted(
        f for f in csv_dir.glob("*.csv")
        if f.name.lower() not in exclude_set
    )

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV files")

    # Read each file
    spectra = {}
    skipped = []
    for csv_file in csv_files:
        stem = csv_file.stem
        try:
            df_single, _ = read_csv_spectra(csv_file)
            # read_csv_spectra returns a DataFrame; use first row
            if len(df_single) == 1:
                series = df_single.iloc[0]
                series.index = series.index.round(0).astype(int)
                series = series[~series.index.duplicated(keep='first')]
                spectra[stem] = series
            else:
                # Multi-row CSV in a directory - use all rows with prefixed IDs
                for idx in df_single.index:
                    key = f"{stem}_{idx}" if len(df_single) > 1 else stem
                    series = df_single.loc[idx]
                    series.index = series.index.round(0).astype(int)
                    series = series[~series.index.duplicated(keep='first')]
                    spectra[key] = series
        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")
            skipped.append(csv_file.name)

    if len(spectra) == 0:
        raise ValueError(
            f"No valid spectra could be read from {csv_dir}. "
            f"Skipped files: {skipped}"
        )

    # Combine into wide matrix
    df = pd.DataFrame(spectra).T

    # Sort columns (wavelengths)
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 100:
        raise ValueError(f"Expected at least 100 wavelengths, got {df.shape[1]}")

    # Check wavelengths are increasing
    wls = np.array(df.columns)
    if not np.all(wls[1:] > wls[:-1]):
        raise ValueError("Wavelengths must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'csv_dir',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return df, metadata


def _rename_duplicate_ids(index: pd.Index) -> tuple:
    """
    Rename duplicate IDs by adding .1, .2, etc. suffix.

    Parameters
    ----------
    index : pd.Index
        Index that may contain duplicates

    Returns
    -------
    tuple
        (new_index, n_renamed, rename_mapping) where:
        - new_index: pd.Index with renamed duplicates
        - n_renamed: number of IDs that were renamed
        - rename_mapping: dict mapping original IDs to list of new IDs
          e.g., {"SampleA": ["SampleA", "SampleA.1", "SampleA.2"]}
    """
    if not index.duplicated().any():
        return index, 0, {}

    new_ids = []
    seen = {}
    rename_mapping = {}  # Track original -> [new names] for warning display

    for idx in index:
        if idx in seen:
            seen[idx] += 1
            new_id = f"{idx}.{seen[idx]}"
            new_ids.append(new_id)
            rename_mapping[idx].append(new_id)
        else:
            seen[idx] = 0
            new_ids.append(idx)
            rename_mapping[idx] = [idx]  # Start tracking this ID

    # Count how many were renamed (exclude originals)
    n_renamed = sum(1 for idx in new_ids if '.' in str(idx) and str(idx).rsplit('.', 1)[-1].isdigit())

    # Filter rename_mapping to only include IDs that had duplicates
    rename_mapping = {k: v for k, v in rename_mapping.items() if len(v) > 1}

    return pd.Index(new_ids), n_renamed, rename_mapping


def read_reference_csv(path, id_column):
    """
    Read reference file (CSV or Excel) with target variables.

    Parameters
    ----------
    path : str or Path
        Path to reference file (CSV or Excel)
    id_column : str
        Column name to use as index (e.g., 'sample_id', 'filename')

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by id_column
    """
    path = Path(path)

    # Detect file type and read accordingly
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {path}. Available: {list(df.columns)}")

    # Check for duplicate IDs BEFORE setting index
    duplicates = df[id_column].duplicated()
    if duplicates.any():
        dup_ids = df.loc[duplicates, id_column].unique()
        n_dups = duplicates.sum()
        print(f"\n⚠️ WARNING: Found {n_dups} duplicate sample IDs in reference file!")
        print(f"Duplicate IDs: {list(dup_ids[:10])}")
        if len(dup_ids) > 10:
            print(f"... and {len(dup_ids) - 10} more")
        print("\nKeeping FIRST occurrence of each duplicate. Please check your file.\n")

        # Keep only first occurrence of each ID
        df = df[~duplicates]

    df = df.set_index(id_column)
    return df


def _normalize_filename_for_matching(filename):
    """
    Normalize filename for flexible matching.

    Removes common file extensions, spaces, and converts to lowercase.

    Parameters
    ----------
    filename : str
        Filename to normalize

    Returns
    -------
    str
        Normalized filename
    """
    # Convert to string in case it's not
    filename = str(filename)

    # Remove common extensions (ASD family derived from ASD_EXTENSIONS so it can't
    # drift when a new ASD extension like .sco is added).
    for ext in [*ASD_EXTENSIONS, ".csv", ".txt", ".spc"]:
        if filename.lower().endswith(ext):
            filename = filename[: -len(ext)]
            break

    # Remove spaces and convert to lowercase
    filename = filename.replace(" ", "").lower()

    return filename


def align_xy(X, ref, id_column, target, return_alignment_info=False, drop_na_y=True):
    """
    Align spectral data with reference target variable.

    Uses smart filename matching to handle:
    - Files with/without extensions (e.g., "sample.asd" vs "sample")
    - Filenames with/without spaces (e.g., "Spectrum 001" vs "Spectrum001")
    - Case differences

    Parameters
    ----------
    X : pd.DataFrame
        Spectral data (wide format, rows = ids)
    ref : pd.DataFrame
        Reference data with targets, indexed by id
    id_column : str
        The id column name (for error messages)
    target : str
        Target variable name
    return_alignment_info : bool, optional
        If True, also return a dict with detailed alignment info
    drop_na_y : bool, optional
        If True (default), drop rows with NaN target values.
        Set to False during import to preserve all rows.

    Returns
    -------
    X_aligned : pd.DataFrame
        Aligned spectral data
    y : pd.Series
        Target values, same order as X_aligned
    alignment_info : dict (only if return_alignment_info=True)
        Dictionary containing:
        - 'matched_ids': List of IDs that were successfully matched
        - 'unmatched_spectra': List of spectral IDs with no reference
        - 'unmatched_reference': List of reference IDs with no spectra
        - 'n_nan_dropped': Number of samples dropped due to NaN targets
        - 'used_fuzzy_matching': Whether fuzzy matching was used
    """
    if target not in ref.columns:
        raise ValueError(
            f"Target '{target}' not found in reference. Available: {list(ref.columns)}"
        )

    # Track alignment info
    used_fuzzy_matching = False
    original_X_ids = set(X.index)
    original_ref_ids = set(ref.index)

    # Try exact match first
    common_ids = X.index.intersection(ref.index)

    # If no exact matches, try normalized matching
    if len(common_ids) == 0:
        used_fuzzy_matching = True
        print("No exact ID matches found. Trying flexible filename matching...")

        # Create mapping of normalized names to original names
        X_norm_map = {_normalize_filename_for_matching(idx): idx for idx in X.index}
        ref_norm_map = {_normalize_filename_for_matching(idx): idx for idx in ref.index}

        # Find common normalized IDs
        common_norm_ids = set(X_norm_map.keys()).intersection(set(ref_norm_map.keys()))

        if len(common_norm_ids) == 0:
            # Show helpful debug info
            print(f"\nSpectral data IDs (first 5): {list(X.index[:5])}")
            print(f"Reference IDs (first 5): {list(ref.index[:5])}")
            print(f"\nNormalized spectral IDs (first 5): {list(X_norm_map.keys())[:5]}")
            print(f"Normalized reference IDs (first 5): {list(ref_norm_map.keys())[:5]}")
            raise ValueError(
                f"No matching IDs between spectral data and reference. "
                f"Check that '{id_column}' values match between files.\n"
                f"Tried matching with and without file extensions/spaces."
            )

        # Build alignment using normalized matching
        # Map: ref_id -> X_id
        # IMPORTANT: Sort to ensure deterministic order (Python's hash randomization
        # causes different set iteration order between program restarts)
        id_mapping = {}
        for norm_id in sorted(common_norm_ids):
            ref_id = ref_norm_map[norm_id]
            X_id = X_norm_map[norm_id]
            id_mapping[ref_id] = X_id

        print(f"Matched {len(id_mapping)} samples using flexible filename matching")

        # Create aligned datasets using the mapping
        aligned_X_ids = [id_mapping[ref_id] for ref_id in id_mapping.keys()]
        aligned_ref_ids = list(id_mapping.keys())

        X_aligned = X.loc[aligned_X_ids]
        y = ref.loc[aligned_ref_ids, target]

        # Track matched SPECTRAL IDs before index replacement (for fuzzy matching)
        matched_spectral_ids = list(aligned_X_ids)

        # Ensure same order and index
        X_aligned.index = aligned_ref_ids
        y.index = aligned_ref_ids

    else:
        # Use exact matches
        if len(common_ids) < len(X):
            print(
                f"Warning: {len(X) - len(common_ids)} samples from spectral data have no reference"
            )

        if len(common_ids) < len(ref):
            print(
                f"Warning: {len(ref) - len(common_ids)} samples from reference have no spectral data"
            )

        X_aligned = X.loc[common_ids]
        y = ref.loc[common_ids, target]

        # Track matched SPECTRAL IDs (for exact matching, these are just common_ids)
        matched_spectral_ids = list(common_ids)

    # Track truly unmatched samples BEFORE NaN filtering
    # (so NaN-dropped samples aren't counted as "unmatched")
    # Use matched_spectral_ids which contains the original spectral file IDs
    matched_before_nan_filter = matched_spectral_ids
    # Also track the matched reference IDs for accurate unmatched_reference reporting
    matched_ref_ids = list(X_aligned.index)

    # Drop any NaN targets (only when drop_na_y is True)
    n_nan_dropped = 0
    if drop_na_y:
        valid_mask = ~y.isna()
        if not valid_mask.all():
            n_nan_dropped = (~valid_mask).sum()
            print(f"Warning: Dropping {n_nan_dropped} samples with missing target values")
            X_aligned = X_aligned[valid_mask]
            y = y[valid_mask]

    if len(X_aligned) == 0:
        raise ValueError("No valid samples after alignment")

    # SAFETY CHECK: Ensure perfect alignment before returning
    if len(X_aligned) != len(y):
        raise ValueError(
            f"Alignment error: X has {len(X_aligned)} samples but y has {len(y)} samples. "
            f"This should never happen - please report this bug."
        )

    if not X_aligned.index.equals(y.index):
        print(f"Warning: X and y have different indices after alignment. Realigning...")
        # Force alignment by ensuring same index
        X_aligned.index = y.index

    # Prepare alignment info if requested
    if return_alignment_info:
        matched_ids = list(X_aligned.index)
        # Use matched_before_nan_filter to exclude NaN-dropped samples from "unmatched" count
        # Convert to strings for sorting to handle mixed types (str/int/float)
        unmatched_spectra = sorted([str(x) for x in (original_X_ids - set(matched_before_nan_filter))])
        # For unmatched_reference, use matched_ref_ids (not matched_before_nan_filter which contains spectral IDs)
        unmatched_reference = sorted([str(x) for x in (original_ref_ids - set(matched_ref_ids))])

        alignment_info = {
            'matched_ids': matched_ids,
            'unmatched_spectra': unmatched_spectra,
            'unmatched_reference': unmatched_reference,
            'n_nan_dropped': n_nan_dropped,
            'used_fuzzy_matching': used_fuzzy_matching
        }

        return X_aligned, y, alignment_info

    return X_aligned, y


# Extensions recognized as ASD spectral files (.sco = legacy float32 ASD-v1 binary,
# e.g. older FieldSpec exports). Single source of truth so the many directory-detection
# sites across the backend and GUI can't drift out of sync when a new extension is added.
ASD_EXTENSIONS = (".asd", ".sig", ".sco")


def list_asd_files(directory) -> list[Path]:
    """Return sorted ASD files (``.asd``/``.sig``/``.sco``, case-insensitive) in a directory.

    Centralizes ASD extension handling so every folder-detection site stays in sync with
    :data:`ASD_EXTENSIONS`. Case-insensitive on every platform (unlike raw ``glob``).

    Args:
        directory: Directory to scan (str or Path).

    Returns:
        Sorted list of matching file paths (empty if the directory has no ASD files).
    """
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ASD_EXTENSIONS
    )


def read_asd_dir(asd_dir, reader_mode="auto"):
    """
    Read ASD files from a directory.

    Supports ASCII .sig and ASCII .asd files (text format).
    Binary .asd files require SpecDAL or will raise an error.

    Parameters
    ----------
    asd_dir : str or Path
        Directory containing ASD files
    reader_mode : str
        Reader mode ('auto', 'python', 'rs-prospectr', 'rs-asdreader')

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = filename, columns = wavelengths (nm)
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.
    """
    asd_dir = Path(asd_dir)

    if not asd_dir.exists():
        raise ValueError(f"Directory not found: {asd_dir}")

    if not asd_dir.is_dir():
        raise ValueError(f"Not a directory: {asd_dir}")

    # Find ASD files (.sco = legacy float32 ASD-v1 binary, e.g. older FieldSpec exports)
    asd_files = list_asd_files(asd_dir)

    if len(asd_files) == 0:
        raise ValueError(f"No .sig, .asd, or .sco files found in {asd_dir}")

    print(f"Found {len(asd_files)} ASD files")

    # Read each file
    spectra = {}
    duplicate_stems = []
    skipped = []
    for asd_file in sorted(asd_files):
        stem = asd_file.stem

        # Check for duplicate filenames (without extension)
        if stem in spectra:
            duplicate_stems.append(stem)
            print(f"⚠️ WARNING: Duplicate filename '{stem}' - later file will overwrite earlier one")

        try:
            if _is_binary_asd(asd_file):
                # Binary ASD (legacy float32 or modern) - bypass the text reader.
                spectrum = _handle_binary_asd(asd_file, reader_mode)
            else:
                try:
                    spectrum = _read_single_asd_ascii(asd_file, reader_mode)
                except UnicodeDecodeError:
                    # Text read hit binary bytes - fall back to the binary handler.
                    spectrum = _handle_binary_asd(asd_file, reader_mode)
            if spectrum is not None:
                spectra[stem] = spectrum
            else:
                # A None return (SpecDAL failed/unavailable, or a binary layout this
                # reader can't decode) means the file couldn't be read. Report it as a
                # skip rather than silently dropping it from the folder.
                msg = f"{asd_file.name}: could not be decoded (unsupported or unreadable binary ASD)"
                skipped.append(msg)
                print(f"Warning: skipping {msg}")
        except Exception as e:
            # Corrupt/unreadable file (e.g. read_legacy_asd's ValueError on a
            # truncated .sco, or a modern binary with no SpecDAL). Skip it and keep
            # loading the rest of the folder — one bad file must not abort the whole
            # import — then report every skipped file together after the loop.
            msg = str(e) if asd_file.name in str(e) else f"{asd_file.name}: {e}"
            skipped.append(msg)
            print(f"Warning: skipping {msg}")

    if duplicate_stems:
        print(f"\n⚠️ Found {len(duplicate_stems)} duplicate ASD filenames (ignoring extensions)")
        print(f"Duplicates: {duplicate_stems[:10]}")
        if len(duplicate_stems) > 10:
            print(f"... and {len(duplicate_stems) - 10} more")
        print("Keeping LAST occurrence of each duplicate.\n")

    if skipped:
        print(f"\n[!] Skipped {len(skipped)} unreadable/corrupt file(s):")
        for line in skipped[:10]:
            print(f"  - {line}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if len(spectra) == 0:
        detail = f" ({'; '.join(skipped[:5])})" if skipped else ""
        raise ValueError(f"No valid spectra could be read{detail}")

    # Combine into wide matrix
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (wavelengths)
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 100:
        raise ValueError(f"Expected at least 100 wavelengths, got {df.shape[1]}")

    # Check wavelengths are increasing
    wls = np.array(df.columns)
    if not np.all(wls[1:] > wls[:-1]):
        raise ValueError("Wavelengths must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'asd',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 95.0,
        'x_unit_detection_method': 'asd_native',
    }

    return df, metadata


def _read_single_asd_ascii(asd_file, reader_mode):
    """
    Read a single ASCII ASD file (.sig or ASCII .asd).

    Uses heuristics to detect wavelength and reflectance columns:
    - Find rows with >= 2 numeric tokens
    - First column = wavelength (nm)
    - Last numeric column = reflectance

    Parameters
    ----------
    asd_file : Path
        Path to ASD file
    reader_mode : str
        Reader mode (currently unused for ASCII)

    Returns
    -------
    pd.Series
        Spectrum with wavelengths as index
    """
    # Read as text
    with open(asd_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse numeric rows
    wavelengths = []
    values = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse as numeric
        tokens = line.split()
        numeric_tokens = []

        for token in tokens:
            try:
                numeric_tokens.append(float(token))
            except ValueError:
                continue

        # Need at least 2 numeric values (wavelength + value)
        if len(numeric_tokens) >= 2:
            # First = wavelength, last = reflectance
            wavelengths.append(numeric_tokens[0])
            values.append(numeric_tokens[-1])

    if len(wavelengths) == 0:
        raise ValueError(f"No numeric data found in {asd_file.name}")

    # Create series
    df = pd.DataFrame({"wavelength": wavelengths, "value": values})

    # Round wavelengths to 0.01 nm to avoid floating point issues
    df["wavelength"] = df["wavelength"].round(2)

    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset="wavelength", keep="first")

    # Sort by wavelength
    df = df.sort_values("wavelength")

    # Return as Series with wavelength as index
    return pd.Series(df["value"].values, index=df["wavelength"].values)


def _is_binary_asd(asd_file: Union[str, Path]) -> bool:
    """Return True if the file is a binary ASD (magic bytes ``ASD\\0``).

    ASCII ASD/.sig files start with text such as ``ASD Field Spec Pro``, so a 3-byte
    ``ASD`` prefix is not sufficient; the 4th byte (NUL) disambiguates binary files.
    """
    try:
        with open(asd_file, "rb") as f:
            return f.read(4) == b"ASD\x00"
    except OSError:
        return False


def _handle_binary_asd(asd_file, reader_mode):
    """
    Handle binary ASD files: native legacy reader first, then SpecDAL.

    Parameters
    ----------
    asd_file : Path
        Path to binary ASD file
    reader_mode : str
        Reader mode

    Returns
    -------
    pd.Series
        Spectrum with wavelengths as index, or None if cannot read

    Raises
    ------
    ValueError
        If binary ASD cannot be read and SpecDAL not available
    """
    # Legacy float32 ASD-v1 files (e.g. .sco / numbered .000) are misread by SpecDAL
    # as float64 -> all-NaN. Decode them natively first; returns None for other layouts.
    from .readers.asd_native import read_legacy_asd

    legacy = read_legacy_asd(asd_file)
    if legacy is not None:
        return legacy

    if reader_mode == "auto":
        # Try to import SpecDAL
        try:
            from specdal import Spectrum

            # Read with SpecDAL
            spec = Spectrum(filepath=str(asd_file))

            # Extract wavelength and reflectance
            # SpecDAL returns wavelengths and values as numpy arrays
            wavelengths = spec.measurement.index.values  # wavelengths
            reflectance = spec.measurement.values  # reflectance values

            # Create series
            df = pd.DataFrame({"wavelength": wavelengths, "value": reflectance})

            # Round wavelengths to 0.01 nm to avoid floating point issues
            df["wavelength"] = df["wavelength"].round(2)

            # Remove duplicates (keep first)
            df = df.drop_duplicates(subset="wavelength", keep="first")

            # Sort by wavelength
            df = df.sort_values("wavelength")

            # Return as Series with wavelength as index
            return pd.Series(df["value"].values, index=df["wavelength"].values)

        except ImportError:
            raise ValueError(
                f"Binary ASD file detected: {asd_file.name}\n"
                "Options:\n"
                "  1. Export to ASCII format (.sig or ASCII .asd)\n"
                "  2. Install SpecDAL: pip install specdal"
            )
        except Exception as e:
            print(f"Warning: SpecDAL failed to read {asd_file.name}: {e}")
            return None
    else:
        raise ValueError(
            f"Binary ASD file detected: {asd_file.name}. "
            f"Reader mode '{reader_mode}' not yet implemented for binary files."
        )


def read_spc_dir(spc_dir):
    """
    Read SPC (GRAMS/Thermo Galactic) files from a directory.

    Uses the spc-io library to read binary .spc files.

    Parameters
    ----------
    spc_dir : str or Path
        Directory containing SPC files

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = filename, columns = wavelengths (nm)
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.

    Raises
    ------
    ValueError
        If directory doesn't exist, no SPC files found, or spc-io not installed
    """
    spc_dir = Path(spc_dir)

    if not spc_dir.exists():
        raise ValueError(f"Directory not found: {spc_dir}")

    if not spc_dir.is_dir():
        raise ValueError(f"Not a directory: {spc_dir}")

    # Find SPC files
    spc_files = list(spc_dir.glob("*.spc"))

    if len(spc_files) == 0:
        raise ValueError(f"No .spc files found in {spc_dir}")

    print(f"Found {len(spc_files)} SPC files")

    # Try to import spc-io
    try:
        import spc_io
    except ImportError:
        raise ValueError(
            "SPC file support requires the spc-io library.\n"
            "Install it with: pip install spc-io"
        )

    # Read each SPC file
    spectra = {}
    duplicate_stems = []

    # Track SPC xtype/ytype from first successful file
    spc_xunit_detected = False
    spc_x_unit_detected = None
    spc_x_unit_confidence = 50.0
    spc_x_unit_method = 'default'
    spc_y_type_detected = None
    spc_y_type_confidence = 0.0
    spc_y_type_method = ''

    for spc_file in sorted(spc_files):
        stem = spc_file.stem

        # Check for duplicate filenames (without extension)
        if stem in spectra:
            duplicate_stems.append(stem)
            print(f"⚠️ WARNING: Duplicate filename '{stem}' - later file will overwrite earlier one")

        try:
            with open(spc_file, 'rb') as f:
                spc = spc_io.SPC.from_bytes_io(f)

                # Extract first subfile (most common case for single spectra)
                # If multiple subfiles exist, we'll concatenate them or use the first one
                if len(spc) > 1:
                    print(f"Note: {spc_file.name} contains {len(spc)} subfiles, using first subfile")

                subfile = spc[0]
                wavelengths = subfile.xarray
                intensities = subfile.yarray

                # Read xtype/ytype for x-unit and data type detection
                if not spc_xunit_detected:
                    _SPC_XTYPE_MAP = {
                        'XWAVEN': 'cm-1', 'XNMETR': 'nm', 'XUMETR': 'um',
                        'XRAMANS': 'cm-1', 'XHERTZ': None, 'XSEC': None,
                    }
                    _SPC_YTYPE_MAP = {
                        'YABSRB': 'absorbance', 'YTRANS': 'transmittance',
                        'YREFLEC': 'reflectance', 'YEMISN': None,
                    }
                    xtype_attr = getattr(spc, 'xtype', None)
                    ytype_attr = getattr(spc, 'ytype', None)
                    if xtype_attr is not None:
                        xtype_str = str(xtype_attr).split('.')[-1].upper()
                        spc_x_unit = _SPC_XTYPE_MAP.get(xtype_str, None)
                        if spc_x_unit:
                            spc_x_unit_detected = spc_x_unit
                            spc_x_unit_confidence = 95.0
                            spc_x_unit_method = 'spc_xtype'
                            print(f"  SPC xtype: {xtype_str} -> x_unit={spc_x_unit}")
                    if ytype_attr is not None:
                        ytype_str = str(ytype_attr).split('.')[-1].upper()
                        spc_y_type = _SPC_YTYPE_MAP.get(ytype_str, None)
                        if spc_y_type:
                            spc_y_type_detected = spc_y_type
                            spc_y_type_confidence = 95.0
                            spc_y_type_method = 'spc_ytype'
                            print(f"  SPC ytype: {ytype_str} -> data_type={spc_y_type}")
                    spc_xunit_detected = True

                # Create a series with wavelength as index
                spectrum = pd.Series(intensities, index=wavelengths)

                # Round wavelengths to avoid floating point issues
                spectrum.index = spectrum.index.round(2)

                # Remove duplicates (keep first)
                spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

                # Sort by wavelength
                spectrum = spectrum.sort_index()

                spectra[stem] = spectrum

        except Exception as e:
            print(f"Warning: Could not read {spc_file.name}: {e}")

    if duplicate_stems:
        print(f"\n⚠️ Found {len(duplicate_stems)} duplicate SPC filenames")
        print(f"Duplicates: {duplicate_stems[:10]}")
        if len(duplicate_stems) > 10:
            print(f"... and {len(duplicate_stems) - 10} more")
        print("Keeping LAST occurrence of each duplicate.\n")

    if len(spectra) == 0:
        raise ValueError("No valid SPC spectra could be read")

    # Combine into wide matrix
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (wavelengths)
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 100:
        raise ValueError(f"Expected at least 100 wavelengths, got {df.shape[1]}")

    # Check wavelengths are strictly increasing
    wls = np.array(df.columns)
    if not np.all(wls[1:] > wls[:-1]):
        raise ValueError("Wavelengths must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    # Override with SPC ytype metadata if available (higher confidence)
    if spc_y_type_detected and spc_y_type_detected in ('absorbance', 'reflectance'):
        data_type = spc_y_type_detected
        type_confidence = spc_y_type_confidence
        detection_method = spc_y_type_method
        value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0
    elif spc_y_type_detected == 'transmittance':
        # Transmittance maps to reflectance-like handling in UI
        data_type = 'reflectance'
        type_confidence = spc_y_type_confidence
        detection_method = f"spc_ytype(transmittance)"
        value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Determine x-unit: use SPC xtype if detected, else heuristic from value range
    x_unit = spc_x_unit_detected or _heuristic_x_unit(df.columns)
    x_unit_confidence = spc_x_unit_confidence
    x_unit_method = spc_x_unit_method

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'spc',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': x_unit,
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
    }

    print(f"Successfully read {len(df)} SPC spectra with {df.shape[1]} data points")
    print(f"  X-axis unit: {x_unit} (confidence: {x_unit_confidence:.0f}%)")

    return df, metadata


def detect_combined_format(directory_path):
    """
    Detect if directory contains a single combined CSV/TXT file.

    A combined file contains all spectra in one table with:
    - Specimen ID column (optional)
    - Wavelength columns (numeric headers)
    - Target y column

    Parameters
    ----------
    directory_path : str or Path
        Path to directory

    Returns
    -------
    tuple : (bool, str or None)
        (is_combined, filepath) or (False, None)
    """
    from glob import glob
    import os

    directory_path = Path(directory_path)

    if not directory_path.exists() or not directory_path.is_dir():
        return False, None

    # Get all CSV and TXT files
    csv_files = list(directory_path.glob("*.csv"))
    txt_files = list(directory_path.glob("*.txt"))

    all_files = csv_files + txt_files

    # If exactly ONE file, treat as combined format
    if len(all_files) == 1:
        return True, str(all_files[0])

    return False, None


def identify_wavelength_columns(df):
    """
    Identify columns that represent wavelengths.
    Can appear anywhere in the column list.

    Criteria:
    - Column name is numeric (or can be converted to float)
    - Value is in reasonable wavelength range (100-10000 nm)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    list
        List of column names that appear to be wavelengths
    """
    wavelength_cols = []

    for col in df.columns:
        # Try to convert column name to float
        try:
            wavelength = float(str(col).strip().strip('"').strip("'"))

            # Check if in reasonable range for spectroscopy
            if 100 <= wavelength <= 10000:
                wavelength_cols.append(col)
        except (ValueError, TypeError):
            continue

    return wavelength_cols


def auto_detect_specimen_id_column(df, exclude_wavelength_cols):
    """
    Detect specimen ID column with flexible positioning.

    The specimen ID could be:
    - First, last, or middle column
    - String, numeric, or mixed type
    - Named with various conventions
    - **ABSENT** - in which case we return None and generate synthetic IDs

    Detection Priority:
    1. First non-wavelength column if it has good uniqueness (>50% unique)
       - ID columns are typically the first column in spectral data files
    2. Column named 'specimen_id', 'sample_id', 'id', 'sample', 'specimen', etc.
       - BUT only if it has reasonable uniqueness (>20% or >10 unique values)
       - This prevents matching columns like "Sample" containing "Yes/No" categorical data
    3. Column with all/mostly unique values (>80% unique)
    4. First non-wavelength column with object/string dtype
    5. Check if all remaining columns are numeric/y-like → No ID column, return None

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    exclude_wavelength_cols : list
        Wavelength columns to exclude from consideration

    Returns
    -------
    str or None
        Column name of detected specimen ID column, or None if absent
    """
    # Get candidate columns (exclude wavelengths)
    candidate_cols = [col for col in df.columns
                     if col not in exclude_wavelength_cols]

    if not candidate_cols:
        # No non-wavelength columns at all → no ID, no y → error
        raise ValueError("No non-wavelength columns found")

    # Helper function to check uniqueness
    def get_uniqueness(col):
        n_unique = df[col].nunique()
        n_total = len(df[col].dropna())
        if n_total == 0:
            return 0, 0, 0
        return n_unique, n_total, n_unique / n_total

    # If only one candidate column, check if it looks like y data
    if len(candidate_cols) == 1:
        col = candidate_cols[0]
        # If it looks like a target variable (numeric, not unique), assume no ID column
        if pd.api.types.is_numeric_dtype(df[col]):
            _, _, uniqueness_ratio = get_uniqueness(col)
            if uniqueness_ratio < 0.8:  # Not very unique → probably y, not ID
                return None

    # Priority 1: Check first non-wavelength column - IDs are typically first
    # Accept if it has good uniqueness (>50% unique)
    first_col = candidate_cols[0]
    n_unique, n_total, uniqueness_ratio = get_uniqueness(first_col)
    if n_total > 0 and uniqueness_ratio > 0.5:
        return first_col

    # Priority 2: Check for common ID names (case-insensitive)
    # BUT verify the column has reasonable uniqueness (>20% unique or >10 unique values)
    # This prevents matching columns like "Sample" that contain "Yes/No" categorical data
    common_names = [
        'specimen_id', 'sample_id', 'specimen', 'sample', 'id',
        'file_number', 'file_name', 'filename', 'name',
        'sample_name', 'specimen_name', 'sampleid', 'specimenid'
    ]

    for name in common_names:
        matches = [col for col in candidate_cols
                  if col.lower() == name.lower() or
                     col.lower().replace('_', '') == name.lower().replace('_', '')]
        if matches:
            col = matches[0]
            n_unique, n_total, uniqueness_ratio = get_uniqueness(col)
            # Accept if >20% unique OR has >10 unique values (for small datasets)
            if n_total > 0 and (uniqueness_ratio > 0.2 or n_unique > 10):
                return col
            # Otherwise skip this match and continue searching

    # Priority 3: Find column with unique/mostly unique values
    # Specimen IDs should be unique identifiers
    for col in candidate_cols:
        _, n_total, uniqueness_ratio = get_uniqueness(col)
        if n_total > 0 and uniqueness_ratio > 0.8:
            return col

    # Priority 4: Find non-numeric dtype column
    # IDs often contain letters/special characters
    for col in candidate_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            return col

    # Priority 5: Check if all remaining columns are numeric (likely all y-like)
    # If so, assume no ID column present
    all_numeric = all(pd.api.types.is_numeric_dtype(df[col])
                     for col in candidate_cols)

    if all_numeric and len(candidate_cols) <= 3:
        # Likely format: wavelengths + 1-3 y columns, no ID
        return None

    # Priority 6: Fallback to first candidate column
    return candidate_cols[0]


def auto_detect_y_column(df, exclude_cols):
    """
    Detect target y column from remaining columns.
    Could be before or after wavelengths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    exclude_cols : list
        Columns to exclude (wavelengths + specimen ID)

    Detection Priority:
    1. Columns with target-related names (collagen, nitrogen, target, y, class, etc.)
    2. Column with numeric or categorical values
    3. If multiple remaining columns, pick the most "target-like" one

    Returns
    -------
    str
        Column name of detected y column
    """
    # Get candidate columns
    candidate_cols = [col for col in df.columns if col not in exclude_cols]

    if not candidate_cols:
        return None

    # If only one candidate, use it
    if len(candidate_cols) == 1:
        return candidate_cols[0]

    # Priority 1: Check for target-related keywords
    priority_keywords = [
        'collagen', 'nitrogen', 'protein', 'target', 'y', 'value',
        'class', 'label', 'category', 'group', 'type',
        '%', 'percent', 'concentration', 'content', 'amount'
    ]

    for keyword in priority_keywords:
        matches = [col for col in candidate_cols
                  if keyword.lower() in str(col).lower()]
        if matches:
            return matches[0]

    # Priority 2: Prefer numeric columns for regression tasks
    numeric_cols = [col for col in candidate_cols
                   if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        return numeric_cols[0]

    # Priority 3: Fall back to first candidate
    return candidate_cols[0]


def read_combined_csv(filepath, specimen_id_col=None, y_col=None, drop_na_y=True):
    """
    Read a combined CSV/TXT file containing spectra + targets in one table.

    Expected format:
    - One row per specimen
    - Specimen ID column (OPTIONAL - will generate if absent)
    - Wavelength columns (numeric headers, possibly quoted, FLEXIBLE POSITION)
    - Target y column (FLEXIBLE POSITION - before or after wavelengths)
    - Optional metadata columns (preserved and returned)

    Example formats supported:

    Format A: With ID column
    specimen_id, "400", "401", ..., "2400", collagen
    A-53, 0.245, 0.248, ..., 0.156, 6.4

    Format B: Without ID column (will generate Sample_1, Sample_2, ...)
    "400", "401", ..., "2400", collagen
    0.245, 0.248, ..., 0.156, 6.4
    0.312, 0.315, ..., 0.201, 7.9

    Format C: ID anywhere with metadata
    specimen_id, site, depth, "400", "401", ..., "2400", collagen
    A-53, "Site1", 10.5, 0.245, 0.248, ..., 0.156, 6.4

    Parameters
    ----------
    filepath : str or Path
        Path to combined CSV/TXT file
    specimen_id_col : str, optional
        Name of specimen ID column. If None, auto-detect. If "__GENERATE__", force generation.
    y_col : str, optional
        Name of target variable column. If None, auto-detect.
    drop_na_y : bool, optional
        If True (default), remove rows with missing y values. If False, keep all rows with valid
        spectral data even if y is NaN. Useful when loading data for prediction.

    Returns
    -------
    X : pd.DataFrame
        Spectral data (rows=specimens, cols=wavelengths)
    y : pd.Series
        Target values
    metadata_df : pd.DataFrame or None
        Additional metadata columns (rows=specimens, cols=metadata fields)
        None if no metadata columns present
    metadata : dict
        {
            'specimen_id_col': detected column name or "__GENERATED__",
            'y_col': detected column name,
            'wavelength_cols': list of wavelength column names,
            'metadata_cols': list of metadata column names,
            'n_spectra': number of spectra loaded,
            'wavelength_range': (min, max),
            'generated_ids': True if IDs were auto-generated
        }
    """
    filepath = Path(filepath)

    # Step 1: Read file with flexible delimiter
    df = None
    for sep in [',', '\t', ';', r'\s+']:
        try:
            df = pd.read_csv(filepath, sep=sep, engine='python' if sep == r'\s+' else 'c')
            # Check if we got multiple columns (not all in one column)
            if len(df.columns) > 10:  # Reasonable threshold
                break
        except Exception as e:
            continue

    if df is None or len(df.columns) <= 10:
        raise ValueError(f"Could not parse file {filepath} with standard delimiters")

    # Step 2: Clean column names (strip quotes, whitespace)
    df.columns = df.columns.astype(str).str.strip().str.strip('"').str.strip("'")

    # Step 3: Identify wavelength columns FIRST (position-independent)
    wavelength_cols = identify_wavelength_columns(df)

    if len(wavelength_cols) < 100:
        raise ValueError(
            f"Too few wavelength columns detected ({len(wavelength_cols)}). "
            f"Expected at least 100. Detected columns: {wavelength_cols[:10] if wavelength_cols else 'none'}..."
        )

    # Step 4: Identify specimen ID column (from non-wavelength columns)
    # Could be None if no ID column present
    generated_ids = False

    if specimen_id_col is None:
        detected_specimen_id_col = auto_detect_specimen_id_column(df, wavelength_cols)

        if detected_specimen_id_col is None:
            # No ID column detected → generate synthetic IDs
            specimen_ids = pd.Series([f"Sample_{i+1}" for i in range(len(df))],
                                    name="specimen_id")
            generated_ids = True
            specimen_id_col = "__GENERATED__"
        else:
            specimen_id_col = detected_specimen_id_col
            specimen_ids = df[specimen_id_col].astype(str)

    elif specimen_id_col == "__GENERATE__":
        # User explicitly requested generated IDs
        specimen_ids = pd.Series([f"Sample_{i+1}" for i in range(len(df))],
                                name="specimen_id")
        generated_ids = True

    else:
        # User provided specific column name
        if specimen_id_col not in df.columns:
            raise ValueError(f"Specimen ID column '{specimen_id_col}' not found in file")
        specimen_ids = df[specimen_id_col].astype(str)

    # Step 5: Identify y column (from remaining non-wavelength, non-ID columns)
    no_target = (y_col == "__NONE__")

    if no_target:
        y_col = None
    elif y_col is None:
        exclude_cols = wavelength_cols.copy()
        if not generated_ids and specimen_id_col != "__GENERATED__":
            exclude_cols.append(specimen_id_col)

        y_col = auto_detect_y_column(df, exclude_cols)
        if y_col is None:
            no_target = True

    if not no_target:
        if y_col not in df.columns:
            raise ValueError(f"Target y column '{y_col}' not found in file")

    # Step 6: Identify and extract metadata columns
    # Metadata columns = all columns that are NOT wavelengths, NOT specimen ID, NOT target
    all_cols = set(df.columns)
    wavelength_cols_set = set(wavelength_cols)
    used_cols = wavelength_cols_set.copy()
    if not no_target and y_col is not None:
        used_cols.add(y_col)
    if not generated_ids and specimen_id_col != "__GENERATED__":
        used_cols.add(specimen_id_col)

    metadata_cols = sorted(list(all_cols - used_cols))  # Preserve alphabetical order

    # Extract metadata DataFrame (if any metadata columns exist)
    if metadata_cols:
        metadata_df = df[metadata_cols].copy()
        metadata_df.index = specimen_ids
    else:
        metadata_df = None

    # Step 7: Extract spectral data
    X = df[wavelength_cols].copy()
    X.index = specimen_ids

    # Convert spectral data values to numeric (handle any string values from CSV)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Convert wavelength column names to float and sort
    X.columns = X.columns.astype(float)
    X = X.sort_index(axis=1)  # Sort by wavelength

    if no_target:
        # No target variable mode
        y = None
        has_nan_y = pd.Series(False, index=X.index)
    else:
        # Extract target data
        y = df[y_col].copy()
        y.index = specimen_ids

        # Try to convert target values to numeric, but preserve categorical data for classification
        y_numeric = pd.to_numeric(y, errors='coerce')

        # If conversion resulted in mostly NaN values, keep original (likely categorical)
        if y_numeric.isna().sum() > len(y) * 0.5:
            has_nan_y = y.isna() | (y == '') | y.isnull()
        else:
            y = y_numeric
            has_nan_y = y.isna()

    # Check for missing values (NaN) and remove affected specimens
    has_nan_X = X.isna().any(axis=1)

    # Determine which rows to remove based on drop_na_y parameter
    if drop_na_y and not no_target:
        has_nan = has_nan_X | has_nan_y
    else:
        has_nan = has_nan_X

    if has_nan.any():
        n_missing = has_nan.sum()
        missing_specimens = X.index[has_nan].tolist()

        print(f"Warning: Found {n_missing} specimen(s) with missing spectral data. Removing them.")
        print(f"  Removed specimens: {missing_specimens[:10]}")  # Show first 10
        if n_missing > 10:
            print(f"  ... and {n_missing - 10} more")

        # Remove rows with missing values
        X = X[~has_nan]
        if y is not None:
            y = y[~has_nan]
        if metadata_df is not None:
            metadata_df = metadata_df[~has_nan]

    # Report on rows kept with missing y values if drop_na_y=False
    if not no_target and not drop_na_y and has_nan_y.any():
        n_missing_y = has_nan_y.sum()
        print(f"Info: Kept {n_missing_y} specimen(s) with missing target values (useful for prediction).")

    # Step 8: Validation
    # Check for duplicate specimen IDs (only if not generated)
    # Use X.index since specimen_ids may be out of sync after NaN removal
    n_duplicates_renamed = 0
    duplicate_rename_mapping = {}
    if not generated_ids and X.index.duplicated().any():
        # Rename duplicates by adding .1, .2, etc. suffix instead of removing them
        new_index, n_duplicates_renamed, duplicate_rename_mapping = _rename_duplicate_ids(X.index)

        print(f"Warning: Found {n_duplicates_renamed} duplicate specimen IDs. "
              f"Auto-renamed with .1, .2, etc. suffix.")

        # Show examples of renamed IDs
        for orig_id, new_ids in list(duplicate_rename_mapping.items())[:3]:
            print(f"  '{orig_id}' -> {new_ids}")

        # Apply renamed index to all DataFrames
        X.index = new_index
        if y is not None:
            y.index = new_index
        if metadata_df is not None:
            metadata_df.index = new_index

    # Check wavelength ordering
    wavelength_values = X.columns.values
    if not all(wavelength_values[i] < wavelength_values[i+1]
              for i in range(len(wavelength_values)-1)):
        print("Warning: Wavelengths were not strictly increasing. Sorted automatically.")

    # Step 9: Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(X)
    value_scale = infer_reflectance_scale(X) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Step 10: Compile metadata
    metadata = {
        'specimen_id_col': specimen_id_col,
        'y_col': y_col,
        'wavelength_cols': wavelength_cols,
        'metadata_cols': metadata_cols if metadata_cols else [],
        'n_spectra': len(X),
        'wavelength_range': (X.columns.min(), X.columns.max()),
        'file_format': 'combined',
        'generated_ids': generated_ids,
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'duplicates_renamed': n_duplicates_renamed,
        'duplicate_rename_mapping': duplicate_rename_mapping,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return X, y, metadata_df, metadata



def read_jcamp_dir(jcamp_dir):
    """
    Read JCAMP-DX files from a directory.

    Supports .jdx and .dx file extensions.

    Parameters
    ----------
    jcamp_dir : str or Path
        Directory containing JCAMP-DX files

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = filename, columns = wavelengths
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.
    """
    jcamp_dir = Path(jcamp_dir)

    if not jcamp_dir.exists():
        raise ValueError(f"Directory not found: {jcamp_dir}")

    if not jcamp_dir.is_dir():
        raise ValueError(f"Not a directory: {jcamp_dir}")

    # Find JCAMP files (use set to deduplicate on case-insensitive filesystems)
    jcamp_files = sorted(set(
        list(jcamp_dir.glob("*.jdx")) + list(jcamp_dir.glob("*.dx"))
        + list(jcamp_dir.glob("*.JDX")) + list(jcamp_dir.glob("*.DX"))
    ))

    if len(jcamp_files) == 0:
        raise ValueError(f"No .jdx or .dx files found in {jcamp_dir}")

    print(f"Found {len(jcamp_files)} JCAMP-DX files")

    # Read each file
    spectra = {}
    file_metadata = {}
    duplicate_stems = []

    for jcamp_file in sorted(jcamp_files):
        stem = jcamp_file.stem

        # Check for duplicate filenames (without extension)
        if stem in spectra:
            duplicate_stems.append(stem)
            print(f"⚠️ WARNING: Duplicate filename '{stem}' - later file will overwrite earlier one")

        try:
            file_df, metadata = read_jcamp_file(jcamp_file)
            series = file_df.iloc[0]
            # Round wavenumbers to integers to avoid float grid mismatches
            # across files (consistent with read_spc_dir and _apply_wavelength_filter)
            series.index = series.index.round(0).astype(int)
            series = series[~series.index.duplicated(keep='first')]
            spectra[stem] = series
            file_metadata[stem] = metadata
        except Exception as e:
            print(f"Warning: Could not read {jcamp_file.name}: {e}")

    if duplicate_stems:
        print(f"\n⚠️ Found {len(duplicate_stems)} duplicate JCAMP filenames (ignoring extensions)")
        print(f"Duplicates: {duplicate_stems[:10]}")
        if len(duplicate_stems) > 10:
            print(f"... and {len(duplicate_stems) - 10} more")
        print("Keeping LAST occurrence of each duplicate.\n")

    if len(spectra) == 0:
        raise ValueError("No valid spectra could be read")

    # Combine into wide matrix
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (wavelengths/wavenumbers)
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 100:
        raise ValueError(f"Expected at least 100 data points, got {df.shape[1]}")

    # Check x-axis values are increasing
    x_values = np.array(df.columns)
    if not np.all(x_values[1:] > x_values[:-1]):
        raise ValueError("X-axis values must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Get x-axis units from first file's already-resolved metadata
    first_file_meta = next(iter(file_metadata.values()))
    x_unit = first_file_meta.get('x_unit', _heuristic_x_unit(df.columns))
    x_unit_confidence = first_file_meta.get('x_unit_confidence', 50.0)
    x_unit_method = first_file_meta.get('x_unit_detection_method', 'default')

    # Get raw xunits from jcamp_header for metadata passthrough
    jcamp_header = first_file_meta.get('jcamp_header', {})
    xunits = jcamp_header.get('xunits', 'unknown')

    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'jcamp-dx',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'xunits': xunits,
        'file_metadata': file_metadata,
        'x_unit': x_unit,
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
    }

    return df, metadata


def write_jcamp(df, output_dir, title_prefix="spectrum", xunits="1/CM", yunits="ABSORBANCE", metadata=None):
    """
    Write spectral data to JCAMP-DX format files.

    Creates one .jdx file per spectrum (row in DataFrame).

    Parameters
    ----------
    df : pd.DataFrame
        Spectral data (rows = samples, columns = x-axis values)
    output_dir : str or Path
        Output directory for JCAMP files
    title_prefix : str, optional
        Prefix for spectrum titles (default: "spectrum")
    xunits : str, optional
        Units for x-axis (default: "1/CM" for wavenumber)
        Common values: "1/CM", "MICROMETERS", "NANOMETERS"
    yunits : str, optional
        Units for y-axis (default: "ABSORBANCE")
        Common values: "ABSORBANCE", "TRANSMITTANCE", "REFLECTANCE"
    metadata : dict, optional
        Additional metadata to include in JCAMP headers

    Returns
    -------
    list
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for idx, (sample_id, spectrum) in enumerate(df.iterrows()):
        # Prepare data
        x = spectrum.index.values
        y = spectrum.values

        # Create JCAMP file content
        lines = []
        lines.append("##TITLE=" + f"{title_prefix}_{sample_id}")
        lines.append("##JCAMP-DX=5.00")
        lines.append("##DATA TYPE=INFRARED SPECTRUM")
        lines.append("##ORIGIN=spectral-predict")
        lines.append(f"##OWNER=Generated by spectral-predict on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"##XUNITS={xunits}")
        lines.append(f"##YUNITS={yunits}")
        lines.append(f"##FIRSTX={x[0]}")
        lines.append(f"##LASTX={x[-1]}")
        lines.append(f"##NPOINTS={len(x)}")
        lines.append(f"##FIRSTY={y[0]}")
        lines.append(f"##MAXY={np.max(y)}")
        lines.append(f"##MINY={np.min(y)}")
        lines.append("##XFACTOR=1.0")
        lines.append("##YFACTOR=1.0")

        # Add custom metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key not in ['x', 'y', 'title', 'xunits', 'yunits']:
                    lines.append(f"##{key.upper()}={value}")

        # Write data in XY pairs format (simpler than compressed formats)
        # Note: Use decimal format, not scientific notation, because jcamp_parse
        # misinterprets the 'e' in scientific notation as a JCAMP DIF digit.
        lines.append("##XYDATA=(X++(Y..Y))")
        for i in range(len(x)):
            lines.append(f"{x[i]:.6f} {y[i]:.8f}")

        lines.append("##END=")

        # Write to file
        output_path = output_dir / f"{sample_id}.jdx"
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        created_files.append(output_path)

    print(f"Wrote {len(created_files)} JCAMP-DX files to {output_dir}")

    return created_files


def read_ascii_spectra(path):
    """
    Read ASCII variant spectral files (.dpt, .dat, .asc).

    Supports:
    - Bruker OPUS .dpt (data point table) format
    - Generic .dat and .asc ASCII formats
    - Various delimiters (tab, space, comma)
    - Comment lines (starting with # or %)
    - Both X,Y pair format and wide format

    Parameters
    ----------
    path : str or Path
        Path to ASCII file or directory

    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame - Wide matrix with rows = id, columns = wavelengths
        - metadata: dict - Contains data_type, type_confidence, detection_method, etc.
    """
    path = Path(path)

    # If directory, read all ASCII files
    if path.is_dir():
        return _read_ascii_dir(path)

    # Single file - read it
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    # Read file and detect format
    df, x_col, y_col = _parse_ascii_file(path)

    if df is None or df.shape[0] == 0:
        raise ValueError(f"No data found in file: {path}")

    # Convert to wide format (single spectrum)
    sample_id = path.stem

    # Create wide format DataFrame
    result = pd.DataFrame([df[y_col].values], columns=df[x_col].values, index=[sample_id])

    # Sort columns by x-axis value
    result = result[sorted(result.columns)]

    # Validate
    if result.shape[1] < 100:
        raise ValueError(f"Expected at least 100 data points, got {result.shape[1]}")

    # Check x-axis values are increasing
    x_values = np.array(result.columns)
    if not np.all(x_values[1:] > x_values[:-1]):
        raise ValueError("X-axis values must be strictly increasing")

    # Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(result)
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Compile metadata
    metadata = {
        'n_spectra': 1,
        'wavelength_range': (result.columns.min(), result.columns.max()),
        'file_format': path.suffix[1:],  # Remove leading dot
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return result, metadata


def _read_ascii_dir(directory):
    """
    Read all ASCII spectral files from a directory.

    Parameters
    ----------
    directory : Path
        Directory containing ASCII files

    Returns
    -------
    tuple
        (df, metadata) - Combined spectra and metadata
    """
    # Find ASCII files (use set to deduplicate on case-insensitive filesystems)
    ascii_files = sorted(set(
        list(directory.glob("*.dpt")) + list(directory.glob("*.dat"))
        + list(directory.glob("*.asc")) + list(directory.glob("*.DPT"))
        + list(directory.glob("*.DAT")) + list(directory.glob("*.ASC"))
    ))

    if len(ascii_files) == 0:
        raise ValueError(f"No .dpt, .dat, or .asc files found in {directory}")

    print(f"Found {len(ascii_files)} ASCII files")

    # Read each file
    spectra = {}
    duplicate_stems = []

    for ascii_file in sorted(ascii_files):
        stem = ascii_file.stem

        # Check for duplicate filenames
        if stem in spectra:
            duplicate_stems.append(stem)
            print(f"⚠️ WARNING: Duplicate filename '{stem}' - later file will overwrite earlier one")

        try:
            df, x_col, y_col = _parse_ascii_file(ascii_file)
            if df is not None and len(df) > 0:
                spectra[stem] = pd.Series(df[y_col].values, index=df[x_col].values)
        except Exception as e:
            print(f"Warning: Could not read {ascii_file.name}: {e}")

    if duplicate_stems:
        print(f"\n⚠️ Found {len(duplicate_stems)} duplicate ASCII filenames")
        print(f"Duplicates: {duplicate_stems[:10]}")
        if len(duplicate_stems) > 10:
            print(f"... and {len(duplicate_stems) - 10} more")
        print("Keeping LAST occurrence of each duplicate.\n")

    if len(spectra) == 0:
        raise ValueError("No valid spectra could be read")

    # Combine into wide matrix
    df = pd.DataFrame(spectra).T

    # Sort columns
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 100:
        raise ValueError(f"Expected at least 100 data points, got {df.shape[1]}")

    # Check x-axis values are increasing
    x_values = np.array(df.columns)
    if not np.all(x_values[1:] > x_values[:-1]):
        raise ValueError("X-axis values must be strictly increasing")

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'ascii',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return df, metadata


def _parse_ascii_file(filepath):
    """
    Parse a single ASCII spectral file with flexible format detection.

    Handles:
    - Comment lines (# or %)
    - Various delimiters (tab, space, comma, semicolon)
    - Header rows
    - X,Y pair format

    Parameters
    ----------
    filepath : Path
        Path to ASCII file

    Returns
    -------
    tuple
        (df, x_col, y_col) - DataFrame and column names, or (None, None, None) if failed
    """
    # Read file content
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Remove comment lines and empty lines
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('%'):
            data_lines.append(stripped)

    if len(data_lines) == 0:
        return None, None, None

    # Detect delimiter
    first_line = data_lines[0]
    delimiters = ['\t', ' ', ',', ';']
    delimiter = None
    max_splits = 0

    for delim in delimiters:
        splits = len([x for x in first_line.split(delim) if x.strip()])
        if splits > max_splits:
            max_splits = splits
            delimiter = delim

    if delimiter is None or max_splits < 2:
        return None, None, None

    # Parse data
    x_values = []
    y_values = []

    for line in data_lines:
        tokens = [t.strip() for t in line.split(delimiter) if t.strip()]

        if len(tokens) < 2:
            continue

        try:
            # Try to parse first two numeric values
            x_val = float(tokens[0])
            # Y value could be second column or last column
            y_val = float(tokens[-1] if len(tokens) > 2 else tokens[1])

            x_values.append(x_val)
            y_values.append(y_val)
        except (ValueError, IndexError):
            # Skip non-numeric lines (could be headers)
            continue

    if len(x_values) == 0:
        return None, None, None

    # Create DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })

    # Remove duplicates
    df = df.drop_duplicates(subset='x', keep='first')

    # Sort by x
    df = df.sort_values('x')

    return df, 'x', 'y'


def detect_spectral_data_type(X, metadata=None):
    """
    Intelligently detect whether spectral data is reflectance or absorbance.

    Uses multiple criteria including value ranges, peak directions, and metadata
    to determine the data type with a confidence score.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Spectral data matrix (rows=specimens, columns=wavelengths)
    metadata : dict, optional
        Metadata dictionary that may contain column names or other hints

    Returns
    -------
    tuple
        (data_type, confidence, method) where:
        - data_type: str, either "reflectance" or "absorbance"
        - confidence: float, 0-100 confidence score
        - method: str, detection method used
    """
    import numpy as np
    import pandas as pd

    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        data = X.values
        col_names = [str(c).lower() for c in X.columns]
    else:
        data = np.array(X)
        col_names = []

    # Initialize confidence scores for each type
    reflectance_score = 0
    absorbance_score = 0
    detection_methods = []

    # Flatten data for statistics
    flat_data = data.flatten()

    # Ensure data is numeric (handle case where conversion hasn't happened yet)
    try:
        flat_data = flat_data.astype(float)
    except (ValueError, TypeError):
        # If conversion fails, return default with low confidence
        return ("reflectance", 50.0, "non_numeric_data")

    # Remove NaN values
    flat_data = flat_data[~np.isnan(flat_data)]

    if len(flat_data) == 0:
        return ("reflectance", 50.0, "no_valid_data")

    # Calculate statistics
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)
    mean_val = np.mean(flat_data)
    negative_ratio = np.mean(flat_data < 0)

    # Criterion 1: Absolute bounds check (weight: ~35%)
    if max_val > 5.0 and min_val >= 0.0 and max_val <= 110.0:
        # Likely percent reflectance (0-100 range)
        reflectance_score += 35
        detection_methods.append("bounds_check(percent_reflectance_range)")
    elif max_val > 1.5:
        # Definitely absorbance - reflectance can't exceed 1.0 (unless % reflectance)
        absorbance_score += 35
        detection_methods.append("bounds_check(max>1.5)")
    elif max_val <= 1.0 and min_val >= 0.0:
        # All values in [0, 1] - likely reflectance
        if mean_val > 0.3:
            # High mean in [0,1] range strongly suggests reflectance
            reflectance_score += 35
            detection_methods.append("bounds_check(0-1_range)")
        else:
            # Low mean could be dark sample reflectance or low absorbance
            reflectance_score += 20
            absorbance_score += 10
            detection_methods.append("bounds_check(0-1_low_mean)")
    elif max_val > 1.2 and max_val <= 1.5:
        # Ambiguous range - could be reflectance with mild overshoot or low absorbance
        absorbance_score += 10
        reflectance_score += 10
        detection_methods.append("bounds_check(ambiguous_1.2-1.5)")
    else:
        # Negative values or very low values
        if min_val < -0.5:
            # Significantly negative suggests absorbance (or errors)
            absorbance_score += 35
            detection_methods.append("bounds_check(negative_values)")
        elif min_val < 0.0:
            # Mildly negative values still lean absorbance
            absorbance_score += 20
            detection_methods.append("bounds_check(negative_values)")
        else:
            absorbance_score += 10
            detection_methods.append("bounds_check(near_zero)")

    # Bonus: proportion of negative values (reflectance rarely negative)
    if negative_ratio > 0.01:
        absorbance_score += 10
        detection_methods.append("bounds_check(negative_fraction>1%)")

    # Criterion 2: Mean value analysis (weight: ~20%)
    if 0.3 <= mean_val <= 0.9:
        # Typical reflectance range
        reflectance_score += 20
        detection_methods.append("mean_check(reflectance_range)")
    elif mean_val > 1.0:
        # High mean suggests absorbance
        absorbance_score += 20
        detection_methods.append("mean_check(absorbance_range)")
    elif mean_val < 0.3 and max_val <= 1.0:
        # Low mean in bounded range - dark reflectance
        reflectance_score += 15
        detection_methods.append("mean_check(dark_reflectance)")
    else:
        # Ambiguous mean
        detection_methods.append("mean_check(ambiguous)")

    # Guardrail: typical reflectance stats should not be overridden by noisy shape votes
    if (0.3 <= mean_val <= 0.9) and (negative_ratio < 0.001) and (min_val > -0.05) and (max_val <= 1.5):
        reflectance_score += 20
        detection_methods.append("reflectance_guard(typical_range_low_negative)")

    # Guardrail: absorbance often shows negative baselines or substantial negatives
    if (negative_ratio > 0.02) or (min_val < -0.05 and mean_val < 0.4):
        absorbance_score += 20
        detection_methods.append("absorbance_guard(negative_baseline)")

    # Criterion 3: Peak/valley shape analysis is currently disabled because it
    # misclassifies reflectance data with sharp peaks after smoothing.
    detection_methods.append("peak_analysis(disabled)")

    # Criterion 4: Column name analysis (bonus weight: +15%)
    if metadata and 'column_names' in metadata:
        col_names.extend([str(c).lower() for c in metadata['column_names']])

    if col_names:
        refl_keywords = ['refl', 'reflect', '%r', 'pct_r', 'r_']
        abs_keywords = ['abs', 'absorb', 'absorbance', 'a_']

        col_string = ' '.join(col_names)

        if any(kw in col_string for kw in refl_keywords):
            reflectance_score += 15
            detection_methods.append("metadata(reflectance_keywords)")
        elif any(kw in col_string for kw in abs_keywords):
            absorbance_score += 15
            detection_methods.append("metadata(absorbance_keywords)")

    # Normalize scores to 0-100 range
    total_score = reflectance_score + absorbance_score
    if total_score > 0:
        reflectance_confidence = (reflectance_score / total_score) * 100
        absorbance_confidence = (absorbance_score / total_score) * 100
    else:
        # No evidence either way - default to reflectance (ASD files typically are)
        reflectance_confidence = 60.0
        absorbance_confidence = 40.0
        detection_methods.append("default(no_evidence)")

    # Determine final classification
    if reflectance_confidence > absorbance_confidence:
        data_type = "reflectance"
        confidence = reflectance_confidence
    else:
        data_type = "absorbance"
        confidence = absorbance_confidence

    method_str = "; ".join(detection_methods)

    return (data_type, confidence, method_str)


def infer_reflectance_scale(X) -> float:
    """
    Infer whether reflectance-like data is unit (0-1) or percent (0-100) scale.

    Returns 100.0 for likely percent reflectance, otherwise 1.0.
    """
    import numpy as np
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        data = X.values
    else:
        data = np.array(X)

    try:
        flat = data.astype(float).flatten()
    except (ValueError, TypeError):
        return 1.0

    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return 1.0

    min_val = np.min(flat)
    max_val = np.max(flat)

    # Percent reflectance typically lives in 0-100 range and exceeds 5
    if max_val > 5.0 and min_val >= 0.0 and max_val <= 110.0:
        return 100.0

    return 1.0


# ============================================================================
# UNIFIED I/O ARCHITECTURE
# ============================================================================


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect spectral file format from file extension and/or content.

    Supports auto-detection for:
    - CSV (wide or long format)
    - Excel (.xlsx, .xls)
    - ASD (.asd, .sig)
    - SPC (.spc)
    - JCAMP-DX (.jdx, .dx, .jcm)
    - ASCII text variants (.txt, .dat)
    - Bruker OPUS (numbered extensions: .0, .1, .2, etc.)
    - PerkinElmer (.sp)
    - Agilent (.seq, .dat in specific format)

    Parameters
    ----------
    path : str or Path
        Path to file or directory

    Returns
    -------
    str
        Format identifier: 'csv', 'excel', 'asd', 'spc', 'jcamp', 'ascii',
        'opus', 'perkinelmer', 'agilent', 'directory', or 'unknown'

    Examples
    --------
    >>> detect_format('data/spectra.csv')
    'csv'
    >>> detect_format('data/sample.0')  # Bruker OPUS
    'opus'
    >>> detect_format('data/asd_files/')
    'directory'
    """
    path = Path(path)

    # Check if directory
    if path.is_dir():
        return 'directory'

    # Get extension (lowercase for comparison)
    ext = path.suffix.lower()

    # Extension-based detection
    format_map = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.asd': 'asd',
        '.sig': 'asd',
        '.sco': 'asd',
        '.spc': 'spc',
        '.jdx': 'jcamp',
        '.dx': 'jcamp',
        '.jcm': 'jcamp',
        '.txt': 'ascii',
        '.dat': 'ascii',
        '.dpt': 'ascii',
        '.asc': 'ascii',
        '.sp': 'perkinelmer',
        '.spa': 'omnic',
        '.spg': 'omnic',
        '.seq': 'agilent',
        '.dmt': 'agilent',
        '.asp': 'agilent',
        '.bsw': 'agilent',
    }

    if ext in format_map:
        return format_map[ext]

    # Check for Bruker OPUS numbered extensions (.0, .1, .2, etc.)
    if ext and ext[1:].isdigit():
        return 'opus'

    # Fallback: try to detect from content (magic bytes)
    if path.exists() and path.is_file():
        try:
            with open(path, 'rb') as f:
                header = f.read(512)

            # Binary ASD magic (legacy float32 ASD-v1, e.g. .sco files)
            if header[:4] == b'ASD\x00':
                return 'asd'

            # SPC magic bytes
            if header[:2] == b'\x4d\x4b':  # 'MK' in ASCII
                return 'spc'

            # JCAMP magic
            if b'##TITLE' in header or b'##JCAMP' in header:
                return 'jcamp'

            # Bruker OPUS magic
            if b'OPUS' in header[:100]:
                return 'opus'

        except Exception:
            pass

    return 'unknown'


def read_spectra(
    path: Union[str, Path],
    format: str = 'auto',
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal spectral data reader with automatic format detection.

    This is the main entry point for reading spectral data. It automatically
    detects the file format and dispatches to the appropriate reader function.

    Supported Formats:
    - CSV (wide or long format)
    - Excel (.xlsx, .xls)
    - ASD files (.asd, .sig) - ASCII or binary
    - SPC (GRAMS/Thermo Galactic)
    - JCAMP-DX (.jdx, .dx)
    - ASCII text files (.txt, .dat)
    - Bruker OPUS (requires brukeropus package)
    - PerkinElmer (requires specio package)
    - Agilent (requires agilent-ir-formats package)
    - Thermo Omnic (.spa, .spg) (requires spectrochempy-omnic package)

    Parameters
    ----------
    path : str or Path
        Path to file or directory containing spectral data
    format : str, optional
        Format specification. Options:
        - 'auto': Auto-detect format (default)
        - 'csv': CSV file
        - 'excel': Excel file
        - 'asd': ASD files (single file or directory)
        - 'spc': SPC files (single file or directory)
        - 'jcamp': JCAMP-DX file
        - 'ascii': Generic ASCII text file
        - 'opus': Bruker OPUS
        - 'perkinelmer': PerkinElmer format
        - 'agilent': Agilent format
    **kwargs
        Additional format-specific arguments passed to reader functions

    Returns
    -------
    df : pd.DataFrame
        Spectral data in wide format (rows=samples, columns=wavelengths)
    metadata : dict
        Format-specific metadata including:
        - 'file_format': str - Detected or specified format
        - 'n_spectra': int - Number of spectra loaded
        - 'wavelength_range': tuple - (min_wl, max_wl) in nm
        - 'data_type': str - 'reflectance' or 'absorbance'
        - 'type_confidence': float - Confidence in data type detection (0-100)
        - Additional format-specific fields

    Raises
    ------
    ValueError
        If format cannot be detected or file cannot be read
    ImportError
        If required package for format is not installed

    Examples
    --------
    >>> # Auto-detect CSV format
    >>> df, meta = read_spectra('data/spectra.csv')

    >>> # Explicitly specify Excel format
    >>> df, meta = read_spectra('data/spectra.xlsx', format='excel')

    >>> # Read ASD directory with custom reader mode
    >>> df, meta = read_spectra('data/asd/', format='asd', reader_mode='auto')

    >>> # Read SPC directory
    >>> df, meta = read_spectra('data/spc_files/', format='spc')

    Notes
    -----
    - All readers return data in standard wide format with wavelengths as columns
    - Wavelengths are automatically sorted in ascending order
    - Data type (reflectance vs absorbance) is auto-detected when possible
    - Missing or invalid spectra are skipped with warnings
    """
    path = Path(path)

    # Auto-detect format if requested
    if format == 'auto':
        format = detect_format(path)

        # If directory, try to infer format from contents
        if format == 'directory':
            format = _detect_directory_format(path)

    # Dispatch to appropriate reader
    if format == 'csv':
        # Handle directory of CSV files vs single CSV file
        if path.is_dir():
            return read_csv_dir(path, **kwargs)
        return read_csv_spectra(path, **kwargs)

    elif format == 'csv_dir':
        return read_csv_dir(path, **kwargs)

    elif format == 'excel':
        return read_excel_spectra(path, **kwargs)

    elif format in ['asd', 'directory']:
        # For ASD, handle both single file and directory
        if path.is_dir():
            return read_asd_dir(path, **kwargs)
        else:
            # Single ASD file - read as directory with one file
            return read_asd_dir(path.parent, **kwargs)

    elif format == 'spc':
        if path.is_dir():
            return read_spc_dir(path, **kwargs)
        else:
            # Single SPC file
            return read_spc_file(path, **kwargs)

    elif format == 'jcamp':
        return read_jcamp_file(path, **kwargs)

    elif format == 'ascii':
        return read_ascii_spectra(path, **kwargs)

    elif format == 'opus':
        return read_opus_file(path, **kwargs)

    elif format == 'perkinelmer':
        return read_perkinelmer_file(path, **kwargs)

    elif format == 'agilent':
        return read_agilent_file(path, **kwargs)

    elif format == 'omnic':
        if path.is_dir():
            return read_omnic_dir(path, **kwargs)
        return read_omnic_file(path, **kwargs)

    else:
        raise ValueError(
            f"Unsupported or unknown format: '{format}'. "
            f"Supported formats: csv, excel, asd, spc, jcamp, ascii, opus, "
            f"perkinelmer, agilent, omnic"
        )


def write_spectra(
    data: pd.DataFrame,
    path: Union[str, Path],
    format: str,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Universal spectral data writer.

    Export spectral data to various formats with format-specific options.

    Supported Export Formats:
    - CSV (wide format)
    - Excel (.xlsx with optional formatting)
    - SPC (GRAMS/Thermo Galactic) - requires spc-io
    - JCAMP-DX - requires jcamp
    - ASCII text (simple two-column format)

    Parameters
    ----------
    data : pd.DataFrame
        Spectral data in wide format (rows=samples, columns=wavelengths)
    path : str or Path
        Output file path
    format : str
        Output format: 'csv', 'excel', 'spc', 'jcamp', 'ascii'
    metadata : dict, optional
        Metadata to include in output (format-dependent)
    **kwargs
        Format-specific options:

        CSV options:
        - float_format : str, default='%.6f' - Number format
        - include_index : bool, default=True - Include sample IDs

        Excel options:
        - sheet_name : str, default='Spectra' - Worksheet name
        - freeze_panes : tuple, default=(1, 1) - Freeze header/index
        - float_format : str, default='0.000000' - Number format

        SPC options:
        - file_type : str, default='TXYVXV' - SPC file type

        JCAMP options:
        - title : str - Dataset title
        - data_type : str - 'INFRARED SPECTRUM', 'RAMAN SPECTRUM', etc.
        - xunits : str, default='NANOMETERS'
        - yunits : str, default='REFLECTANCE'

    Returns
    -------
    None
        File is written to disk

    Raises
    ------
    ValueError
        If format is unsupported or data is invalid
    ImportError
        If required package for format is not installed

    Examples
    --------
    >>> # Export to CSV
    >>> write_spectra(df, 'output.csv', format='csv')

    >>> # Export to Excel with custom formatting
    >>> write_spectra(df, 'output.xlsx', format='excel',
    ...               sheet_name='VIS-NIR', float_format='0.0000')

    >>> # Export single spectrum to JCAMP-DX
    >>> write_spectra(df.iloc[[0]], 'spectrum.jdx', format='jcamp',
    ...               title='Sample A', data_type='INFRARED SPECTRUM')

    Notes
    -----
    - Data must be in wide format with wavelengths as columns
    - Sample IDs are taken from DataFrame index
    - Wavelengths are taken from DataFrame columns
    """
    path = Path(path)

    if format == 'csv':
        write_csv_spectra(data, path, metadata=metadata, **kwargs)

    elif format == 'excel':
        write_excel_spectra(data, path, metadata=metadata, **kwargs)

    elif format == 'spc':
        write_spc_file(data, path, metadata=metadata, **kwargs)

    elif format == 'jcamp':
        write_jcamp_file(data, path, metadata=metadata, **kwargs)

    elif format == 'ascii':
        write_ascii_spectra(data, path, metadata=metadata, **kwargs)

    else:
        raise ValueError(
            f"Unsupported export format: '{format}'. "
            f"Supported formats: csv, excel, spc, jcamp, ascii"
        )


# ============================================================================
# FORMAT-SPECIFIC READERS/WRITERS
# ============================================================================


def _detect_directory_format(directory: Path) -> str:
    """Detect format from directory contents."""
    files = list(directory.iterdir())

    if any(f.suffix.lower() in ASD_EXTENSIONS for f in files):
        return 'asd'
    elif any(f.suffix.lower() == '.spc' for f in files):
        return 'spc'
    elif any(f.suffix.lower() in ['.jdx', '.dx'] for f in files):
        return 'jcamp'
    elif any(f.suffix.lower() in ['.csv'] for f in files):
        return 'csv_dir'
    elif any(f.suffix.lower() in ['.spa', '.spg'] for f in files):
        return 'omnic'
    elif any(f.suffix.lower() in ['.xlsx', '.xls'] for f in files):
        return 'excel'
    else:
        return 'unknown'


def read_excel_spectra(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read spectral data from Excel file.

    Supports same formats as CSV reader:
    - Wide format: first column = id, remaining columns = wavelengths
    - Long format: wavelength, value columns

    Parameters
    ----------
    path : str or Path
        Path to Excel file
    sheet_name : str or int, optional
        Sheet name or index (default: 0 = first sheet)
    **kwargs
        Additional arguments passed to pd.read_excel

    Returns
    -------
    df : pd.DataFrame
        Wide format spectral data
    metadata : dict
        File metadata
    """
    path = Path(path)

    # Read Excel file
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    if df.shape[0] == 0:
        raise ValueError(f"Empty Excel file: {path}")

    # Detect long format
    wl_cols = [c for c in df.columns if str(c).lower() in ["wavelength", "wavelength_nm"]]
    val_cols = [
        c for c in df.columns
        if str(c).lower() in ["value", "intensity", "reflectance", "pct_reflect"]
    ]

    if wl_cols and val_cols:
        # Long format - convert to wide
        wl_col = wl_cols[0]
        val_col = val_cols[0]
        sample_id = path.stem

        df_clean = df[[wl_col, val_col]].dropna()
        wavelengths = df_clean[wl_col].astype(float).values
        values = df_clean[val_col].values
        result = pd.DataFrame([values], columns=wavelengths, index=[sample_id])
        result = result[sorted(result.columns)]
    else:
        # Wide format
        id_col = df.columns[0]
        df = df.set_index(id_col)

        # Parse column names as wavelengths, filtering out non-numeric columns
        wl_cols = {}
        non_wl_cols = []
        for col in df.columns:
            try:
                wl_cols[col] = float(col)
            except ValueError:
                non_wl_cols.append(col)

        if not wl_cols:
            raise ValueError(f"No numeric wavelength columns found. Columns: {list(df.columns)}")

        if non_wl_cols:
            print(f"Note: Ignoring non-wavelength columns: {non_wl_cols}")
            df = df.drop(columns=non_wl_cols)

        df = df.rename(columns=wl_cols)
        df = df[sorted(df.columns)]
        result = df

    # Validate
    if result.shape[1] < 100:
        raise ValueError(f"Expected at least 100 wavelengths, got {result.shape[1]}")

    wls = np.array(result.columns)
    if not np.all(wls[1:] > wls[:-1]):
        raise ValueError("Wavelengths must be strictly increasing")

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(result)
    value_scale = infer_reflectance_scale(result) if data_type == "reflectance" else 1.0

    metadata = {
        'n_spectra': len(result),
        'wavelength_range': (result.columns.min(), result.columns.max()),
        'file_format': 'excel',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'sheet_name': sheet_name,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return result, metadata


def read_combined_excel(filepath, specimen_id_col=None, y_col=None, sheet_name=0, drop_na_y=True):
    """
    Read a combined Excel file containing spectra + targets in one table.

    Uses the same logic as read_combined_csv() but for Excel files.

    Expected format:
    - One row per specimen
    - Specimen ID column (OPTIONAL - will generate if absent)
    - Wavelength columns (numeric headers, FLEXIBLE POSITION)
    - Target y column (FLEXIBLE POSITION - before or after wavelengths)
    - Optional metadata columns (preserved and returned)

    Example formats supported:

    Format A: With ID column
    | specimen_id | 400    | 401    | ... | 2400   | collagen |
    | A-53        | 0.245  | 0.248  | ... | 0.156  | 6.4      |

    Format B: Without ID column (will generate Sample_1, Sample_2, ...)
    | 400    | 401    | ... | 2400   | collagen |
    | 0.245  | 0.248  | ... | 0.156  | 6.4      |
    | 0.312  | 0.315  | ... | 0.201  | 7.9      |

    Format C: ID and target anywhere with metadata
    | specimen_id | site   | depth | 400    | 401    | ... | 2400   | collagen |
    | A-53        | Site1  | 10.5  | 0.245  | 0.248  | ... | 0.156  | 6.4      |

    Parameters
    ----------
    filepath : str or Path
        Path to combined Excel file
    specimen_id_col : str, optional
        Name of specimen ID column. If None, auto-detect. If "__GENERATE__", force generation.
    y_col : str, optional
        Name of target variable column. If None, auto-detect.
    sheet_name : str or int, optional
        Sheet name or index (default: 0 = first sheet)
    drop_na_y : bool, optional
        If True (default), remove rows with missing y values. If False, keep all rows with valid
        spectral data even if y is NaN. Useful when loading data for prediction.

    Returns
    -------
    X : pd.DataFrame
        Spectral data (rows=specimens, cols=wavelengths)
    y : pd.Series
        Target values
    metadata_df : pd.DataFrame or None
        Additional metadata columns (rows=specimens, cols=metadata fields)
        None if no metadata columns present
    metadata : dict
        {
            'specimen_id_col': detected column name or "__GENERATED__",
            'y_col': detected column name,
            'wavelength_cols': list of wavelength column names,
            'metadata_cols': list of metadata column names,
            'n_spectra': number of spectra loaded,
            'wavelength_range': (min, max),
            'generated_ids': True if IDs were auto-generated,
            'file_format': 'combined_excel',
            'sheet_name': sheet name/index used
        }
    """
    filepath = Path(filepath)

    # Step 1: Read Excel file
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"Could not read Excel file {filepath}: {e}")

    if df.shape[0] == 0:
        raise ValueError(f"Empty Excel file: {filepath}")

    # Step 2: Clean column names (strip whitespace)
    df.columns = df.columns.astype(str).str.strip()

    # Step 3: Identify wavelength columns FIRST (position-independent)
    wavelength_cols = identify_wavelength_columns(df)

    if len(wavelength_cols) < 100:
        raise ValueError(
            f"Too few wavelength columns detected ({len(wavelength_cols)}). "
            f"Expected at least 100. Detected columns: {wavelength_cols[:10] if wavelength_cols else 'none'}..."
        )

    # Step 4: Identify specimen ID column (from non-wavelength columns)
    generated_ids = False

    if specimen_id_col is None:
        detected_specimen_id_col = auto_detect_specimen_id_column(df, wavelength_cols)

        if detected_specimen_id_col is None:
            # No ID column detected → generate synthetic IDs
            specimen_ids = pd.Series([f"Sample_{i+1}" for i in range(len(df))],
                                    name="specimen_id")
            generated_ids = True
            specimen_id_col = "__GENERATED__"
        else:
            specimen_id_col = detected_specimen_id_col
            specimen_ids = df[specimen_id_col].astype(str)

    elif specimen_id_col == "__GENERATE__":
        # User explicitly requested generated IDs
        specimen_ids = pd.Series([f"Sample_{i+1}" for i in range(len(df))],
                                name="specimen_id")
        generated_ids = True

    else:
        # User provided specific column name
        if specimen_id_col not in df.columns:
            raise ValueError(f"Specimen ID column '{specimen_id_col}' not found in file")
        specimen_ids = df[specimen_id_col].astype(str)

    # Step 5: Identify y column (from remaining non-wavelength, non-ID columns)
    no_target = (y_col == "__NONE__")

    if no_target:
        y_col = None
    elif y_col is None:
        exclude_cols = wavelength_cols.copy()
        if not generated_ids and specimen_id_col != "__GENERATED__":
            exclude_cols.append(specimen_id_col)

        y_col = auto_detect_y_column(df, exclude_cols)
        if y_col is None:
            no_target = True

    if not no_target:
        if y_col not in df.columns:
            raise ValueError(f"Target y column '{y_col}' not found in file")

    # Step 6: Identify and extract metadata columns
    # Metadata columns = all columns that are NOT wavelengths, NOT specimen ID, NOT target
    all_cols = set(df.columns)
    wavelength_cols_set = set(wavelength_cols)
    used_cols = wavelength_cols_set.copy()
    if not no_target and y_col is not None:
        used_cols.add(y_col)
    if not generated_ids and specimen_id_col != "__GENERATED__":
        used_cols.add(specimen_id_col)

    metadata_cols = sorted(list(all_cols - used_cols))  # Preserve alphabetical order

    # Extract metadata DataFrame (if any metadata columns exist)
    if metadata_cols:
        metadata_df = df[metadata_cols].copy()
        metadata_df.index = specimen_ids
    else:
        metadata_df = None

    # Step 7: Extract spectral data
    X = df[wavelength_cols].copy()
    X.index = specimen_ids

    # Convert spectral data values to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Convert wavelength column names to float and sort
    X.columns = X.columns.astype(float)
    X = X.sort_index(axis=1)  # Sort by wavelength

    if no_target:
        # No target variable mode
        y = None
        has_nan_y = pd.Series(False, index=X.index)
    else:
        # Extract target data
        y = df[y_col].copy()
        y.index = specimen_ids

        # Try to convert target values to numeric, but preserve categorical data
        y_numeric = pd.to_numeric(y, errors='coerce')

        if y_numeric.isna().sum() > len(y) * 0.5:
            has_nan_y = y.isna() | (y == '') | y.isnull()
        else:
            y = y_numeric
            has_nan_y = y.isna()

    # Check for missing values (NaN) and remove affected specimens
    has_nan_X = X.isna().any(axis=1)

    # Determine which rows to remove based on drop_na_y parameter
    if drop_na_y and not no_target:
        has_nan = has_nan_X | has_nan_y
    else:
        has_nan = has_nan_X

    if has_nan.any():
        n_missing = has_nan.sum()
        missing_specimens = X.index[has_nan].tolist()

        print(f"Warning: Found {n_missing} specimen(s) with missing spectral data. Removing them.")
        print(f"  Removed specimens: {missing_specimens[:10]}")  # Show first 10
        if n_missing > 10:
            print(f"  ... and {n_missing - 10} more")

        # Remove rows with missing values
        X = X[~has_nan]
        if y is not None:
            y = y[~has_nan]
        if metadata_df is not None:
            metadata_df = metadata_df[~has_nan]

    # Report on rows kept with missing y values if drop_na_y=False
    if not no_target and not drop_na_y and has_nan_y.any():
        n_missing_y = has_nan_y.sum()
        print(f"Info: Kept {n_missing_y} specimen(s) with missing target values (useful for prediction).")

    # Step 8: Validation
    # Check for duplicate specimen IDs (only if not generated)
    # Use X.index since specimen_ids may be out of sync after NaN removal
    n_duplicates_renamed = 0
    duplicate_rename_mapping = {}
    if not generated_ids and X.index.duplicated().any():
        # Rename duplicates by adding .1, .2, etc. suffix instead of removing them
        new_index, n_duplicates_renamed, duplicate_rename_mapping = _rename_duplicate_ids(X.index)

        print(f"Warning: Found {n_duplicates_renamed} duplicate specimen IDs. "
              f"Auto-renamed with .1, .2, etc. suffix.")

        # Show examples of renamed IDs
        for orig_id, new_ids in list(duplicate_rename_mapping.items())[:3]:
            print(f"  '{orig_id}' -> {new_ids}")

        # Apply renamed index to all DataFrames
        X.index = new_index
        if y is not None:
            y.index = new_index
        if metadata_df is not None:
            metadata_df.index = new_index

    # Check wavelength ordering
    wavelength_values = X.columns.values
    if not all(wavelength_values[i] < wavelength_values[i+1]
              for i in range(len(wavelength_values)-1)):
        print("Warning: Wavelengths were not strictly increasing. Sorted automatically.")

    # Step 9: Detect data type (reflectance vs absorbance) for Excel
    data_type, type_confidence, detection_method = detect_spectral_data_type(X)
    value_scale = infer_reflectance_scale(X) if data_type == "reflectance" else 1.0
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")
    if data_type == "reflectance" and value_scale != 1.0:
        print("  INFO: Detected percent reflectance (0-100). Conversions will scale to 0-1.")

    # Step 10: Compile metadata
    metadata = {
        'specimen_id_col': specimen_id_col,
        'y_col': y_col,
        'wavelength_cols': wavelength_cols,
        'metadata_cols': metadata_cols if metadata_cols else [],
        'n_spectra': len(X),
        'wavelength_range': (X.columns.min(), X.columns.max()),
        'file_format': 'combined_excel',
        'sheet_name': sheet_name,
        'generated_ids': generated_ids,
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'duplicates_renamed': n_duplicates_renamed,
        'duplicate_rename_mapping': duplicate_rename_mapping,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    print(f"Successfully read {len(X)} spectra with {X.shape[1]} wavelengths from Excel file")
    print(f"  Specimen ID column: {specimen_id_col}")
    print(f"  Target column: {y_col}")
    if metadata_cols:
        print(f"  Metadata columns: {', '.join(metadata_cols)}")

    return X, y, metadata_df, metadata


def detect_combined_excel_format(directory_path):
    """
    Detect if directory contains a single combined Excel file.

    A combined Excel file contains all spectra in one table with:
    - Specimen ID column (optional)
    - Wavelength columns (numeric headers)
    - Target y column

    Parameters
    ----------
    directory_path : str or Path
        Path to directory

    Returns
    -------
    tuple : (bool, str or None, str or None)
        (is_combined, filepath, sheet_name) or (False, None, None)
    """
    directory_path = Path(directory_path)

    if not directory_path.exists() or not directory_path.is_dir():
        return False, None, None

    # Get all Excel files
    xlsx_files = list(directory_path.glob("*.xlsx"))
    xls_files = list(directory_path.glob("*.xls"))

    all_files = xlsx_files + xls_files

    # If exactly ONE Excel file, treat as combined format
    if len(all_files) == 1:
        # Return with default sheet (first sheet)
        return True, str(all_files[0]), 0

    return False, None, None


def read_spc_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a single SPC file.

    Parameters
    ----------
    path : str or Path
        Path to SPC file

    Returns
    -------
    df : pd.DataFrame
        Single-row DataFrame with spectrum
    metadata : dict
        SPC file metadata
    """
    try:
        import spc_io
    except ImportError:
        raise ImportError(
            "SPC file support requires spc-io package.\n"
            "Install with: pip install spc-io"
        )

    path = Path(path)

    # Read SPC file
    with open(path, 'rb') as f:
        spc = spc_io.SPC.from_bytes_io(f)

    # Extract wavelengths and intensities
    # SPC files can have multiple sub-files
    if len(spc) == 0:
        raise ValueError(f"No spectral data found in SPC file: {path}")

    # Take first sub-file
    subfile = spc[0]
    wavelengths = subfile.xarray
    intensities = subfile.yarray

    # Read xtype/ytype for x-unit and data type detection
    _SPC_XTYPE_MAP = {
        'XWAVEN': 'cm-1', 'XNMETR': 'nm', 'XUMETR': 'um',
        'XRAMANS': 'cm-1', 'XHERTZ': None, 'XSEC': None,
    }
    _SPC_YTYPE_MAP = {
        'YABSRB': 'absorbance', 'YTRANS': 'transmittance',
        'YREFLEC': 'reflectance', 'YEMISN': None,
    }
    x_unit = 'nm'
    x_unit_confidence = 50.0
    x_unit_method = 'default'
    xtype_attr = getattr(spc, 'xtype', None)
    ytype_attr = getattr(spc, 'ytype', None)
    if xtype_attr is not None:
        xtype_str = str(xtype_attr).split('.')[-1].upper()
        mapped = _SPC_XTYPE_MAP.get(xtype_str)
        if mapped:
            x_unit = mapped
            x_unit_confidence = 95.0
            x_unit_method = 'spc_xtype'

    # Create DataFrame
    df = pd.DataFrame([intensities], columns=wavelengths, index=[path.stem])
    df = df[sorted(df.columns)]

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    # Override data type with SPC ytype if available
    if ytype_attr is not None:
        ytype_str = str(ytype_attr).split('.')[-1].upper()
        mapped_y = _SPC_YTYPE_MAP.get(ytype_str)
        if mapped_y in ('absorbance', 'reflectance'):
            data_type = mapped_y
            type_confidence = 95.0
            detection_method = 'spc_ytype'
            value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'spc',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': x_unit,
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
    }

    return df, metadata


def read_jcamp_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read JCAMP-DX format file.

    Parameters
    ----------
    path : str or Path
        Path to JCAMP file

    Returns
    -------
    df : pd.DataFrame
        Single-row DataFrame with spectrum
    metadata : dict
        JCAMP metadata
    """
    try:
        import jcamp
    except ImportError:
        raise ImportError(
            "JCAMP-DX support requires jcamp package.\n"
            "Install with: pip install jcamp"
        )

    path = Path(path)

    # Read JCAMP file
    jcamp_data = jcamp.jcamp_readfile(str(path))

    # Extract x and y data
    wavelengths = jcamp_data['x']
    intensities = jcamp_data['y']

    # Create DataFrame
    df = pd.DataFrame([intensities], columns=wavelengths, index=[path.stem])
    df = df[sorted(df.columns)]

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    # Map JCAMP xunits to standard x_unit
    _JCAMP_XUNIT_MAP = {
        '1/CM': 'cm-1', 'CM-1': 'cm-1', 'CM^-1': 'cm-1',
        'NANOMETERS': 'nm', 'NM': 'nm',
        'MICROMETERS': 'um', 'UM': 'um',
    }
    raw_xunits = jcamp_data.get('xunits', '')
    xunits_upper = str(raw_xunits).upper().strip() if raw_xunits else ''
    x_unit = _JCAMP_XUNIT_MAP.get(xunits_upper, _heuristic_x_unit(df.columns))
    x_unit_confidence = 90.0 if xunits_upper in _JCAMP_XUNIT_MAP else 50.0
    x_unit_method = 'jcamp_xunits' if xunits_upper in _JCAMP_XUNIT_MAP else 'default'

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'jcamp',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'jcamp_header': {k: v for k, v in jcamp_data.items() if k not in ['x', 'y']},
        'x_unit': x_unit,
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
    }

    return df, metadata


def read_ascii_spectra(
    path: Union[str, Path],
    delimiter: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read generic ASCII text file with spectral data.

    Expected format: two columns (wavelength, intensity)

    Parameters
    ----------
    path : str or Path
        Path to ASCII file
    delimiter : str, optional
        Column delimiter (auto-detected if None)

    Returns
    -------
    df : pd.DataFrame
        Single-row DataFrame with spectrum
    metadata : dict
        File metadata
    """
    path = Path(path)

    # Try multiple delimiters
    if delimiter is None:
        delimiters = [None, '\t', ',', ' ', ';']
    else:
        delimiters = [delimiter]

    df_read = None
    for delim in delimiters:
        try:
            df_read = pd.read_csv(
                path,
                delimiter=delim,
                comment='#',
                skip_blank_lines=True,
                engine='python' if delim is None else 'c',
                **kwargs
            )
            if df_read.shape[1] >= 2:
                break
        except Exception:
            continue

    if df_read is None or df_read.shape[1] < 2:
        raise ValueError(f"Could not parse ASCII file: {path}")

    # Take first two columns as wavelength and intensity
    wavelengths = df_read.iloc[:, 0].values
    intensities = df_read.iloc[:, 1].values

    # Create DataFrame
    df = pd.DataFrame([intensities], columns=wavelengths, index=[path.stem])
    df = df[sorted(df.columns)]

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'ascii',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'value_scale': value_scale,
        'x_unit': 'nm',
        'x_unit_confidence': 50.0,
        'x_unit_detection_method': 'default',
    }

    return df, metadata


def read_opus_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read Bruker OPUS format file.

    Wrapper around spectral_predict.readers.opus_reader.

    Parameters
    ----------
    path : str or Path
        Path to OPUS file

    Returns
    -------
    df : pd.DataFrame
        Spectral data
    metadata : dict
        OPUS metadata
    """
    from spectral_predict.readers.opus_reader import read_opus_file as _read_opus_file

    path = Path(path)

    # Read single OPUS file
    spectrum, file_metadata = _read_opus_file(path)

    # Convert to DataFrame format (single row)
    df = pd.DataFrame([spectrum.values], columns=spectrum.index, index=[path.stem])

    # Detect data type if not already provided
    source_data_type = file_metadata.get('data_type')
    if source_data_type in ['absorbance', 'transmittance']:
        # Map OPUS metadata into supported types for UI/pipelines
        data_type = 'absorbance' if source_data_type == 'absorbance' else 'reflectance'
        type_confidence = 95.0
        detection_method = f"opus_metadata({source_data_type})"
    else:
        data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    # Merge metadata
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('wavenumber_range', (df.columns.min(), df.columns.max())),
        'file_format': 'opus',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'source_data_type': source_data_type,
        'value_scale': value_scale,
        'x_unit': 'cm-1',
        'x_unit_confidence': 99.0,
        'x_unit_detection_method': 'opus_native',
        **file_metadata
    }

    return df, metadata


def read_perkinelmer_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read PerkinElmer .sp format file.

    Wrapper around spectral_predict.readers.perkinelmer_reader.

    Parameters
    ----------
    path : str or Path
        Path to .sp file

    Returns
    -------
    df : pd.DataFrame
        Spectral data
    metadata : dict
        File metadata
    """
    from spectral_predict.readers.perkinelmer_reader import read_sp_file

    path = Path(path)

    # Read single .sp file
    spectrum, file_metadata = read_sp_file(path)

    # Convert to DataFrame format (single row)
    df = pd.DataFrame([spectrum.values], columns=spectrum.index, index=[path.stem])

    # Detect data type if not already provided
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)

    # PerkinElmer .sp files are typically wavenumber (cm⁻¹)
    # Merge metadata
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('x_range', (df.columns.min(), df.columns.max())),
        'file_format': 'perkinelmer',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'x_unit': file_metadata.get('x_unit', 'cm-1'),
        'x_unit_confidence': file_metadata.get('x_unit_confidence', 80.0),
        'x_unit_detection_method': file_metadata.get('x_unit_detection_method', 'perkinelmer_default'),
        **file_metadata
    }

    return df, metadata


def read_agilent_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read Agilent format file (.seq, .dmt, .asp, .bsw).

    Wrapper around spectral_predict.readers.agilent_reader.

    Parameters
    ----------
    path : str or Path
        Path to Agilent file
    **kwargs
        Passed to reader (e.g., extract_mode='total'|'first'|'mean')

    Returns
    -------
    df : pd.DataFrame
        Spectral data
    metadata : dict
        File metadata
    """
    from spectral_predict.readers.agilent_reader import read_agilent_file as _read_agilent_file

    path = Path(path)

    # Read single Agilent file
    spectrum, file_metadata = _read_agilent_file(path, **kwargs)

    # Convert to DataFrame format (single row)
    df = pd.DataFrame([spectrum.values], columns=spectrum.index, index=[path.stem])

    # Detect data type if not already provided
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)

    # Merge metadata
    # Agilent FTIR files are typically wavenumber (cm⁻¹)
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('wavenumber_range', (df.columns.min(), df.columns.max())),
        'file_format': 'agilent',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'x_unit': file_metadata.get('x_unit', 'cm-1'),
        'x_unit_confidence': file_metadata.get('x_unit_confidence', 80.0),
        'x_unit_detection_method': file_metadata.get('x_unit_detection_method', 'agilent_default'),
        **file_metadata
    }

    return df, metadata


def write_csv_spectra(
    data: pd.DataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    float_format: str = '%.6f',
    include_index: bool = True,
    **kwargs
) -> None:
    """Write spectral data to CSV file."""
    path = Path(path)
    data.to_csv(path, float_format=float_format, index=include_index, **kwargs)


def write_excel_spectra(
    data: pd.DataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    sheet_name: str = 'Spectra',
    freeze_panes: Tuple[int, int] = (1, 1),
    float_format: str = '0.000000',
    **kwargs
) -> None:
    """
    Write spectral data to Excel file with formatting.

    Features:
    - Bold headers
    - Auto-adjusted column widths
    - Number formatting for spectral values
    - Frozen header row and ID column

    Parameters
    ----------
    data : pd.DataFrame
        Spectral data (rows=samples, columns=wavelengths)
    path : str or Path
        Output path (.xlsx)
    metadata : dict, optional
        Metadata (not used currently)
    sheet_name : str, default='Spectra'
        Worksheet name
    freeze_panes : tuple, default=(1, 1)
        Cell position to freeze (row, col)
    float_format : str, default='0.000000'
        Number format string for spectral values
    **kwargs
        Additional arguments passed to to_excel
    """
    path = Path(path)

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name=sheet_name, **kwargs)

        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Create format for bold headers
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # Create format for numbers
        number_format = workbook.add_format({'num_format': float_format})

        # Format header row (row 0)
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num + 1, value, header_format)

        # Format index column header
        worksheet.write(0, 0, data.index.name or 'ID', header_format)

        # Apply number format to data cells
        for row_num in range(len(data)):
            for col_num in range(len(data.columns)):
                worksheet.write(row_num + 1, col_num + 1, data.iloc[row_num, col_num], number_format)

        # Auto-adjust column widths
        # ID column
        max_id_len = max(len(str(idx)) for idx in data.index)
        worksheet.set_column(0, 0, max(max_id_len + 2, 10))

        # Wavelength columns (assuming they're numeric)
        # Set a reasonable width for wavelength columns
        worksheet.set_column(1, len(data.columns), 12)

        # Freeze panes
        if freeze_panes:
            worksheet.freeze_panes(freeze_panes[0], freeze_panes[1])

    print(f"Wrote {len(data)} spectra to {path}")


def write_spc_file(
    data: pd.DataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Write spectral data to SPC format.

    Note: Only writes single spectrum (first row if multiple rows provided)

    Parameters
    ----------
    data : pd.DataFrame
        Spectral data (rows=samples, columns=wavelengths)
    path : str or Path
        Output file path
    metadata : dict, optional
        Metadata (not currently used)
    **kwargs
        Additional arguments (not currently used)
    """
    try:
        import spc_io
        import spc_io.high_level as spc_high
    except ImportError:
        raise ImportError(
            "SPC export requires spc-io package.\n"
            "Install with: pip install spc-io"
        )

    import numpy as np

    path = Path(path)

    # Take first spectrum if multiple
    if len(data) > 1:
        print(f"Warning: SPC format supports single spectrum. Writing first row only.")
        data = data.iloc[[0]]

    wavelengths = data.columns.values.astype(float)
    intensities = data.iloc[0].values.astype(float)

    # Create SPC object using high-level API
    # Assume evenly spaced wavelengths for simplicity
    first_wl = float(wavelengths[0])
    last_wl = float(wavelengths[-1])
    n_points = len(wavelengths)

    spc = spc_high.SPC(xarray=spc_high.EvenAxis(first_wl, last_wl, n_points))
    spc.add_subfile(yarray=intensities)

    # Write to file
    with open(path, 'wb') as f:
        f.write(spc.to_spc_raw().to_bytes())


def _build_jcamp_dx_string(jcamp_dict: Dict[str, Any], linewidth: int = 75) -> str:
    """Build a JCAMP-DX formatted string from a data dictionary.

    Vendored from jcamp 1.3.0's jcamp_write() so we can keep the pyproject
    pin at jcamp<1.3 — the 1.3.0 PyPI release declares stdlib modules
    (re, pdb, datetime) as runtime deps in its setup.py, making the
    package un-pip-installable. jcamp 1.0–1.2.x are read-only PyPI
    releases (no write functions). Vendoring this ~60-line helper avoids
    the upstream packaging bug while keeping us on the read-functional
    1.2.x line.
    """
    if 'x' not in jcamp_dict:
        raise ValueError('input dictionary must contain "x"')
    if 'y' not in jcamp_dict:
        raise ValueError('input dictionary must contain "y"')

    x = np.asarray(jcamp_dict['x'])
    y = np.asarray(jcamp_dict['y'])

    parts = ['##JCAMP-DX=5.01\n']
    for key, value in jcamp_dict.items():
        if key in ('x', 'y', 'xydata', 'end'):
            continue
        parts.append(f"##{key.upper()}={value}\n")

    if 'firstx' not in jcamp_dict:
        parts.append(f"##FIRSTX={x[0]:.6f}\n")
    if 'lastx' not in jcamp_dict:
        parts.append(f"##LASTX={x[-1]:.6f}\n")
    if 'maxx' not in jcamp_dict:
        parts.append(f"##MAXX={np.amax(x):.6f}\n")
    if 'minx' not in jcamp_dict:
        parts.append(f"##MINX={np.amin(x):.6f}\n")
    if 'firsty' not in jcamp_dict:
        parts.append(f"##FIRSTY={y[0]:.4f}\n")
    if 'lasty' not in jcamp_dict:
        parts.append(f"##LASTY={y[-1]:.4f}\n")
    if 'maxy' not in jcamp_dict:
        parts.append(f"##MAXY={np.amax(y):.4f}\n")
    if 'miny' not in jcamp_dict:
        parts.append(f"##MINY={np.amin(y):.4f}\n")

    npts = jcamp_dict.get('npts', len(x))
    parts.append(f"##NPOINTS={npts}\n")
    parts.append(f"##XFACTOR={jcamp_dict.get('xfactor', 1)}\n")
    yfactor = jcamp_dict.get('yfactor', 1)
    parts.append(f"##YFACTOR={yfactor}\n")
    parts.append("##XYDATA=(X++(Y..Y))\n")

    line = f"{x[0]:.6f} "
    for j in range(npts):
        if np.isnan(y[j]):
            line += '? '
        else:
            line += f"{y[j] / yfactor:.4f} "
        if len(line) >= linewidth or j == npts - 1:
            parts.append(line + '\n')
            if j < npts - 1:
                line = f"{x[j + 1]:.6f} "

    parts.append('##END=\n')
    return ''.join(parts)


def write_jcamp_file(
    data: pd.DataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    data_type: str = 'INFRARED SPECTRUM',
    xunits: str = 'NANOMETERS',
    yunits: str = 'REFLECTANCE',
    **kwargs
) -> None:
    """
    Write spectral data to JCAMP-DX format.

    Note: Only writes single spectrum (first row if multiple)
    """
    path = Path(path)

    # Take first spectrum if multiple
    if len(data) > 1:
        print(f"Warning: JCAMP format supports single spectrum. Writing first row only.")
        data = data.iloc[[0]]

    wavelengths = data.columns.values
    intensities = data.iloc[0].values

    # Build JCAMP dictionary
    jcamp_dict = {
        'title': title or path.stem,
        'data type': data_type,
        'xunits': xunits,
        'yunits': yunits,
        'x': wavelengths,
        'y': intensities
    }

    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            if key not in jcamp_dict:
                jcamp_dict[key] = value

    # Use vendored write helper rather than jcamp.jcamp_writefile — the only
    # PyPI release with that function (1.3.0) is broken upstream (declares
    # stdlib re/pdb/datetime as deps).
    path.write_text(_build_jcamp_dx_string(jcamp_dict))


def write_ascii_spectra(
    data: pd.DataFrame,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    delimiter: str = '\t',
    include_header: bool = True,
    **kwargs
) -> None:
    """
    Write spectral data to simple ASCII text file.

    Format: two columns (wavelength, intensity)
    Note: Only writes single spectrum (first row if multiple)
    """
    path = Path(path)

    # Take first spectrum if multiple
    if len(data) > 1:
        print(f"Warning: ASCII format supports single spectrum. Writing first row only.")
        data = data.iloc[[0]]

    wavelengths = data.columns.values
    intensities = data.iloc[0].values

    # Create output DataFrame
    output = pd.DataFrame({
        'Wavelength': wavelengths,
        'Intensity': intensities
    })

    output.to_csv(
        path,
        sep=delimiter,
        index=False,
        header=include_header,
        **kwargs
    )


# ============================================================================
# VENDOR-SPECIFIC DIRECTORY READERS
# ============================================================================


def read_opus_dir(directory: Union[str, Path], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read all Bruker OPUS files from a directory.

    Wrapper around spectral_predict.readers.opus_reader.read_opus_dir.

    Parameters
    ----------
    directory : str or Path
        Directory containing OPUS files (.0, .1, .2, etc.)
    **kwargs
        Additional arguments passed to reader

    Returns
    -------
    df : pd.DataFrame
        Spectral data (rows=samples, columns=wavenumbers)
    metadata : dict
        Combined metadata

    Examples
    --------
    >>> df, meta = read_opus_dir('data/bruker_files/')
    >>> print(f"Loaded {len(df)} OPUS spectra")
    """
    from spectral_predict.readers.opus_reader import read_opus_dir as _read_opus_dir

    df, metadata = _read_opus_dir(directory, **kwargs)

    # Use OPUS metadata when available, but map into supported types
    source_data_type = metadata.get('dominant_data_type') or metadata.get('data_type')
    if source_data_type in ['absorbance', 'transmittance']:
        data_type = 'absorbance' if source_data_type == 'absorbance' else 'reflectance'
        type_confidence = 95.0
        detection_method = f"opus_metadata_dir({source_data_type})"
    else:
        data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    metadata.update({
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'source_data_type': source_data_type,
        'value_scale': value_scale
    })

    return df, metadata


def read_sp_dir(directory: Union[str, Path], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read all PerkinElmer .sp files from a directory.

    Wrapper around spectral_predict.readers.perkinelmer_reader.read_sp_dir.

    Parameters
    ----------
    directory : str or Path
        Directory containing .sp files
    **kwargs
        Additional arguments passed to reader

    Returns
    -------
    df : pd.DataFrame
        Spectral data (rows=samples, columns=wavelengths/wavenumbers)
    metadata : dict
        Combined metadata

    Examples
    --------
    >>> df, meta = read_sp_dir('data/perkinelmer_files/')
    >>> print(f"Loaded {len(df)} PerkinElmer spectra")
    """
    from spectral_predict.readers.perkinelmer_reader import read_sp_dir as _read_sp_dir

    return _read_sp_dir(directory, **kwargs)


def read_omnic_file(
    path: Union[str, Path],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read Thermo Omnic .spa or .spg format file.

    Wrapper around spectral_predict.readers.omnic_reader.

    Parameters
    ----------
    path : str or Path
        Path to .spa or .spg file

    Returns
    -------
    df : pd.DataFrame
        Spectral data (rows=samples, columns=wavenumbers)
    metadata : dict
        Omnic metadata including data_type, source_data_type, x_unit
    """
    from spectral_predict.readers.omnic_reader import read_spa_file, read_spg_file

    path = Path(path)

    # Dispatch based on extension
    if path.suffix.lower() == '.spg':
        df, file_metadata = read_spg_file(path)
    else:
        spectrum, file_metadata = read_spa_file(path)
        df = pd.DataFrame([spectrum.values], columns=spectrum.index, index=[path.stem])

    # Use Omnic metadata when available, but map into supported types
    source_data_type = file_metadata.get('source_data_type')
    omnic_data_type = file_metadata.get('data_type')

    if omnic_data_type in ['absorbance', 'reflectance']:
        data_type = omnic_data_type
        type_confidence = 95.0
        detection_method = f"omnic_metadata({source_data_type})"
    else:
        data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    # Merge metadata
    x_unit = file_metadata.get('x_unit', 'cm-1')
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': file_metadata.get('x_range', (df.columns.min(), df.columns.max())),
        'file_format': 'omnic',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'source_data_type': source_data_type,
        'value_scale': value_scale,
        'x_unit': x_unit,
        'x_unit_confidence': 95.0,
        'x_unit_detection_method': 'omnic_metadata',
        **file_metadata
    }

    return df, metadata


def read_omnic_dir(directory: Union[str, Path], **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read all Thermo Omnic .spa/.spg files from a directory.

    Wrapper around spectral_predict.readers.omnic_reader.read_omnic_dir.

    Parameters
    ----------
    directory : str or Path
        Directory containing .spa and/or .spg files
    **kwargs
        Additional arguments passed to reader

    Returns
    -------
    df : pd.DataFrame
        Spectral data (rows=samples, columns=wavenumbers)
    metadata : dict
        Combined metadata
    """
    from spectral_predict.readers.omnic_reader import read_omnic_dir as _read_omnic_dir

    df, metadata = _read_omnic_dir(directory, **kwargs)

    # Use Omnic metadata when available, but map into supported types
    source_data_type = metadata.get('dominant_data_type') or metadata.get('data_type')
    if source_data_type in ['absorbance', 'transmittance']:
        data_type = 'absorbance' if source_data_type == 'absorbance' else 'reflectance'
        type_confidence = 95.0
        detection_method = f"omnic_metadata_dir({source_data_type})"
    else:
        data_type, type_confidence, detection_method = detect_spectral_data_type(df)
    value_scale = infer_reflectance_scale(df) if data_type == "reflectance" else 1.0

    metadata.update({
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'source_data_type': source_data_type,
        'value_scale': value_scale
    })

    return df, metadata


def read_agilent_dir(
    directory: Union[str, Path],
    extensions: Optional[list] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read all Agilent files from a directory.

    Wrapper around spectral_predict.readers.agilent_reader.read_agilent_dir.

    Parameters
    ----------
    directory : str or Path
        Directory containing Agilent files
    extensions : list of str, optional
        File extensions to search for (default: ['seq', 'dmt', 'asp', 'bsw'])
    **kwargs
        Additional arguments passed to reader (e.g., extract_mode)

    Returns
    -------
    df : pd.DataFrame
        Spectral data (rows=samples, columns=wavenumbers)
    metadata : dict
        Combined metadata

    Examples
    --------
    >>> df, meta = read_agilent_dir('data/agilent_files/')
    >>> print(f"Loaded {len(df)} Agilent spectra")

    >>> # Read only .seq files with mean extraction
    >>> df, meta = read_agilent_dir('data/', extensions=['seq'], extract_mode='mean')
    """
    from spectral_predict.readers.agilent_reader import read_agilent_dir as _read_agilent_dir

    return _read_agilent_dir(directory, extensions=extensions, **kwargs)
