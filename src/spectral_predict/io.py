"""I/O functions for reading spectral data and reference files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union


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
        id_col = df.columns[0]
        df = df.set_index(id_col)

        # Parse column names as wavelengths
        try:
            wl_cols = {col: float(col) for col in df.columns}
        except ValueError as e:
            raise ValueError(f"Could not parse all column names as wavelengths: {e}")

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
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Compile metadata
    metadata = {
        'n_spectra': len(result),
        'wavelength_range': (result.columns.min(), result.columns.max()),
        'file_format': 'csv',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
    }

    return result, metadata


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

    # Remove common extensions
    for ext in [".asd", ".sig", ".csv", ".txt", ".spc"]:
        if filename.lower().endswith(ext):
            filename = filename[: -len(ext)]
            break

    # Remove spaces and convert to lowercase
    filename = filename.replace(" ", "").lower()

    return filename


def align_xy(X, ref, id_column, target, return_alignment_info=False):
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

    print(f"DEBUG: Initial alignment - X has {len(X)} samples, ref has {len(ref)} samples")
    print(f"DEBUG: Found {len(common_ids)} common IDs (exact match)")

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
        id_mapping = {}
        for norm_id in common_norm_ids:
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

        print(f"DEBUG: common_ids has {len(common_ids)} elements")
        print(f"DEBUG: common_ids type: {type(common_ids)}")
        print(f"DEBUG: First 5 common_ids: {list(common_ids[:5]) if len(common_ids) > 0 else []}")

        # Check for duplicates in indices
        if len(X.index) != len(X.index.unique()):
            print(f"WARNING: X has duplicate indices! Total: {len(X.index)}, Unique: {len(X.index.unique())}")
        if len(ref.index) != len(ref.index.unique()):
            print(f"WARNING: ref has duplicate indices! Total: {len(ref.index)}, Unique: {len(ref.index.unique())}")

        X_aligned = X.loc[common_ids]
        y = ref.loc[common_ids, target]

        # Track matched SPECTRAL IDs (for exact matching, these are just common_ids)
        matched_spectral_ids = list(common_ids)

        print(f"DEBUG: After subsetting - X_aligned: {len(X_aligned)}, y: {len(y)}")

    # Track truly unmatched samples BEFORE NaN filtering
    # (so NaN-dropped samples aren't counted as "unmatched")
    # Use matched_spectral_ids which contains the original spectral file IDs
    matched_before_nan_filter = matched_spectral_ids

    # Drop any NaN targets
    valid_mask = ~y.isna()
    print(f"DEBUG: Before NaN filtering - X_aligned: {len(X_aligned)}, y: {len(y)}")
    n_nan_dropped = 0
    if not valid_mask.all():
        n_nan_dropped = (~valid_mask).sum()
        print(f"Warning: Dropping {n_nan_dropped} samples with missing target values")
        X_aligned = X_aligned[valid_mask]
        y = y[valid_mask]
        print(f"DEBUG: After NaN filtering - X_aligned: {len(X_aligned)}, y: {len(y)}")

    if len(y) == 0:
        raise ValueError("No valid samples after alignment and NaN removal")

    # SAFETY CHECK: Ensure perfect alignment before returning
    print(f"DEBUG: Final check before return - X_aligned: {len(X_aligned)}, y: {len(y)}")
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
        unmatched_reference = sorted([str(x) for x in (original_ref_ids - set(matched_before_nan_filter))])

        alignment_info = {
            'matched_ids': matched_ids,
            'unmatched_spectra': unmatched_spectra,
            'unmatched_reference': unmatched_reference,
            'n_nan_dropped': n_nan_dropped,
            'used_fuzzy_matching': used_fuzzy_matching
        }

        return X_aligned, y, alignment_info

    return X_aligned, y


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

    # Find ASD files
    asd_files = list(asd_dir.glob("*.sig")) + list(asd_dir.glob("*.asd"))

    if len(asd_files) == 0:
        raise ValueError(f"No .sig or .asd files found in {asd_dir}")

    print(f"Found {len(asd_files)} ASD files")

    # Read each file
    spectra = {}
    duplicate_stems = []
    for asd_file in sorted(asd_files):
        stem = asd_file.stem

        # Check for duplicate filenames (without extension)
        if stem in spectra:
            duplicate_stems.append(stem)
            print(f"⚠️ WARNING: Duplicate filename '{stem}' - later file will overwrite earlier one")

        try:
            spectrum = _read_single_asd_ascii(asd_file, reader_mode)
            spectra[stem] = spectrum
        except UnicodeDecodeError:
            # Binary ASD file detected - try to read with SpecDAL
            spectrum = _handle_binary_asd(asd_file, reader_mode)
            if spectrum is not None:
                spectra[stem] = spectrum
        except Exception as e:
            print(f"Warning: Could not read {asd_file.name}: {e}")

    if duplicate_stems:
        print(f"\n⚠️ Found {len(duplicate_stems)} duplicate ASD filenames (ignoring extensions)")
        print(f"Duplicates: {duplicate_stems[:10]}")
        if len(duplicate_stems) > 10:
            print(f"... and {len(duplicate_stems) - 10} more")
        print("Keeping LAST occurrence of each duplicate.\n")

    if len(spectra) == 0:
        raise ValueError("No valid spectra could be read")

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
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'asd',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
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


def _handle_binary_asd(asd_file, reader_mode):
    """
    Handle binary ASD files using SpecDAL.

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
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'spc',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
    }

    print(f"Successfully read {len(df)} SPC spectra with {df.shape[1]} wavelengths")

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
    1. Column named 'specimen_id', 'sample_id', 'id', 'sample', 'specimen', etc.
    2. Column with all/mostly unique values (>80% unique)
    3. First non-wavelength column with object/string dtype
    4. Check if all remaining columns are numeric/y-like → No ID column, return None

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

    # If only one candidate column, check if it looks like y data
    if len(candidate_cols) == 1:
        col = candidate_cols[0]
        # If it looks like a target variable (numeric, not unique), assume no ID column
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            if n_total > 0 and n_unique / n_total < 0.8:  # Not very unique → probably y, not ID
                return None

    # Priority 1: Check for common ID names (case-insensitive)
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
            return matches[0]

    # Priority 2: Find column with unique/mostly unique values
    # Specimen IDs should be unique identifiers
    for col in candidate_cols:
        n_unique = df[col].nunique()
        n_total = len(df[col].dropna())

        if n_total > 0:
            uniqueness_ratio = n_unique / n_total

            # If >80% unique, likely an ID column
            if uniqueness_ratio > 0.8:
                return col

    # Priority 3: Find non-numeric dtype column
    # IDs often contain letters/special characters
    for col in candidate_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            return col

    # Priority 4: Check if all remaining columns are numeric (likely all y-like)
    # If so, assume no ID column present
    all_numeric = all(pd.api.types.is_numeric_dtype(df[col])
                     for col in candidate_cols)

    if all_numeric and len(candidate_cols) <= 3:
        # Likely format: wavelengths + 1-3 y columns, no ID
        return None

    # Priority 5: Fallback to first candidate column
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
        raise ValueError("No remaining columns found for target y variable")

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


def read_combined_csv(filepath, specimen_id_col=None, y_col=None):
    """
    Read a combined CSV/TXT file containing spectra + targets in one table.

    Expected format:
    - One row per specimen
    - Specimen ID column (OPTIONAL - will generate if absent)
    - Wavelength columns (numeric headers, possibly quoted, FLEXIBLE POSITION)
    - Target y column (FLEXIBLE POSITION - before or after wavelengths)

    Example formats supported:

    Format A: With ID column
    specimen_id, "400", "401", ..., "2400", collagen
    A-53, 0.245, 0.248, ..., 0.156, 6.4

    Format B: Without ID column (will generate Sample_1, Sample_2, ...)
    "400", "401", ..., "2400", collagen
    0.245, 0.248, ..., 0.156, 6.4
    0.312, 0.315, ..., 0.201, 7.9

    Format C: ID anywhere
    collagen, specimen_id, "400", "401", ..., "2400"
    6.4, A-53, 0.245, 0.248, ..., 0.156

    Parameters
    ----------
    filepath : str or Path
        Path to combined CSV/TXT file
    specimen_id_col : str, optional
        Name of specimen ID column. If None, auto-detect. If "__GENERATE__", force generation.
    y_col : str, optional
        Name of target variable column. If None, auto-detect.

    Returns
    -------
    X : pd.DataFrame
        Spectral data (rows=specimens, cols=wavelengths)
    y : pd.Series
        Target values
    metadata : dict
        {
            'specimen_id_col': detected column name or "__GENERATED__",
            'y_col': detected column name,
            'wavelength_cols': list of wavelength column names,
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
    if y_col is None:
        exclude_cols = wavelength_cols.copy()
        if not generated_ids and specimen_id_col != "__GENERATED__":
            exclude_cols.append(specimen_id_col)

        y_col = auto_detect_y_column(df, exclude_cols)

    if y_col not in df.columns:
        raise ValueError(f"Target y column '{y_col}' not found in file")

    # Step 6: Extract data
    # Extract spectral data
    X = df[wavelength_cols].copy()
    X.index = specimen_ids

    # Convert spectral data values to numeric (handle any string values from CSV)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Convert wavelength column names to float and sort
    X.columns = X.columns.astype(float)
    X = X.sort_index(axis=1)  # Sort by wavelength

    # Extract target data
    y = df[y_col].copy()
    y.index = specimen_ids

    # Convert target values to numeric
    y = pd.to_numeric(y, errors='coerce')

    # Check for missing values (NaN) and remove affected specimens
    has_nan_X = X.isna().any(axis=1)
    has_nan_y = y.isna()
    has_nan = has_nan_X | has_nan_y

    if has_nan.any():
        n_missing = has_nan.sum()
        missing_specimens = X.index[has_nan].tolist()

        print(f"Warning: Found {n_missing} specimen(s) with missing values. Removing them.")
        print(f"  Removed specimens: {missing_specimens[:10]}")  # Show first 10
        if n_missing > 10:
            print(f"  ... and {n_missing - 10} more")

        # Remove rows with missing values
        X = X[~has_nan]
        y = y[~has_nan]

    # Step 7: Validation
    # Check for duplicate specimen IDs (only if not generated)
    if not generated_ids and specimen_ids.duplicated().any():
        n_duplicates = specimen_ids.duplicated().sum()
        duplicates = specimen_ids[specimen_ids.duplicated()].unique()[:5]
        print(f"Warning: Found {n_duplicates} duplicate specimen IDs. "
              f"Keeping first occurrence. Examples: {list(duplicates)}")

        # Keep first occurrence of each duplicate
        keep_mask = ~specimen_ids.duplicated(keep='first')
        X = X[keep_mask]
        y = y[keep_mask]

    # Check wavelength ordering
    wavelength_values = X.columns.values
    if not all(wavelength_values[i] < wavelength_values[i+1]
              for i in range(len(wavelength_values)-1)):
        print("Warning: Wavelengths were not strictly increasing. Sorted automatically.")

    # Step 8: Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(X)
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Step 9: Compile metadata
    metadata = {
        'specimen_id_col': specimen_id_col,
        'y_col': y_col,
        'wavelength_cols': wavelength_cols,
        'n_spectra': len(X),
        'wavelength_range': (X.columns.min(), X.columns.max()),
        'file_format': 'combined',
        'generated_ids': generated_ids,
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
    }

    return X, y, metadata


def read_jcamp_file(path):
    """
    Read a single JCAMP-DX file (.jdx, .dx).

    JCAMP-DX is a text-based spectral data format with embedded metadata.

    Parameters
    ----------
    path : str or Path
        Path to JCAMP-DX file

    Returns
    -------
    tuple
        (spectrum, metadata) where:
        - spectrum: pd.Series - Spectrum with wavelengths as index
        - metadata: dict - Header metadata from JCAMP file
    """
    try:
        import jcamp
    except ImportError:
        raise ValueError(
            "JCAMP-DX file support requires the jcamp library.\n"
            "Install it with: pip install jcamp"
        )

    path = Path(path)

    # Read JCAMP file
    try:
        jcamp_dict = jcamp.jcamp_reader(str(path))
    except Exception as e:
        raise ValueError(f"Failed to read JCAMP file {path.name}: {e}")

    # Extract spectral data
    x = jcamp_dict.get('x', None)
    y = jcamp_dict.get('y', None)

    if x is None or y is None:
        raise ValueError(f"No spectral data found in JCAMP file: {path.name}")

    # Convert to pandas Series
    spectrum = pd.Series(y, index=x)

    # Sort by wavelength/wavenumber
    spectrum = spectrum.sort_index()

    # Remove duplicates (keep first)
    spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

    # Extract metadata
    metadata = {
        'title': jcamp_dict.get('title', path.stem),
        'xunits': jcamp_dict.get('xunits', 'unknown'),
        'yunits': jcamp_dict.get('yunits', 'unknown'),
        'npoints': jcamp_dict.get('npoints', len(x)),
        'firstx': jcamp_dict.get('firstx', None),
        'lastx': jcamp_dict.get('lastx', None),
        'xfactor': jcamp_dict.get('xfactor', 1.0),
        'yfactor': jcamp_dict.get('yfactor', 1.0),
        'longdate': jcamp_dict.get('longdate', None),
        'file_format': 'jcamp-dx',
        'filename': path.name
    }

    # Add any other fields from JCAMP header
    for key, value in jcamp_dict.items():
        if key not in ['x', 'y', 'title', 'xunits', 'yunits', 'npoints',
                       'firstx', 'lastx', 'xfactor', 'yfactor', 'longdate',
                       'children', 'filename']:
            metadata[key] = value

    return spectrum, metadata


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

    # Find JCAMP files
    jcamp_files = list(jcamp_dir.glob("*.jdx")) + list(jcamp_dir.glob("*.dx")) + list(jcamp_dir.glob("*.JDX")) + list(jcamp_dir.glob("*.DX"))

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
            spectrum, metadata = read_jcamp_file(jcamp_file)
            spectra[stem] = spectrum
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
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Get x-axis units from first file
    first_file_meta = next(iter(file_metadata.values()))
    xunits = first_file_meta.get('xunits', 'unknown')

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'jcamp-dx',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'xunits': xunits,
        'file_metadata': file_metadata  # Store individual file metadata
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
        lines.append("##XYDATA=(X++(Y..Y))")
        for i in range(len(x)):
            lines.append(f"{x[i]:.6f} {y[i]:.6e}")

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
        'detection_method': detection_method
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
    # Find ASCII files
    ascii_files = (list(directory.glob("*.dpt")) +
                   list(directory.glob("*.dat")) +
                   list(directory.glob("*.asc")) +
                   list(directory.glob("*.DPT")) +
                   list(directory.glob("*.DAT")) +
                   list(directory.glob("*.ASC")))

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
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'ascii',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
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

    # Criterion 1: Absolute bounds check (weight: 40%)
    if max_val > 1.5:
        # Definitely absorbance - reflectance can't exceed 1.0
        absorbance_score += 40
        detection_methods.append("bounds_check(max>1.5)")
    elif max_val <= 1.0 and min_val >= 0.0:
        # All values in [0, 1] - likely reflectance
        if mean_val > 0.3:
            # High mean in [0,1] range strongly suggests reflectance
            reflectance_score += 40
            detection_methods.append("bounds_check(0-1_range)")
        else:
            # Low mean could be dark sample reflectance or low absorbance
            reflectance_score += 25
            absorbance_score += 15
            detection_methods.append("bounds_check(0-1_low_mean)")
    elif max_val > 1.0 and max_val <= 1.5:
        # Ambiguous range - could be reflectance with errors or low absorbance
        absorbance_score += 20
        reflectance_score += 15
        detection_methods.append("bounds_check(ambiguous_1.0-1.5)")
    else:
        # Negative values or very low values
        if min_val < -0.5:
            # Significantly negative suggests absorbance (or errors)
            absorbance_score += 35
            detection_methods.append("bounds_check(negative_values)")
        else:
            absorbance_score += 10
            detection_methods.append("bounds_check(near_zero)")

    # Criterion 2: Mean value analysis (weight: 30%)
    if 0.3 <= mean_val <= 0.9:
        # Typical reflectance range
        reflectance_score += 30
        detection_methods.append("mean_check(reflectance_range)")
    elif mean_val > 1.0:
        # High mean suggests absorbance
        absorbance_score += 30
        detection_methods.append("mean_check(absorbance_range)")
    elif mean_val < 0.3 and max_val <= 1.0:
        # Low mean in bounded range - dark reflectance
        reflectance_score += 20
        detection_methods.append("mean_check(dark_reflectance)")
    else:
        # Ambiguous mean
        detection_methods.append("mean_check(ambiguous)")

    # Criterion 3: Peak direction analysis (weight: 30%)
    # Analyze first spectrum for peak/valley characteristics
    if len(data) > 0:
        first_spectrum = data[0, :]
        first_spectrum = first_spectrum[~np.isnan(first_spectrum)]

        if len(first_spectrum) > 10:
            # Find local maxima and minima
            from scipy.signal import find_peaks

            # Find peaks (high points)
            peaks, _ = find_peaks(first_spectrum, prominence=0.01 * (max_val - min_val))
            # Find valleys (low points) by inverting
            valleys, _ = find_peaks(-first_spectrum, prominence=0.01 * (max_val - min_val))

            # Calculate prominence of peaks vs valleys
            if len(peaks) > 0:
                peak_heights = first_spectrum[peaks]
                peak_prominence = np.mean(peak_heights - mean_val)
            else:
                peak_prominence = 0

            if len(valleys) > 0:
                valley_depths = first_spectrum[valleys]
                valley_prominence = np.mean(mean_val - valley_depths)
            else:
                valley_prominence = 0

            # In reflectance spectra, absorption features appear as valleys
            # In absorbance spectra, absorption features appear as peaks
            if valley_prominence > peak_prominence * 1.2:
                # Valleys more prominent - suggests reflectance
                reflectance_score += 25
                detection_methods.append("peak_analysis(valleys_prominent)")
            elif peak_prominence > valley_prominence * 1.2:
                # Peaks more prominent - suggests absorbance
                absorbance_score += 25
                detection_methods.append("peak_analysis(peaks_prominent)")
            else:
                # Ambiguous peak structure
                detection_methods.append("peak_analysis(ambiguous)")

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
        '.spc': 'spc',
        '.jdx': 'jcamp',
        '.dx': 'jcamp',
        '.jcm': 'jcamp',
        '.txt': 'ascii',
        '.dat': 'ascii',
        '.dpt': 'ascii',
        '.asc': 'ascii',
        '.sp': 'perkinelmer',
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
        return read_csv_spectra(path, **kwargs)

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

    else:
        raise ValueError(
            f"Unsupported or unknown format: '{format}'. "
            f"Supported formats: csv, excel, asd, spc, jcamp, ascii, opus, "
            f"perkinelmer, agilent"
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

    if any(f.suffix.lower() in ['.asd', '.sig'] for f in files):
        return 'asd'
    elif any(f.suffix.lower() == '.spc' for f in files):
        return 'spc'
    elif any(f.suffix.lower() in ['.jdx', '.dx'] for f in files):
        return 'jcamp'
    elif any(f.suffix.lower() in ['.csv'] for f in files):
        return 'csv'
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

        try:
            wl_cols = {col: float(col) for col in df.columns}
        except ValueError as e:
            raise ValueError(f"Could not parse all column names as wavelengths: {e}")

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

    metadata = {
        'n_spectra': len(result),
        'wavelength_range': (result.columns.min(), result.columns.max()),
        'file_format': 'excel',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'sheet_name': sheet_name
    }

    return result, metadata


def read_combined_excel(filepath, specimen_id_col=None, y_col=None, sheet_name=0):
    """
    Read a combined Excel file containing spectra + targets in one table.

    Uses the same logic as read_combined_csv() but for Excel files.

    Expected format:
    - One row per specimen
    - Specimen ID column (OPTIONAL - will generate if absent)
    - Wavelength columns (numeric headers, FLEXIBLE POSITION)
    - Target y column (FLEXIBLE POSITION - before or after wavelengths)

    Example formats supported:

    Format A: With ID column
    | specimen_id | 400    | 401    | ... | 2400   | collagen |
    | A-53        | 0.245  | 0.248  | ... | 0.156  | 6.4      |

    Format B: Without ID column (will generate Sample_1, Sample_2, ...)
    | 400    | 401    | ... | 2400   | collagen |
    | 0.245  | 0.248  | ... | 0.156  | 6.4      |
    | 0.312  | 0.315  | ... | 0.201  | 7.9      |

    Format C: ID and target anywhere
    | collagen | specimen_id | 400    | 401    | ... | 2400   |
    | 6.4      | A-53        | 0.245  | 0.248  | ... | 0.156  |

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

    Returns
    -------
    X : pd.DataFrame
        Spectral data (rows=specimens, cols=wavelengths)
    y : pd.Series
        Target values
    metadata : dict
        {
            'specimen_id_col': detected column name or "__GENERATED__",
            'y_col': detected column name,
            'wavelength_cols': list of wavelength column names,
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
    if y_col is None:
        exclude_cols = wavelength_cols.copy()
        if not generated_ids and specimen_id_col != "__GENERATED__":
            exclude_cols.append(specimen_id_col)

        y_col = auto_detect_y_column(df, exclude_cols)

    if y_col not in df.columns:
        raise ValueError(f"Target y column '{y_col}' not found in file")

    # Step 6: Extract data
    # Extract spectral data
    X = df[wavelength_cols].copy()
    X.index = specimen_ids

    # Convert spectral data values to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Convert wavelength column names to float and sort
    X.columns = X.columns.astype(float)
    X = X.sort_index(axis=1)  # Sort by wavelength

    # Extract target data
    y = df[y_col].copy()
    y.index = specimen_ids

    # Convert target values to numeric
    y = pd.to_numeric(y, errors='coerce')

    # Check for missing values (NaN) and remove affected specimens
    has_nan_X = X.isna().any(axis=1)
    has_nan_y = y.isna()
    has_nan = has_nan_X | has_nan_y

    if has_nan.any():
        n_missing = has_nan.sum()
        missing_specimens = X.index[has_nan].tolist()

        print(f"Warning: Found {n_missing} specimen(s) with missing values. Removing them.")
        print(f"  Removed specimens: {missing_specimens[:10]}")  # Show first 10
        if n_missing > 10:
            print(f"  ... and {n_missing - 10} more")

        # Remove rows with missing values
        X = X[~has_nan]
        y = y[~has_nan]

    # Step 7: Validation
    # Check for duplicate specimen IDs (only if not generated)
    if not generated_ids and specimen_ids.duplicated().any():
        n_duplicates = specimen_ids.duplicated().sum()
        duplicates = specimen_ids[specimen_ids.duplicated()].unique()[:5]
        print(f"Warning: Found {n_duplicates} duplicate specimen IDs. "
              f"Keeping first occurrence. Examples: {list(duplicates)}")

        # Keep first occurrence of each duplicate
        keep_mask = ~specimen_ids.duplicated(keep='first')
        X = X[keep_mask]
        y = y[keep_mask]

    # Check wavelength ordering
    wavelength_values = X.columns.values
    if not all(wavelength_values[i] < wavelength_values[i+1]
              for i in range(len(wavelength_values)-1)):
        print("Warning: Wavelengths were not strictly increasing. Sorted automatically.")

    # Step 8: Detect data type (reflectance vs absorbance)
    data_type, type_confidence, detection_method = detect_spectral_data_type(X)
    print(f"Detected data type: {data_type.capitalize()} (confidence: {type_confidence:.1f}%)")
    if type_confidence < 70:
        print(f"  WARNING: Low confidence detection. Method: {detection_method}")

    # Step 9: Compile metadata
    metadata = {
        'specimen_id_col': specimen_id_col,
        'y_col': y_col,
        'wavelength_cols': wavelength_cols,
        'n_spectra': len(X),
        'wavelength_range': (X.columns.min(), X.columns.max()),
        'file_format': 'combined_excel',
        'sheet_name': sheet_name,
        'generated_ids': generated_ids,
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
    }

    print(f"Successfully read {len(X)} spectra with {X.shape[1]} wavelengths from Excel file")
    print(f"  Specimen ID column: {specimen_id_col}")
    print(f"  Target column: {y_col}")

    return X, y, metadata


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

    # Create DataFrame
    df = pd.DataFrame([intensities], columns=wavelengths, index=[path.stem])
    df = df[sorted(df.columns)]

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'spc',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
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
    jcamp_data = jcamp.jcamp_read(str(path))

    # Extract x and y data
    wavelengths = jcamp_data['x']
    intensities = jcamp_data['y']

    # Create DataFrame
    df = pd.DataFrame([intensities], columns=wavelengths, index=[path.stem])
    df = df[sorted(df.columns)]

    # Detect data type
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'jcamp',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
        'jcamp_header': {k: v for k, v in jcamp_data.items() if k not in ['x', 'y']}
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

    metadata = {
        'n_spectra': 1,
        'wavelength_range': (df.columns.min(), df.columns.max()),
        'file_format': 'ascii',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method
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
    data_type, type_confidence, detection_method = detect_spectral_data_type(df)

    # Merge metadata
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('wavenumber_range', (df.columns.min(), df.columns.max())),
        'file_format': 'opus',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
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

    # Merge metadata
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('x_range', (df.columns.min(), df.columns.max())),
        'file_format': 'perkinelmer',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
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
    metadata = {
        'n_spectra': 1,
        'wavelength_range': file_metadata.get('wavenumber_range', (df.columns.min(), df.columns.max())),
        'file_format': 'agilent',
        'data_type': data_type,
        'type_confidence': type_confidence,
        'detection_method': detection_method,
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
    try:
        import jcamp
    except ImportError:
        raise ImportError(
            "JCAMP-DX export requires jcamp package.\n"
            "Install with: pip install jcamp"
        )

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

    # Write JCAMP file
    jcamp.jcamp_write(str(path), jcamp_dict)


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

    return _read_opus_dir(directory, **kwargs)


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