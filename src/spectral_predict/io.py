"""I/O functions for reading spectral data and reference files."""

import pandas as pd
import numpy as np
from pathlib import Path


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
    Read reference CSV with target variables.

    Parameters
    ----------
    path : str or Path
        Path to reference CSV
    id_column : str
        Column name to use as index (e.g., 'sample_id', 'filename')

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by id_column
    """
    path = Path(path)
    df = pd.read_csv(path)

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {path}. Available: {list(df.columns)}")

    # Check for duplicate IDs BEFORE setting index
    duplicates = df[id_column].duplicated()
    if duplicates.any():
        dup_ids = df.loc[duplicates, id_column].unique()
        n_dups = duplicates.sum()
        print(f"\n⚠️ WARNING: Found {n_dups} duplicate sample IDs in reference CSV!")
        print(f"Duplicate IDs: {list(dup_ids[:10])}")
        if len(dup_ids) > 10:
            print(f"... and {len(dup_ids) - 10} more")
        print("\nKeeping FIRST occurrence of each duplicate. Please check your CSV file.\n")

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
        unmatched_spectra = sorted(list(original_X_ids - set(matched_before_nan_filter)))
        unmatched_reference = sorted(list(original_ref_ids - set(matched_before_nan_filter)))

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

    Uses the pyspectra library to read binary .spc files.

    Parameters
    ----------
    spc_dir : str or Path
        Directory containing SPC files

    Returns
    -------
    pd.DataFrame
        Wide matrix with rows = filename, columns = wavelengths (nm)

    Raises
    ------
    ValueError
        If directory doesn't exist, no SPC files found, or pyspectra not installed
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

    # Try to import pyspectra
    try:
        from pyspectra.readers.read_spc import read_spc_dir as pyspectra_read_spc_dir
    except ImportError:
        raise ValueError(
            "SPC file support requires the pyspectra library.\n"
            "Install it with: pip install pyspectra"
        )

    # Read all SPC files
    try:
        df_spc, dict_spc = pyspectra_read_spc_dir(str(spc_dir))

        # pyspectra returns DataFrame with columns=files, rows=wavelengths
        # We need to transpose: rows=samples, columns=wavelengths
        df = df_spc.T

        # Ensure column names are floats (wavelengths)
        df.columns = df.columns.astype(float)

        # Sort columns by wavelength
        df = df[sorted(df.columns)]

        # Use stem (filename without extension) as index
        df.index = [Path(idx).stem if isinstance(idx, str) else idx for idx in df.index]

        # Validate
        if df.shape[1] < 100:
            raise ValueError(f"Expected at least 100 wavelengths, got {df.shape[1]}")

        # Check wavelengths are increasing
        wls = np.array(df.columns)
        if not np.all(wls[1:] > wls[:-1]):
            raise ValueError("Wavelengths must be strictly increasing")

        print(f"Successfully read {len(df)} SPC spectra with {df.shape[1]} wavelengths")
        return df

    except Exception as e:
        raise ValueError(f"Failed to read SPC files: {e}")


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