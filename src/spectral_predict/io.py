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
    pd.DataFrame
        Wide matrix with rows = id, columns = float wavelengths (nm), sorted ascending
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

        # Convert to wide: single row with wavelengths as columns
        result = pd.DataFrame(
            {float(row[wl_col]): [row[val_col]] for _, row in df_wide.iterrows()}, index=[sample_id]
        )

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

    return result


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


def align_xy(X, ref, id_column, target):
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

    Returns
    -------
    X_aligned : pd.DataFrame
        Aligned spectral data
    y : pd.Series
        Target values, same order as X_aligned
    """
    if target not in ref.columns:
        raise ValueError(
            f"Target '{target}' not found in reference. Available: {list(ref.columns)}"
        )

    # Try exact match first
    common_ids = X.index.intersection(ref.index)

    # If no exact matches, try normalized matching
    if len(common_ids) == 0:
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

    # Drop any NaN targets
    valid_mask = ~y.isna()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"Warning: Dropping {n_dropped} samples with missing target values")
        X_aligned = X_aligned[valid_mask]
        y = y[valid_mask]

    if len(y) == 0:
        raise ValueError("No valid samples after alignment and NaN removal")

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
    pd.DataFrame
        Wide matrix with rows = filename, columns = wavelengths (nm)
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
    for asd_file in sorted(asd_files):
        try:
            spectrum = _read_single_asd_ascii(asd_file, reader_mode)
            spectra[asd_file.stem] = spectrum
        except UnicodeDecodeError:
            # Binary ASD file detected - try to read with SpecDAL
            spectrum = _handle_binary_asd(asd_file, reader_mode)
            if spectrum is not None:
                spectra[asd_file.stem] = spectrum
        except Exception as e:
            print(f"Warning: Could not read {asd_file.name}: {e}")

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

    return df


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
