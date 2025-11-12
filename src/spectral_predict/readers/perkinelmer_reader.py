"""Reader for PerkinElmer spectroscopy files.

PerkinElmer instruments produce .sp files containing infrared spectral data
in a binary format. This module uses the specio library to read these files.

This module requires the optional 'specio' library.
Install with: pip install specio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def read_sp_file(filepath: str | Path) -> Tuple[pd.Series, Dict]:
    """
    Read a single PerkinElmer .sp file.

    .sp files are binary files produced by PerkinElmer IR instruments
    containing spectral data and metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to .sp file

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers/wavelengths as index
    metadata : dict
        Dictionary containing instrument metadata and file information

    Raises
    ------
    ImportError
        If specio library is not installed
    ValueError
        If file cannot be read or contains no spectral data

    Examples
    --------
    >>> spectrum, metadata = read_sp_file('sample.sp')
    >>> print(f"Wavelength range: {spectrum.index.min()}-{spectrum.index.max()}")
    >>> print(f"Number of points: {len(spectrum)}")
    """
    try:
        from specio import specread
    except ImportError:
        raise ImportError(
            "PerkinElmer .sp file support requires the 'specio' library.\n"
            "Install with: pip install specio\n"
            "Or install all vendor formats: pip install spectral-predict[all-formats]"
        )

    filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")

    if not filepath.suffix.lower() == '.sp':
        print(f"Warning: File {filepath.name} does not have .sp extension")

    try:
        # Read the .sp file using specio
        spectra_obj = specread(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to read .sp file {filepath.name}: {e}")

    # Extract wavelength/wavenumber and amplitude data
    try:
        # specio returns objects with 'wavelength' and 'amplitudes' attributes
        if hasattr(spectra_obj, 'wavelength') and hasattr(spectra_obj, 'amplitudes'):
            x_data = np.array(spectra_obj.wavelength)
            y_data = np.array(spectra_obj.amplitudes)
        # Also try 'wavenumber' in case that's used
        elif hasattr(spectra_obj, 'wavenumber') and hasattr(spectra_obj, 'amplitudes'):
            x_data = np.array(spectra_obj.wavenumber)
            y_data = np.array(spectra_obj.amplitudes)
        else:
            raise ValueError(
                f"Spectral object does not have expected attributes. "
                f"Available: {dir(spectra_obj)}"
            )
    except Exception as e:
        raise ValueError(f"Failed to extract spectral data from {filepath.name}: {e}")

    # Validate data
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError(f"Empty spectral data in {filepath.name}")

    # Handle multi-dimensional data (e.g., multiple spectra in one file)
    if y_data.ndim > 1:
        if y_data.shape[0] == 1:
            # Single spectrum stored as 2D array
            y_data = y_data.flatten()
        else:
            # Multiple spectra - take the first one and warn
            print(
                f"Warning: {filepath.name} contains {y_data.shape[0]} spectra. "
                f"Using first spectrum only."
            )
            y_data = y_data[0]

    if len(x_data) != len(y_data):
        raise ValueError(
            f"Mismatched data lengths in {filepath.name}: "
            f"x={len(x_data)}, y={len(y_data)}"
        )

    # Determine if x_data is wavelengths or wavenumbers based on typical ranges
    # Wavenumbers: typically 400-4000 cm⁻¹
    # Wavelengths: typically 2500-25000 nm (2.5-25 μm for IR)
    x_min, x_max = x_data.min(), x_data.max()

    if x_max <= 5000 and x_min >= 100:
        # Likely wavenumbers (cm⁻¹)
        x_unit = 'wavenumber_cm-1'
    elif x_max >= 1000 and x_min >= 100:
        # Likely wavelengths (nm)
        x_unit = 'wavelength_nm'
    else:
        # Ambiguous - assume wavenumbers (more common for IR)
        x_unit = 'wavenumber_cm-1'
        print(
            f"Warning: Could not determine x-axis units for {filepath.name}. "
            f"Assuming wavenumbers (cm⁻¹). Range: {x_min:.1f}-{x_max:.1f}"
        )

    # Ensure data is in ascending order
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    # Create Series
    spectrum = pd.Series(y_data, index=x_data)

    # Remove any duplicate indices (keep first occurrence)
    spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

    # Extract metadata
    metadata = {
        'filename': filepath.name,
        'x_unit': x_unit,
        'x_range': (float(x_data.min()), float(x_data.max())),
        'n_points': len(spectrum),
        'file_format': 'sp',
        'vendor': 'PerkinElmer',
    }

    # Try to extract additional metadata from spectra_obj
    try:
        # Check for metadata attributes
        if hasattr(spectra_obj, 'meta'):
            metadata['specio_metadata'] = spectra_obj.meta
    except Exception:
        pass  # Metadata extraction is best-effort

    return spectrum, metadata


def read_sp_dir(directory: str | Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Read all PerkinElmer .sp files from a directory.

    Searches for .sp files and combines them into a single DataFrame.

    Parameters
    ----------
    directory : str or Path
        Directory containing .sp files

    Returns
    -------
    df : pd.DataFrame
        Wide matrix with rows = filename (stem), columns = x-axis values
    metadata : dict
        Contains n_spectra, x_range, file_format, x_unit, etc.

    Raises
    ------
    ValueError
        If directory doesn't exist, no .sp files found, or files can't be read
    ImportError
        If specio library is not installed

    Examples
    --------
    >>> df, metadata = read_sp_dir('data/perkinelmer_samples/')
    >>> print(f"Loaded {len(df)} spectra")
    >>> print(f"X-axis unit: {metadata['x_unit']}")
    >>> print(f"Range: {metadata['x_range']}")

    Notes
    -----
    - All .sp files in the directory are read
    - Files with the same base name will overwrite each other (last one wins)
    - X-axis units (wavelength vs wavenumber) are auto-detected
    - If files have inconsistent x-axis units, a warning is printed
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Find .sp files
    sp_files = list(directory.glob("*.sp")) + list(directory.glob("*.SP"))

    if len(sp_files) == 0:
        raise ValueError(f"No .sp files found in {directory}")

    print(f"Found {len(sp_files)} .sp files")

    # Read each file
    spectra = {}
    x_units = []
    duplicate_stems = []
    failed_files = []

    for sp_file in sorted(sp_files):
        stem = sp_file.stem

        # Check for duplicate stems
        if stem in spectra:
            duplicate_stems.append(stem)
            print(
                f"Warning: Duplicate filename '{stem}' - "
                f"later file will overwrite earlier one"
            )

        try:
            spectrum, file_metadata = read_sp_file(sp_file)
            spectra[stem] = spectrum
            x_units.append(file_metadata.get('x_unit', 'unknown'))
        except Exception as e:
            print(f"Warning: Could not read {sp_file.name}: {e}")
            failed_files.append(sp_file.name)
            continue

    if len(spectra) == 0:
        error_msg = f"No valid .sp spectra could be read from {directory}"
        if failed_files:
            error_msg += f"\nFailed files: {failed_files[:5]}"
        raise ValueError(error_msg)

    if duplicate_stems:
        print(
            f"\nWarning: Found {len(set(duplicate_stems))} duplicate filenames. "
            f"Only the last occurrence of each is kept."
        )

    # Check for inconsistent x-axis units
    from collections import Counter
    unit_counts = Counter(x_units)
    dominant_unit = unit_counts.most_common(1)[0][0] if unit_counts else 'unknown'

    if len(unit_counts) > 1:
        print(
            f"Warning: Files have inconsistent x-axis units: {dict(unit_counts)}\n"
            f"Proceeding with dominant unit: {dominant_unit}"
        )

    # Combine into DataFrame
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (x-axis values) in ascending order
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 50:
        print(
            f"Warning: Only {df.shape[1]} data points found. "
            f"This is unusually low for IR spectroscopy."
        )

    # Check if x-values are strictly increasing
    x_values = np.array(df.columns)
    if not np.all(x_values[1:] > x_values[:-1]):
        print("Warning: X-axis values were not strictly increasing after sorting.")

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'x_range': (float(df.columns.min()), float(df.columns.max())),
        'x_unit': dominant_unit,
        'file_format': 'sp',
        'vendor': 'PerkinElmer',
        'x_unit_counts': dict(unit_counts),
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
    }

    print(f"Successfully read {len(df)} PerkinElmer .sp spectra")
    print(f"X-axis range: {metadata['x_range'][0]:.1f} - {metadata['x_range'][1]:.1f} {dominant_unit}")
    print(f"Data points per spectrum: {df.shape[1]}")

    return df, metadata
