"""Reader for PerkinElmer spectroscopy files.

PerkinElmer instruments produce .sp files containing infrared spectral data
in a binary format. This module uses the specio library to read these files.

Note: Uses specio-py310 for Python 3.10+ compatibility.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# PerkinElmer UV/Vis/NIR (Lambda, UV WinLab) .sp files are in nm and stop at 3300 nm;
# PerkinElmer FT-IR/FT-NIR (Spectrum) files are in cm-1 and a full range reaches
# 4000 cm-1 or more. Below this maximum the two overlap (e.g. 400-2500 could be a
# Vis-NIR range in nm or a truncated mid-IR range in cm-1).
_SP_MAX_NM = 3300.0
_UNIT_PATTERNS = (
    ('cm-1', re.compile(r'cm\s*(-|\^-|⁻)\s*1|1\s*/\s*cm|wavenumber', re.IGNORECASE)),
    ('nm', re.compile(r'\bnm\b|nanomet', re.IGNORECASE)),
)


def _unit_from_meta(meta: Any) -> Optional[str]:
    """Return an x unit stated in the reader's metadata, if any.

    specio-py310's .sp plugin records no x unit (only description, instrument
    fields and the wavelength range), but a unit-bearing key is honoured if present.
    """
    if not isinstance(meta, dict):
        return None
    for key, value in meta.items():
        if 'unit' not in str(key).lower() or not isinstance(value, str):
            continue
        found = {unit for unit, pattern in _UNIT_PATTERNS if pattern.search(value)}
        if len(found) == 1:
            return found.pop()
    return None


def _infer_sp_x_unit(
    x_min: float, x_max: float, meta: Any = None
) -> tuple[str, float, str, Optional[str]]:
    """Return ``(x_unit, confidence, method, warning)`` for a PerkinElmer spectrum."""
    stated = _unit_from_meta(meta)
    if stated is not None:
        return stated, 95.0, 'perkinelmer_metadata', None
    if x_max > _SP_MAX_NM:
        return 'cm-1', 80.0, 'perkinelmer_range', None
    return (
        'nm',
        40.0,
        'perkinelmer_ambiguous_range',
        f"x range {x_min:.1f}-{x_max:.1f} does not identify the unit: it fits a "
        f"UV/Vis/NIR spectrum in nm or a truncated IR spectrum in cm-1. Assumed nm; "
        f"check the x unit and switch it if this is an IR spectrum.",
    )


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
    from specio_py310 import specread

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

    x_unit, x_unit_confidence, x_unit_method, unit_warning = _infer_sp_x_unit(
        float(x_data.min()), float(x_data.max()), getattr(spectra_obj, 'meta', None)
    )
    if unit_warning:
        warnings.warn(f"{filepath.name}: {unit_warning}", UserWarning, stacklevel=2)

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
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
        'x_unit_warning': unit_warning,
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
    # set(): on case-insensitive filesystems (Windows) both globs return every file.
    sp_files = sorted(set(directory.glob("*.sp")) | set(directory.glob("*.SP")))

    if len(sp_files) == 0:
        raise ValueError(f"No .sp files found in {directory}")

    print(f"Found {len(sp_files)} .sp files")

    # Read each file
    spectra = {}
    x_units = []
    duplicate_stems = []
    failed_files = []

    unit_details: list[tuple[str, float, str, Optional[str]]] = []
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
            unit_details.append(
                (
                    sp_file.name,
                    file_metadata.get('x_unit_confidence', 50.0),
                    file_metadata.get('x_unit_detection_method', 'default'),
                    file_metadata.get('x_unit_warning'),
                )
            )
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

    # Collected for the GUI, which shows metadata['import_warnings'] in a dialog.
    import_warnings: list[str] = []
    if len(unit_counts) > 1:
        import_warnings.append(
            f"The .sp files have inconsistent x-axis units {dict(unit_counts)}; "
            f"proceeding with the dominant unit {dominant_unit}."
        )
    ambiguous = [name for name, _, _, warning in unit_details if warning]
    if ambiguous:
        import_warnings.append(
            f"{len(ambiguous)} of {len(unit_details)} .sp files have an x range that does "
            f"not identify the unit (UV/Vis/NIR in nm, or truncated IR in cm-1); nm was "
            f"assumed. Check the x unit. Files: {ambiguous[:10]}"
        )
    if failed_files:
        import_warnings.append(
            f"{len(failed_files)} .sp file(s) could not be read: {failed_files[:10]}"
        )
    for message in import_warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
    dominant_details = [d for d, u in zip(unit_details, x_units) if u == dominant_unit]
    x_unit_confidence = min((d[1] for d in dominant_details), default=50.0)
    x_unit_method = min(dominant_details, key=lambda d: d[1])[2] if dominant_details else 'default'

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
        'x_unit_confidence': x_unit_confidence,
        'x_unit_detection_method': x_unit_method,
        'file_format': 'sp',
        'vendor': 'PerkinElmer',
        'x_unit_counts': dict(unit_counts),
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
        'import_warnings': import_warnings,
    }

    print(f"Successfully read {len(df)} PerkinElmer .sp spectra")
    print(f"X-axis range: {metadata['x_range'][0]:.1f} - {metadata['x_range'][1]:.1f} {dominant_unit}")
    print(f"Data points per spectrum: {df.shape[1]}")

    return df, metadata
