"""Reader for Bruker OPUS binary spectroscopy files.

Bruker OPUS files use numbered extensions (.0, .1, .2, ..., .999)
and store infrared spectral data along with extensive metadata.

This module requires the optional 'brukeropus' library.
Install with: pip install brukeropus
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# OPUS data blocks in the order the reader prefers them, as
# (brukeropus attribute key, data_type reported in metadata). The keys are the
# attribute names brukeropus (1.4.x) gives each 1-D block on its OPUSFile object
# (see brukeropus.file.constants). Processed result spectra come first: Bruker
# files routinely carry the absorbance (AB) block *together with* the single-channel
# sample (ScSm) and background reference (ScRf) blocks, and the reference is the
# same instrument background in every file of a batch.
#
# 'aria'/'arit' are brukeropus's keys for arithmetic-result blocks that OPUS marks as
# absorbance-like / transmittance-like (difference spectra, saved manipulations); they
# rank after the native AB/TR blocks and carry the same data_type.
OPUS_BLOCK_PRIORITY: tuple[tuple[str, str], ...] = (
    ('a', 'absorbance'),
    ('t', 'transmittance'),
    ('aria', 'absorbance'),
    ('arit', 'transmittance'),
    ('r', 'reflectance'),
    ('logr', 'log_reflectance'),
    ('km', 'kubelka_munk'),
    ('atr', 'atr'),
    ('pas', 'photoacoustic'),
    ('ra', 'raman'),
    ('e', 'emission'),
    ('sm', 'sample'),
    ('rf', 'reference'),
)

# Single-channel blocks are only used when no processed spectrum is present.
_SINGLE_CHANNEL_KEYS = {'sm': 'sample (ScSm)', 'rf': 'background reference (ScRf)'}


def _select_opus_block(
    opus_file: Any, available_keys: list[str]
) -> tuple[Optional[str], Optional[str], Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """Return the first usable data block of an OPUS file in priority order.

    Args:
        opus_file: Object returned by ``brukeropus.read_opus``.
        available_keys: The file's ``data_keys`` (1-D blocks). When empty, each
            priority key is looked up as an attribute instead.

    Returns:
        ``(key, data_type, x, y, rejected)``. ``key`` is None when no block is
        usable. ``rejected`` lists ``"key: reason"`` for present but unusable blocks.
    """
    rejected: list[str] = []
    for key, data_type in OPUS_BLOCK_PRIORITY:
        if available_keys and key not in available_keys:
            continue
        try:
            block = getattr(opus_file, key, None)
        except (AttributeError, TypeError, RecursionError):
            block = None
        x = getattr(block, 'x', None) if block is not None else None
        y = getattr(block, 'y', None) if block is not None else None
        if x is None or y is None:
            if available_keys:
                rejected.append(f"{key}: no x/y arrays")
            continue
        try:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            rejected.append(f"{key}: non-numeric data")
            continue
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            rejected.append(f"{key}: not a 1-D spectrum (shape {y_arr.shape})")
            continue
        if x_arr.size == 0 or y_arr.size == 0:
            rejected.append(f"{key}: empty")
            continue
        if x_arr.size != y_arr.size:
            rejected.append(f"{key}: x has {x_arr.size} points, y has {y_arr.size}")
            continue
        # brukeropus builds x from the block's FXV/LXV/NPT parameters, so a corrupt
        # parameter gives a NaN or collapsed axis rather than an error.
        if not np.isfinite(x_arr).all():
            rejected.append(f"{key}: non-finite x coordinates")
            continue
        if np.unique(x_arr).size < 2:
            rejected.append(f"{key}: fewer than 2 distinct x coordinates")
            continue
        if not np.isfinite(y_arr).any():
            rejected.append(f"{key}: no finite values")
            continue
        return key, data_type, x_arr, y_arr, rejected
    return None, None, None, None, rejected


def read_opus_file(filepath: str | Path) -> Tuple[pd.Series, Dict]:
    """
    Read a single Bruker OPUS file.

    OPUS files use numbered extensions (.0, .1, .2, etc.) and contain
    binary spectral data from Bruker FTIR instruments.

    A file usually holds several data blocks. Exactly one is returned: the
    first usable block in ``OPUS_BLOCK_PRIORITY`` (absorbance, transmittance,
    reflectance and other processed spectra, then the single-channel sample, and
    the background reference only as a last resort, with a warning).
    ``metadata['opus_block']`` records which brukeropus block key was used.

    Parameters
    ----------
    filepath : str or Path
        Path to OPUS file (e.g., 'sample.0', 'sample.1')

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers/wavelengths as index
    metadata : dict
        Dictionary containing instrument metadata and file information

    Raises
    ------
    ImportError
        If brukeropus library is not installed
    ValueError
        If file cannot be read or contains no spectral data

    Examples
    --------
    >>> spectrum, metadata = read_opus_file('sample.0')
    >>> print(f"Wavenumber range: {spectrum.index.min()}-{spectrum.index.max()} cm⁻¹")
    >>> print(f"Data type: {metadata['data_type']}")
    """
    try:
        from brukeropus import read_opus
    except ImportError:
        raise ImportError(
            "Bruker OPUS file support requires the 'brukeropus' library.\n"
            "Install with: pip install brukeropus\n"
            "Or install all vendor formats: pip install spectral-predict[all-formats]"
        )

    filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")

    try:
        opus_file = read_opus(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to read OPUS file {filepath.name}: {e}")

    # brukeropus returns an object with is_opus=False (rather than raising) when the
    # file lacks the OPUS magic header. Its __getattr__ then recurses, so stop here.
    if getattr(opus_file, 'is_opus', True) is False:
        raise ValueError(f"Not a valid Bruker OPUS file: {filepath.name}")

    available_keys = list(getattr(opus_file, 'data_keys', None) or [])
    block_key, data_type, x_data, y_data, rejected = _select_opus_block(opus_file, available_keys)

    if block_key is None:
        detail = f" Unusable blocks: {rejected}." if rejected else ""
        raise ValueError(
            f"No spectral data found in {filepath.name}. "
            f"Available data keys: {available_keys}.{detail}"
        )

    if rejected:
        print(f"Warning: {filepath.name}: skipped unusable OPUS blocks {rejected}")

    if block_key in _SINGLE_CHANNEL_KEYS:
        warnings.warn(
            f"{filepath.name} has no processed spectrum (absorbance, transmittance, "
            f"reflectance, ...); using the single-channel {_SINGLE_CHANNEL_KEYS[block_key]} "
            f"block '{block_key}'. Values are raw detector intensities, not absorbance.",
            UserWarning,
            stacklevel=2,
        )

    # OPUS files typically use wavenumbers (cm⁻¹), need to convert to wavelengths (nm)
    # Wavelength (nm) = 10^7 / wavenumber (cm⁻¹)
    # But we'll keep as wavenumbers for now since that's the native format
    # Users can convert if needed

    # Create Series with wavenumbers as index
    # Note: OPUS data is typically in descending wavenumber order
    # We need to reverse to ascending order for consistency
    if x_data[0] > x_data[-1]:  # Descending order
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    spectrum = pd.Series(y_data, index=x_data)

    # Remove any duplicate indices (keep first occurrence)
    spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

    # Extract metadata
    block_label = getattr(getattr(opus_file, block_key, None), 'label', None)
    metadata = {
        'filename': filepath.name,
        'data_type': data_type,
        'opus_block': block_key,
        'opus_block_label': block_label if isinstance(block_label, str) else None,
        'wavenumber_range': (float(x_data.min()), float(x_data.max())),
        'n_points': len(spectrum),
        'file_format': 'opus',
        'available_data_types': available_keys,
    }

    # Try to extract additional metadata if available
    # brukeropus's __getattr__ returns None (not AttributeError) for absent parameters.
    for attr, meta_key in (('snm', 'sample_name'), ('sfm', 'sample_form')):
        try:
            value = getattr(opus_file, attr, None)
        except Exception:  # metadata extraction is best-effort
            value = None
        if value is not None:
            metadata[meta_key] = str(value)

    return spectrum, metadata


def read_opus_dir(directory: str | Path, pattern: str = "*.[0-9]*") -> Tuple[pd.DataFrame, Dict]:
    """
    Read all Bruker OPUS files from a directory.

    Searches for files with numbered extensions (.0, .1, .2, ..., .999)
    and combines them into a single DataFrame.

    Parameters
    ----------
    directory : str or Path
        Directory containing OPUS files
    pattern : str, optional
        Glob pattern for finding OPUS files. Default: "*.[0-9]*"
        Matches files with numeric extensions like .0, .1, .2, etc.

    Returns
    -------
    df : pd.DataFrame
        Wide matrix with rows = filename (stem), columns = wavenumbers (cm⁻¹)
    metadata : dict
        Contains n_spectra, wavenumber_range, file_format, data_types, etc.

    Raises
    ------
    ValueError
        If directory doesn't exist, no OPUS files found, or files can't be read
    ImportError
        If brukeropus library is not installed

    Examples
    --------
    >>> df, metadata = read_opus_dir('data/bruker_samples/')
    >>> print(f"Loaded {len(df)} spectra")
    >>> print(f"Wavenumber range: {metadata['wavenumber_range']}")

    Notes
    -----
    - OPUS files use numbered extensions (.0, .1, .2, ..., .999)
    - Each file may contain multiple data types (absorbance, transmittance, etc.)
    - This function prioritizes absorbance data when available
    - Wavenumbers are kept in native cm⁻¹ units (not converted to nm)
    - If multiple files have the same base name (e.g., sample.0 and sample.1),
      the last one read will overwrite earlier ones
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Find OPUS files with numbered extensions
    opus_files = list(directory.glob(pattern))

    # Also try common OPUS extensions explicitly
    for ext in range(100):  # Check .0 through .99
        opus_files.extend(directory.glob(f"*.{ext}"))

    # Remove duplicates (glob might find same file multiple times)
    opus_files = list(set(opus_files))

    if len(opus_files) == 0:
        raise ValueError(
            f"No OPUS files found in {directory}\n"
            f"OPUS files typically have numbered extensions like .0, .1, .2, etc."
        )

    print(f"Found {len(opus_files)} OPUS files")

    # Read each file
    spectra = {}
    # Per kept stem, so a file overwritten by a same-stem file is not counted.
    data_types: dict[str, str] = {}
    opus_blocks: dict[str, str] = {}
    duplicate_stems = []
    failed_files = []

    for opus_file in sorted(opus_files):
        # Use stem (filename without extension) as identifier
        stem = opus_file.stem

        # Check for duplicate stems (files with same name but different extensions)
        if stem in spectra:
            duplicate_stems.append(stem)
            print(
                f"Warning: Multiple OPUS files with base name '{stem}' "
                f"(extensions {opus_file.suffix}). Keeping last one."
            )

        try:
            spectrum, file_metadata = read_opus_file(opus_file)
            spectra[stem] = spectrum
            data_types[stem] = file_metadata.get('data_type', 'unknown')
            opus_blocks[stem] = file_metadata.get('opus_block', 'unknown')
        except Exception as e:
            print(f"Warning: Could not read {opus_file.name}: {e}")
            failed_files.append(opus_file.name)
            continue

    if len(spectra) == 0:
        error_msg = f"No valid OPUS spectra could be read from {directory}"
        if failed_files:
            error_msg += f"\nFailed files: {failed_files[:5]}"
        raise ValueError(error_msg)

    if duplicate_stems:
        print(
            f"\nWarning: Found {len(set(duplicate_stems))} files with duplicate base names. "
            f"Only the last occurrence of each is kept."
        )

    # Combine into DataFrame
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (wavenumbers) in ascending order
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 50:  # OPUS files usually have hundreds of points
        print(
            f"Warning: Only {df.shape[1]} wavenumbers found. "
            f"This is unusually low for OPUS files."
        )

    # Check if wavenumbers are strictly increasing
    wavenumbers = np.array(df.columns)
    if not np.all(wavenumbers[1:] > wavenumbers[:-1]):
        print("Warning: Wavenumbers were not strictly increasing after sorting.")

    # Detect dominant data type
    from collections import Counter
    type_counts = Counter(data_types.values())
    dominant_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
    block_counts = Counter(opus_blocks.values())

    # Collected for the GUI, which shows metadata['import_warnings'] in a dialog
    # (warnings.warn and print only reach the console).
    import_warnings: list[str] = []
    if len(type_counts) > 1:
        # e.g. some files carry AB and others only single channels: the rows of X are
        # then in different units, which no downstream step can reconcile.
        import_warnings.append(
            f"The OPUS files contain different data types {dict(type_counts)}, so the "
            f"combined matrix mixes them. Check the per-file blocks before modelling."
        )
    single_channel = sorted(s for s, k in opus_blocks.items() if k in _SINGLE_CHANNEL_KEYS)
    if single_channel:
        import_warnings.append(
            f"{len(single_channel)} of {len(spectra)} OPUS files have no processed spectrum "
            f"(absorbance, transmittance, reflectance, ...) and were read from a "
            f"single-channel block (ScSm/ScRf): raw detector intensities, not absorbance. "
            f"Files: {single_channel[:10]}"
        )
    if failed_files:
        import_warnings.append(
            f"{len(failed_files)} OPUS file(s) could not be read: {failed_files[:10]}"
        )
    for message in import_warnings:
        warnings.warn(message, UserWarning, stacklevel=2)

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavenumber_range': (float(df.columns.min()), float(df.columns.max())),
        'file_format': 'opus',
        'data_types': dict(type_counts),
        'dominant_data_type': dominant_type,
        'opus_blocks': dict(block_counts),
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
        'import_warnings': import_warnings,
        'x_unit': 'cm-1',
        'x_unit_confidence': 99.0,
        'x_unit_detection_method': 'opus_native',
    }

    print(f"Successfully read {len(df)} OPUS spectra")
    print(f"Wavenumber range: {metadata['wavenumber_range'][0]:.1f} - {metadata['wavenumber_range'][1]:.1f} cm⁻¹")
    print(f"Data types: {dict(type_counts)}")

    return df, metadata


def convert_wavenumber_to_wavelength(wavenumber_cm: float) -> float:
    """
    Convert wavenumber (cm⁻¹) to wavelength (nm).

    Parameters
    ----------
    wavenumber_cm : float
        Wavenumber in cm⁻¹

    Returns
    -------
    float
        Wavelength in nm

    Examples
    --------
    >>> convert_wavenumber_to_wavelength(4000)  # 4000 cm⁻¹
    2500.0  # nm
    >>> convert_wavenumber_to_wavelength(400)   # 400 cm⁻¹
    25000.0  # nm
    """
    return 1e7 / wavenumber_cm


def convert_wavelength_to_wavenumber(wavelength_nm: float) -> float:
    """
    Convert wavelength (nm) to wavenumber (cm⁻¹).

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nm

    Returns
    -------
    float
        Wavenumber in cm⁻¹

    Examples
    --------
    >>> convert_wavelength_to_wavenumber(2500)  # 2500 nm
    4000.0  # cm⁻¹
    >>> convert_wavelength_to_wavenumber(25000)  # 25000 nm
    400.0  # cm⁻¹
    """
    return 1e7 / wavelength_nm
