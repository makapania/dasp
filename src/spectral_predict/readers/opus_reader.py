"""Reader for Bruker OPUS binary spectroscopy files.

Bruker OPUS files use numbered extensions (.0, .1, .2, ..., .999)
and store infrared spectral data along with extensive metadata.

This module requires the optional 'brukeropus' library.
Install with: pip install brukeropus
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def read_opus_file(filepath: str | Path) -> Tuple[pd.Series, Dict]:
    """
    Read a single Bruker OPUS file.

    OPUS files use numbered extensions (.0, .1, .2, etc.) and contain
    binary spectral data from Bruker FTIR instruments.

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

    # Extract spectral data - try different data types in order of preference
    # Priority: absorbance > transmittance > sample > reference
    spectrum = None
    data_type = None
    x_data = None
    y_data = None

    # Check what data is available
    available_keys = getattr(opus_file, 'data_keys', [])

    # Try absorbance first (most common for analysis)
    if 'a' in available_keys or hasattr(opus_file, 'a'):
        try:
            abs_data = opus_file.a
            if hasattr(abs_data, 'x') and hasattr(abs_data, 'y'):
                x_data = abs_data.x  # wavenumbers (cm⁻¹)
                y_data = abs_data.y  # absorbance values
                data_type = 'absorbance'
        except (AttributeError, TypeError):
            pass

    # Try transmittance if absorbance not available
    if spectrum is None and ('t' in available_keys or hasattr(opus_file, 't')):
        try:
            trans_data = opus_file.t
            if hasattr(trans_data, 'x') and hasattr(trans_data, 'y'):
                x_data = trans_data.x
                y_data = trans_data.y
                data_type = 'transmittance'
        except (AttributeError, TypeError):
            pass

    # Try sample spectrum if neither absorbance nor transmittance available
    if spectrum is None and ('sm' in available_keys or hasattr(opus_file, 'sm')):
        try:
            sample_data = opus_file.sm
            if hasattr(sample_data, 'x') and hasattr(sample_data, 'y'):
                x_data = sample_data.x
                y_data = sample_data.y
                data_type = 'sample'
        except (AttributeError, TypeError):
            pass

    # Try reference spectrum as last resort
    if spectrum is None and ('rf' in available_keys or hasattr(opus_file, 'rf')):
        try:
            ref_data = opus_file.rf
            if hasattr(ref_data, 'x') and hasattr(ref_data, 'y'):
                x_data = ref_data.x
                y_data = ref_data.y
                data_type = 'reference'
        except (AttributeError, TypeError):
            pass

    if x_data is None or y_data is None:
        raise ValueError(
            f"No spectral data found in {filepath.name}. "
            f"Available data keys: {available_keys}"
        )

    # Convert to numpy arrays if needed
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Check for valid data
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError(f"Empty spectral data in {filepath.name}")

    if len(x_data) != len(y_data):
        raise ValueError(
            f"Mismatched data lengths in {filepath.name}: "
            f"x={len(x_data)}, y={len(y_data)}"
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
    metadata = {
        'filename': filepath.name,
        'data_type': data_type,
        'wavenumber_range': (float(x_data.min()), float(x_data.max())),
        'n_points': len(spectrum),
        'file_format': 'opus',
        'available_data_types': available_keys,
    }

    # Try to extract additional metadata if available
    try:
        # Sample name
        if hasattr(opus_file, 'snm'):
            metadata['sample_name'] = str(opus_file.snm)

        # Sample form/type
        if hasattr(opus_file, 'sfm'):
            metadata['sample_form'] = str(opus_file.sfm)

        # Instrument parameters (if available)
        # The opus_file object may have various instrument parameters
        # but they're not consistently documented in the API
    except Exception:
        pass  # Metadata extraction is best-effort

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
    data_types = []
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
            data_types.append(file_metadata.get('data_type', 'unknown'))
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
    type_counts = Counter(data_types)
    dominant_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'

    # Compile metadata
    metadata = {
        'n_spectra': len(df),
        'wavenumber_range': (float(df.columns.min()), float(df.columns.max())),
        'file_format': 'opus',
        'data_types': dict(type_counts),
        'dominant_data_type': dominant_type,
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
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
