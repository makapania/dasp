"""Reader for Agilent spectroscopy files.

Agilent instruments produce various file formats for infrared spectroscopy:
- .seq files: Single tile hyperspectral images
- .dmt files: Multi-tile mosaic hyperspectral images
- .asp files: Agilent IR spectrum files (alternative parser available)
- .bsw files: Agilent batch files (limited support)

This module uses the agilent-ir-formats library to read these files.

This module requires the optional 'agilent-ir-formats' library.
Install with: pip install agilent-ir-formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List


def read_agilent_file(filepath: str | Path, extract_mode: str = 'total') -> Tuple[pd.Series, Dict]:
    """
    Read a single Agilent file (.seq, .dmt, .asp, or .bsw).

    For hyperspectral imaging files (.seq, .dmt), this extracts either:
    - 'total': Sum of all pixel spectra (default)
    - 'first': First pixel spectrum only
    - 'mean': Mean spectrum across all pixels

    Parameters
    ----------
    filepath : str or Path
        Path to Agilent file
    extract_mode : str, optional
        How to extract spectrum from hyperspectral data:
        'total' (sum), 'first' (first pixel), or 'mean' (average).
        Default: 'total'

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers (cm⁻¹) as index
    metadata : dict
        Dictionary containing instrument metadata, file information,
        and image dimensions (for hyperspectral files)

    Raises
    ------
    ImportError
        If agilent-ir-formats library is not installed
    ValueError
        If file cannot be read or contains no spectral data

    Examples
    --------
    >>> spectrum, metadata = read_agilent_file('sample.seq')
    >>> print(f"Wavenumber range: {spectrum.index.min()}-{spectrum.index.max()} cm⁻¹")
    >>> print(f"Image dimensions: {metadata.get('image_shape', 'N/A')}")
    """
    try:
        from agilent_ir_formats.agilent_ir_file import AgilentIRFile
    except ImportError:
        raise ImportError(
            "Agilent file support requires the 'agilent-ir-formats' library.\n"
            "Install with: pip install agilent-ir-formats\n"
            "Or install all vendor formats: pip install spectral-predict[all-formats]"
        )

    filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")

    # Check file extension
    suffix = filepath.suffix.lower()
    supported_extensions = ['.seq', '.dmt', '.asp', '.bsw']

    if suffix not in supported_extensions:
        print(
            f"Warning: File {filepath.name} has extension {suffix}. "
            f"Supported: {supported_extensions}"
        )

    try:
        # Create reader and read file
        reader = AgilentIRFile()
        reader.read(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to read Agilent file {filepath.name}: {e}")

    # Extract spectral data
    try:
        # Get wavenumbers
        if hasattr(reader, 'wavenumbers'):
            wavenumbers = np.array(reader.wavenumbers)
        else:
            raise ValueError("File does not contain wavenumber data")

        # Get intensity data
        # For hyperspectral files, this is typically a 3D array (height, width, spectral_points)
        if hasattr(reader, 'intensities'):
            intensities = np.array(reader.intensities)
        elif hasattr(reader, 'data'):
            intensities = np.array(reader.data)
        else:
            raise ValueError("File does not contain intensity data")

    except Exception as e:
        raise ValueError(f"Failed to extract spectral data from {filepath.name}: {e}")

    # Handle different data shapes
    if intensities.ndim == 1:
        # Single spectrum
        spectrum_data = intensities
        image_shape = None

    elif intensities.ndim == 2:
        # Could be (n_spectra, n_points) or (height, width) for single wavelength
        # Assume first dimension is spectra
        if extract_mode == 'first':
            spectrum_data = intensities[0, :]
        elif extract_mode == 'mean':
            spectrum_data = intensities.mean(axis=0)
        else:  # 'total'
            spectrum_data = intensities.sum(axis=0)
        image_shape = (intensities.shape[0], 1)

    elif intensities.ndim == 3:
        # Hyperspectral image: (height, width, spectral_points)
        height, width, n_points = intensities.shape
        image_shape = (height, width)

        if extract_mode == 'first':
            # Take first pixel (top-left corner)
            spectrum_data = intensities[0, 0, :]
        elif extract_mode == 'mean':
            # Average across all pixels
            spectrum_data = intensities.reshape(-1, n_points).mean(axis=0)
        else:  # 'total'
            # Sum across all pixels (equivalent to reader.total_spectrum if available)
            spectrum_data = intensities.reshape(-1, n_points).sum(axis=0)

    else:
        raise ValueError(
            f"Unexpected data shape: {intensities.shape}. "
            f"Expected 1D, 2D, or 3D array."
        )

    # Validate data
    if len(wavenumbers) != len(spectrum_data):
        raise ValueError(
            f"Mismatched data lengths: "
            f"wavenumbers={len(wavenumbers)}, intensities={len(spectrum_data)}"
        )

    if len(wavenumbers) == 0:
        raise ValueError("Empty spectral data")

    # Ensure ascending order (some files may be descending)
    if wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1]
        spectrum_data = spectrum_data[::-1]

    # Create Series
    spectrum = pd.Series(spectrum_data, index=wavenumbers)

    # Remove any duplicate indices (keep first occurrence)
    spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

    # Extract metadata
    metadata = {
        'filename': filepath.name,
        'file_format': suffix.lstrip('.'),
        'vendor': 'Agilent',
        'wavenumber_range': (float(wavenumbers.min()), float(wavenumbers.max())),
        'n_points': len(spectrum),
        'extract_mode': extract_mode,
        'image_shape': image_shape,
        'is_hyperspectral': image_shape is not None,
    }

    # Try to extract additional metadata from reader
    try:
        if hasattr(reader, 'metadata'):
            agilent_metadata = reader.metadata
            if isinstance(agilent_metadata, dict):
                metadata['agilent_metadata'] = agilent_metadata
    except Exception:
        pass  # Metadata extraction is best-effort

    return spectrum, metadata


def read_agilent_dir(
    directory: str | Path,
    extensions: Optional[List[str]] = None,
    extract_mode: str = 'total'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Read all Agilent files from a directory.

    Searches for Agilent files (.seq, .dmt, .asp, .bsw) and combines
    them into a single DataFrame.

    Parameters
    ----------
    directory : str or Path
        Directory containing Agilent files
    extensions : list of str, optional
        List of file extensions to search for (without dots).
        Default: ['seq', 'dmt', 'asp', 'bsw']
    extract_mode : str, optional
        How to extract spectra from hyperspectral data:
        'total' (sum), 'first' (first pixel), or 'mean' (average).
        Default: 'total'

    Returns
    -------
    df : pd.DataFrame
        Wide matrix with rows = filename (stem), columns = wavenumbers (cm⁻¹)
    metadata : dict
        Contains n_spectra, wavenumber_range, file_format, etc.

    Raises
    ------
    ValueError
        If directory doesn't exist, no files found, or files can't be read
    ImportError
        If agilent-ir-formats library is not installed

    Examples
    --------
    >>> df, metadata = read_agilent_dir('data/agilent_samples/')
    >>> print(f"Loaded {len(df)} spectra")
    >>> print(f"Wavenumber range: {metadata['wavenumber_range']}")

    >>> # Read only .seq files with mean extraction
    >>> df, metadata = read_agilent_dir('data/', extensions=['seq'], extract_mode='mean')

    Notes
    -----
    - Supported formats: .seq, .dmt, .asp, .bsw
    - .seq and .dmt files contain hyperspectral imaging data
    - For hyperspectral files, spectra are extracted using extract_mode
    - Files with the same base name will overwrite each other (last one wins)
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Set default extensions if not provided
    if extensions is None:
        extensions = ['seq', 'dmt', 'asp', 'bsw']

    # Find Agilent files
    agilent_files = []
    for ext in extensions:
        agilent_files.extend(directory.glob(f"*.{ext}"))
        agilent_files.extend(directory.glob(f"*.{ext.upper()}"))

    # Remove duplicates
    agilent_files = list(set(agilent_files))

    if len(agilent_files) == 0:
        raise ValueError(
            f"No Agilent files found in {directory}\n"
            f"Searched for extensions: {extensions}"
        )

    print(f"Found {len(agilent_files)} Agilent files")

    # Read each file
    spectra = {}
    file_formats = []
    duplicate_stems = []
    failed_files = []
    hyperspectral_count = 0

    for agilent_file in sorted(agilent_files):
        stem = agilent_file.stem

        # Check for duplicate stems
        if stem in spectra:
            duplicate_stems.append(stem)
            print(
                f"Warning: Duplicate filename '{stem}' - "
                f"later file will overwrite earlier one"
            )

        try:
            spectrum, file_metadata = read_agilent_file(agilent_file, extract_mode=extract_mode)
            spectra[stem] = spectrum
            file_formats.append(file_metadata.get('file_format', 'unknown'))

            if file_metadata.get('is_hyperspectral', False):
                hyperspectral_count += 1

        except Exception as e:
            print(f"Warning: Could not read {agilent_file.name}: {e}")
            failed_files.append(agilent_file.name)
            continue

    if len(spectra) == 0:
        error_msg = f"No valid Agilent spectra could be read from {directory}"
        if failed_files:
            error_msg += f"\nFailed files: {failed_files[:5]}"
        raise ValueError(error_msg)

    if duplicate_stems:
        print(
            f"\nWarning: Found {len(set(duplicate_stems))} duplicate filenames. "
            f"Only the last occurrence of each is kept."
        )

    # Combine into DataFrame
    df = pd.DataFrame(spectra).T  # Transpose so rows = samples

    # Sort columns (wavenumbers) in ascending order
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 50:
        print(
            f"Warning: Only {df.shape[1]} wavenumbers found. "
            f"This is unusually low for IR spectroscopy."
        )

    # Check if wavenumbers are strictly increasing
    wavenumbers = np.array(df.columns)
    if not np.all(wavenumbers[1:] > wavenumbers[:-1]):
        print("Warning: Wavenumbers were not strictly increasing after sorting.")

    # Compile metadata
    from collections import Counter
    format_counts = Counter(file_formats)

    metadata = {
        'n_spectra': len(df),
        'wavenumber_range': (float(df.columns.min()), float(df.columns.max())),
        'vendor': 'Agilent',
        'file_formats': dict(format_counts),
        'extract_mode': extract_mode,
        'n_hyperspectral': hyperspectral_count,
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
    }

    print(f"Successfully read {len(df)} Agilent spectra")
    print(f"Wavenumber range: {metadata['wavenumber_range'][0]:.1f} - {metadata['wavenumber_range'][1]:.1f} cm⁻¹")
    print(f"File formats: {dict(format_counts)}")
    if hyperspectral_count > 0:
        print(f"Hyperspectral images: {hyperspectral_count} (extracted using '{extract_mode}' mode)")

    return df, metadata


def read_seq_file(filepath: str | Path, extract_mode: str = 'total') -> Tuple[pd.Series, Dict]:
    """
    Read a single Agilent .seq file (single-tile hyperspectral image).

    This is a convenience wrapper around read_agilent_file() specifically for .seq files.

    Parameters
    ----------
    filepath : str or Path
        Path to .seq file
    extract_mode : str, optional
        How to extract spectrum: 'total', 'first', or 'mean'. Default: 'total'

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers as index
    metadata : dict
        File metadata including image dimensions
    """
    return read_agilent_file(filepath, extract_mode=extract_mode)


def read_dmt_file(filepath: str | Path, extract_mode: str = 'total') -> Tuple[pd.Series, Dict]:
    """
    Read a single Agilent .dmt file (multi-tile mosaic hyperspectral image).

    This is a convenience wrapper around read_agilent_file() specifically for .dmt files.

    Parameters
    ----------
    filepath : str or Path
        Path to .dmt file
    extract_mode : str, optional
        How to extract spectrum: 'total', 'first', or 'mean'. Default: 'total'

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers as index
    metadata : dict
        File metadata including image dimensions
    """
    return read_agilent_file(filepath, extract_mode=extract_mode)


def read_asp_file(filepath: str | Path) -> Tuple[pd.Series, Dict]:
    """
    Read a single Agilent .asp file.

    This is a convenience wrapper around read_agilent_file() specifically for .asp files.

    Parameters
    ----------
    filepath : str or Path
        Path to .asp file

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers as index
    metadata : dict
        File metadata
    """
    return read_agilent_file(filepath, extract_mode='total')
