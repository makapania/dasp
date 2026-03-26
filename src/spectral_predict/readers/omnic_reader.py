"""Reader for Thermo Omnic spectroscopy files.

Thermo Omnic (Nicolet) instruments produce .spa (single spectrum) and .spg
(spectral group) files in a proprietary binary format. This module uses the
spectrochempy-omnic library to read these files.

Note: Requires spectrochempy-omnic and requests packages.
Install with: pip install spectrochempy-omnic requests
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


# Mapping from OMNICReader .units to DASP data_type / source_data_type
_OMNIC_DATATYPE_MAP = {
    'absorbance': ('absorbance', 'absorbance'),
    'log(1/r)': ('absorbance', 'Log(1/R)'),
    'transmittance': ('reflectance', 'transmittance'),
    'reflectance': ('reflectance', 'reflectance'),
    'kubelka-munk': ('absorbance', 'Kubelka-Munk'),
}

# Mapping from OMNICReader .x_units to DASP x_unit
_OMNIC_XUNIT_MAP = {
    'cm^-1': 'cm-1',
    'nm': 'nm',
    'um': 'nm',  # micrometers -> convert values to nm
}


def _import_omnic_reader():
    """Lazy import with clear error message."""
    try:
        from spectrochempy_omnic import OMNICReader
        return OMNICReader
    except ImportError:
        raise ImportError(
            "Thermo Omnic file support requires 'spectrochempy-omnic' and 'requests'.\n"
            "Install with: pip install spectrochempy-omnic requests"
        )


def _map_data_type(units_str: str) -> Tuple[str, str]:
    """Map OMNICReader units string to (data_type, source_data_type)."""
    if units_str is None:
        return 'absorbance', 'unknown'
    key = units_str.strip().lower()
    if key in _OMNIC_DATATYPE_MAP:
        return _OMNIC_DATATYPE_MAP[key]
    # Unknown unit type — default to absorbance with warning
    print(f"Warning: Unknown Omnic data type '{units_str}'. Defaulting to absorbance.")
    return 'absorbance', units_str


def _map_x_unit(x_units_str: str) -> Tuple[str, bool]:
    """Map OMNICReader x_units to DASP x_unit. Returns (x_unit, needs_conversion)."""
    if x_units_str is None:
        return 'cm-1', False
    key = x_units_str.strip().lower()
    if key in ('cm^-1', 'cm-1', '1/cm'):
        return 'cm-1', False
    elif key == 'nm':
        return 'nm', False
    elif key in ('um', 'micrometer', 'micrometers'):
        return 'nm', True  # Need to convert um -> nm
    # Default to cm-1 for FTIR
    print(f"Warning: Unknown Omnic x-unit '{x_units_str}'. Defaulting to cm-1.")
    return 'cm-1', False


def read_spa_file(filepath: str | Path) -> Tuple[pd.Series, Dict]:
    """
    Read a single Thermo Omnic .spa file.

    Parameters
    ----------
    filepath : str or Path
        Path to .spa file

    Returns
    -------
    spectrum : pd.Series
        Spectral data with wavenumbers/wavelengths as index
    metadata : dict
        File metadata including data_type, x_unit, source_data_type
    """
    OMNICReader = _import_omnic_reader()
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")

    try:
        r = OMNICReader(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to read Omnic file {filepath.name}: {e}")

    # Extract core data
    x_data = np.array(r.x)
    y_data = np.array(r.data)

    if x_data.size == 0 or y_data.size == 0:
        raise ValueError(f"Empty spectral data in {filepath.name}")

    # Flatten if single spectrum stored as 2D
    if y_data.ndim > 1:
        if y_data.shape[0] == 1:
            y_data = y_data.flatten()
        else:
            print(
                f"Warning: {filepath.name} contains {y_data.shape[0]} spectra. "
                f"Using first spectrum only. Use read_spg_file() for multi-spectrum files."
            )
            y_data = y_data[0]

    if len(x_data) != len(y_data):
        raise ValueError(
            f"Mismatched data lengths in {filepath.name}: "
            f"x={len(x_data)}, y={len(y_data)}"
        )

    # Map units
    units_str = getattr(r, 'units', None)
    data_type, source_data_type = _map_data_type(units_str)

    x_units_str = getattr(r, 'x_units', None)
    x_unit, needs_um_conversion = _map_x_unit(x_units_str)

    # Convert micrometers to nm if needed
    if needs_um_conversion:
        x_data = x_data * 1000.0

    # Ensure ascending order
    if len(x_data) > 1 and x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    # Create Series
    spectrum = pd.Series(y_data, index=x_data)
    spectrum = spectrum[~spectrum.index.duplicated(keep='first')]

    # Extract optional metadata safely
    name = getattr(r, 'name', None) or filepath.stem
    y_labels = getattr(r, 'y_labels', None)
    history = getattr(r, 'history', None)

    acquisition_date = None
    if y_labels and isinstance(y_labels, (list, tuple)) and len(y_labels) > 0:
        try:
            dates = y_labels[0]
            if dates and len(dates) > 0:
                acquisition_date = str(dates[0])
        except (IndexError, TypeError):
            pass

    metadata = {
        'filename': filepath.name,
        'data_type': data_type,
        'source_data_type': source_data_type,
        'x_unit': x_unit,
        'x_range': (float(x_data.min()), float(x_data.max())),
        'n_points': len(spectrum),
        'file_format': 'omnic',
        'vendor': 'Thermo',
        'sample_name': name,
    }

    if acquisition_date:
        metadata['acquisition_date'] = acquisition_date
    if history:
        metadata['processing_history'] = history

    # Optional instrument metadata
    for attr in ('laser_frequency', 'collection_length', 'optical_velocity'):
        val = getattr(r, attr, None)
        if val is not None:
            metadata[attr] = val

    return spectrum, metadata


def read_spg_file(filepath: str | Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Read a Thermo Omnic .spg (spectral group) file containing multiple spectra.

    Parameters
    ----------
    filepath : str or Path
        Path to .spg file

    Returns
    -------
    df : pd.DataFrame
        Wide matrix with rows = spectra, columns = x-axis values
    metadata : dict
        File metadata including n_spectra, data_type, x_unit
    """
    OMNICReader = _import_omnic_reader()
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")

    try:
        r = OMNICReader(str(filepath))
    except Exception as e:
        raise ValueError(f"Failed to read Omnic SPG file {filepath.name}: {e}")

    x_data = np.array(r.x)
    y_data = np.array(r.data)

    if x_data.size == 0 or y_data.size == 0:
        raise ValueError(f"Empty spectral data in {filepath.name}")

    # Ensure 2D
    if y_data.ndim == 1:
        y_data = y_data.reshape(1, -1)

    n_spectra = y_data.shape[0]

    if n_spectra > 500:
        print(
            f"Warning: {filepath.name} contains {n_spectra} spectra — "
            f"this may use significant memory."
        )

    # Map units
    units_str = getattr(r, 'units', None)
    data_type, source_data_type = _map_data_type(units_str)

    x_units_str = getattr(r, 'x_units', None)
    x_unit, needs_um_conversion = _map_x_unit(x_units_str)

    if needs_um_conversion:
        x_data = x_data * 1000.0

    # Build row names from y_labels if available, else use stem_i
    stem = filepath.stem
    y_labels = getattr(r, 'y_labels', None)
    row_names = []

    if y_labels and isinstance(y_labels, (list, tuple)) and len(y_labels) > 1:
        try:
            names_list = y_labels[1]
            if names_list and len(names_list) == n_spectra:
                row_names = [str(n) for n in names_list]
        except (IndexError, TypeError):
            pass

    if len(row_names) != n_spectra:
        row_names = [f"{stem}_{i}" for i in range(n_spectra)]

    # Build DataFrame
    df = pd.DataFrame(y_data, columns=x_data, index=row_names)

    # Sort columns ascending
    df = df[sorted(df.columns)]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    metadata = {
        'filename': filepath.name,
        'n_spectra': n_spectra,
        'data_type': data_type,
        'source_data_type': source_data_type,
        'x_unit': x_unit,
        'x_range': (float(df.columns.min()), float(df.columns.max())),
        'n_points': df.shape[1],
        'file_format': 'omnic',
        'vendor': 'Thermo',
    }

    return df, metadata


def read_omnic_dir(directory: str | Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Read all Thermo Omnic .spa and .spg files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing .spa and/or .spg files

    Returns
    -------
    df : pd.DataFrame
        Wide matrix with rows = spectra, columns = x-axis values
    metadata : dict
        Aggregated metadata including n_spectra, data_type, x_unit
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Find all Omnic files (case-insensitive)
    spa_files = sorted(set(
        list(directory.glob("*.spa")) + list(directory.glob("*.SPA"))
    ))
    spg_files = sorted(set(
        list(directory.glob("*.spg")) + list(directory.glob("*.SPG"))
    ))

    all_files = spa_files + spg_files
    if len(all_files) == 0:
        raise ValueError(f"No .spa or .spg files found in {directory}")

    print(f"Found {len(spa_files)} .spa and {len(spg_files)} .spg files")

    spectra = {}
    data_types = []
    x_units = []
    failed_files = []
    duplicate_names = []

    # Process .spg files first (multi-spectrum)
    for spg_file in spg_files:
        try:
            df_spg, spg_meta = read_spg_file(spg_file)
            for row_name in df_spg.index:
                if row_name in spectra:
                    duplicate_names.append(row_name)
                spectra[row_name] = df_spg.loc[row_name]
            data_types.append(spg_meta.get('data_type', 'unknown'))
            x_units.append(spg_meta.get('x_unit', 'unknown'))
        except Exception as e:
            print(f"Warning: Could not read {spg_file.name}: {e}")
            failed_files.append(spg_file.name)

    # Process .spa files
    for spa_file in spa_files:
        stem = spa_file.stem
        if stem in spectra:
            duplicate_names.append(stem)
            print(f"Warning: Duplicate name '{stem}' — later file overwrites earlier one")

        try:
            spectrum, file_meta = read_spa_file(spa_file)
            spectra[stem] = spectrum
            data_types.append(file_meta.get('data_type', 'unknown'))
            x_units.append(file_meta.get('x_unit', 'unknown'))
        except Exception as e:
            print(f"Warning: Could not read {spa_file.name}: {e}")
            failed_files.append(spa_file.name)

    if len(spectra) == 0:
        error_msg = f"No valid Omnic spectra could be read from {directory}"
        if failed_files:
            error_msg += f"\nFailed files: {failed_files[:5]}"
        raise ValueError(error_msg)

    if duplicate_names:
        print(
            f"\nWarning: Found {len(set(duplicate_names))} duplicate spectrum names. "
            f"Only the last occurrence of each is kept."
        )

    # Combine into DataFrame
    df = pd.DataFrame(spectra).T
    df = df[sorted(df.columns)]

    # Validate
    if df.shape[1] < 50:
        print(
            f"Warning: Only {df.shape[1]} data points found. "
            f"This is unusually low for FTIR spectroscopy."
        )

    x_values = np.array(df.columns)
    if len(x_values) > 1 and not np.all(x_values[1:] > x_values[:-1]):
        print("Warning: X-axis values were not strictly increasing after sorting.")

    # Determine dominant units
    from collections import Counter
    type_counts = Counter(data_types)
    dominant_type = type_counts.most_common(1)[0][0] if type_counts else 'unknown'
    unit_counts = Counter(x_units)
    dominant_unit = unit_counts.most_common(1)[0][0] if unit_counts else 'cm-1'

    if len(unit_counts) > 1:
        print(
            f"Warning: Files have inconsistent x-axis units: {dict(unit_counts)}\n"
            f"Proceeding with dominant unit: {dominant_unit}"
        )

    metadata = {
        'n_spectra': len(df),
        'wavelength_range': (float(df.columns.min()), float(df.columns.max())),
        'file_format': 'omnic',
        'data_types': dict(type_counts),
        'dominant_data_type': dominant_type,
        'x_unit': dominant_unit,
        'x_unit_counts': dict(unit_counts),
        'n_failed': len(failed_files),
        'failed_files': failed_files[:10] if failed_files else [],
    }

    print(f"Successfully read {len(df)} Omnic spectra")
    print(f"X-axis range: {metadata['wavelength_range'][0]:.1f} - {metadata['wavelength_range'][1]:.1f} {dominant_unit}")
    print(f"Data types: {dict(type_counts)}")

    return df, metadata
