"""
I/O utilities for v3.

Provides type-safe interfaces for spectral data I/O.
V3 is now standalone - no V1 dependency.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np
import pandas as pd

# Import v3 io module (standalone)
from . import io

# Import v3 types
from .types import SpectralDataset, LoadResult, MergeResult


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect file format from extension and/or content.

    Parameters
    ----------
    path : str or Path
        Path to file or directory

    Returns
    -------
    str
        Format identifier: 'csv', 'excel', 'asd', 'spc', 'jcamp',
        'ascii', 'opus', 'perkinelmer', 'agilent', 'directory', 'unknown'
    """
    return io.detect_format(path)


def read_csv_spectra(path: Union[str, Path]) -> LoadResult:
    """
    Read spectral data from CSV file.

    Supports wide format (rows=samples, cols=wavelengths) and
    long format (wavelength, value columns).

    Parameters
    ----------
    path : str or Path
        Path to CSV file

    Returns
    -------
    LoadResult
        Contains SpectralDataset and loading metadata
    """
    df, metadata = io.read_csv_spectra(path)

    # Convert DataFrame to SpectralDataset
    wavelengths = np.array(df.columns, dtype=float)
    X = df.values.astype(float)
    sample_ids = list(df.index.astype(str))

    dataset = SpectralDataset(
        X=X,
        wavelengths=wavelengths,
        sample_ids=sample_ids,
        metadata=metadata
    )

    return LoadResult(
        dataset=dataset,
        format_detected='csv',
        warnings=[]
    )


def read_reference(
    path: Union[str, Path],
    id_column: str
) -> pd.DataFrame:
    """
    Read reference file with target variables.

    Parameters
    ----------
    path : str or Path
        Path to reference file (CSV or Excel)
    id_column : str
        Column name to use as index

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by id_column
    """
    return io.read_reference_csv(path, id_column)


def align_with_reference(
    dataset: SpectralDataset,
    ref_path: Union[str, Path],
    id_column: str,
    target_column: str
) -> MergeResult:
    """
    Align spectral data with reference file using smart matching.

    Uses fuzzy filename matching to handle:
    - Files with/without extensions
    - Filenames with/without spaces
    - Case differences

    Parameters
    ----------
    dataset : SpectralDataset
        Spectral data to align
    ref_path : str or Path
        Path to reference file
    id_column : str
        Column in reference file containing sample IDs
    target_column : str
        Column in reference file containing target values

    Returns
    -------
    MergeResult
        Contains aligned SpectralDataset and merge statistics
    """
    # Convert SpectralDataset back to DataFrame for v1 function
    X_df = pd.DataFrame(
        dataset.X,
        index=dataset.sample_ids,
        columns=dataset.wavelengths
    )

    # Read reference
    ref_df = read_reference(ref_path, id_column)

    # Use v1's align_xy with detailed info
    X_aligned, y, info = io.align_xy(
        X_df, ref_df, id_column, target_column,
        return_alignment_info=True
    )

    # Convert back to SpectralDataset
    aligned_dataset = SpectralDataset(
        X=X_aligned.values.astype(float),
        wavelengths=np.array(X_aligned.columns, dtype=float),
        sample_ids=list(X_aligned.index.astype(str)),
        y=y.values.astype(float),
        target_name=target_column,
        metadata={
            **dataset.metadata,
            'reference_file': str(ref_path),
            'id_column': id_column,
        }
    )

    # Build merge result
    warnings = []
    if info.get('unmatched_spectral'):
        n = len(info['unmatched_spectral'])
        warnings.append(f"{n} spectral samples had no reference match")
    if info.get('unmatched_reference'):
        n = len(info['unmatched_reference'])
        warnings.append(f"{n} reference samples had no spectral data")
    if info.get('n_nan_dropped', 0) > 0:
        warnings.append(f"{info['n_nan_dropped']} samples dropped due to missing target values")

    return MergeResult(
        dataset=aligned_dataset,
        n_matched=len(aligned_dataset.sample_ids),
        n_unmatched_spectra=len(info.get('unmatched_spectral', [])),
        n_unmatched_reference=len(info.get('unmatched_reference', [])),
        used_fuzzy_matching=info.get('used_fuzzy_matching', False),
        warnings=warnings
    )


def find_reference_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all potential reference files in a directory.

    Searches for CSV/Excel files that could contain reference data.

    Parameters
    ----------
    directory : str or Path
        Directory to search

    Returns
    -------
    List[Path]
        List of potential reference files (CSV/Excel), sorted with
        priority matches first
    """
    import re

    directory = Path(directory)
    if not directory.is_dir():
        directory = directory.parent

    # Priority patterns (most likely to be reference files)
    priority_patterns = [
        r'.*ref.*\.csv$',
        r'.*reference.*\.csv$',
        r'.*lab.*\.csv$',
        r'.*target.*\.csv$',
        r'.*metadata.*\.csv$',
        r'.*ref.*\.xlsx$',
        r'.*reference.*\.xlsx$',
    ]

    # Find priority matches
    priority_candidates = []
    for pattern in priority_patterns:
        for file in directory.iterdir():
            if file.is_file() and re.match(pattern, file.name, re.IGNORECASE):
                priority_candidates.append(file)

    priority_candidates = list(set(priority_candidates))

    # Find ALL CSV/Excel files
    all_csv_xlsx = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            all_csv_xlsx.append(file)

    # Sort: priority files first, then others alphabetically
    other_files = [f for f in all_csv_xlsx if f not in priority_candidates]
    return priority_candidates + sorted(other_files, key=lambda x: x.name)


def find_reference_file(directory: Union[str, Path]) -> Optional[Path]:
    """
    Find a single reference file in a directory.

    Returns the file if exactly one found, None otherwise.
    For multiple files, use find_reference_files() and prompt user.

    Parameters
    ----------
    directory : str or Path
        Directory to search

    Returns
    -------
    Path or None
        Path to reference file if exactly one found
    """
    files = find_reference_files(directory)
    if len(files) == 1:
        return files[0]
    return None


def load_spectra_folder(folder_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all spectral files from a folder.

    Wraps v1's read_spectra() function with auto-detection of file formats.
    Supports ASD, SPC, JCAMP, ASCII, OPUS, and other spectral formats.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing spectral files

    Returns
    -------
    X : np.ndarray
        Spectral data matrix, shape (n_samples, n_features)
    wavelengths : np.ndarray
        Wavelengths corresponding to spectral features
    sample_ids : List[str]
        Sample identifiers (typically filenames without extensions)

    Raises
    ------
    FileNotFoundError
        If folder does not exist
    ValueError
        If no valid spectral files found in folder
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Use v1's read_spectra with auto format detection
    df, metadata = io.read_spectra(str(folder_path), format='auto')

    if df is None or len(df) == 0:
        raise ValueError(f"No valid spectral files found in folder: {folder_path}")

    # Convert DataFrame to arrays
    wavelengths = np.array(df.columns, dtype=float)
    X = df.values.astype(float)
    sample_ids = list(df.index.astype(str))

    return X, wavelengths, sample_ids


def load_spectra(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Universal spectral data loader - handles both files and folders.

    Automatically detects whether the path is a file or folder and uses
    the appropriate loading method.

    Parameters
    ----------
    path : str or Path
        Path to spectral file or folder

    Returns
    -------
    X : np.ndarray
        Spectral data matrix, shape (n_samples, n_features)
    wavelengths : np.ndarray
        Wavelengths corresponding to spectral features
    sample_ids : List[str]
        Sample identifiers

    Raises
    ------
    FileNotFoundError
        If path does not exist
    ValueError
        If path is neither a valid file nor folder, or contains no valid data
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        return load_spectra_folder(path)
    else:
        # Load single file using v1's read_spectra
        df, metadata = io.read_spectra(str(path), format='auto')

        if df is None or len(df) == 0:
            raise ValueError(f"Could not load spectral data from file: {path}")

        # Convert DataFrame to arrays
        wavelengths = np.array(df.columns, dtype=float)
        X = df.values.astype(float)
        sample_ids = list(df.index.astype(str))

        return X, wavelengths, sample_ids


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect column types in a DataFrame.

    Identifies:
    - Potential ID columns (string/object with unique values)
    - Potential target columns (numeric, non-wavelength)
    - Wavelength columns (numeric names that form a range)
    - Metadata columns (other columns)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    dict
        {
            'id_candidates': list of column names,
            'target_candidates': list of column names,
            'wavelength_columns': list of column names,
            'metadata_columns': list of column names,
        }
    """
    result = {
        'id_candidates': [],
        'target_candidates': [],
        'wavelength_columns': [],
        'metadata_columns': []
    }

    wavelength_cols = []
    for col in df.columns:
        try:
            wl = float(col)
            # Plausible wavelength range (UV-VIS-NIR-MIR)
            if 200 <= wl <= 25000:
                wavelength_cols.append(col)
        except (ValueError, TypeError):
            pass

    result['wavelength_columns'] = wavelength_cols

    # Non-wavelength columns
    other_cols = [c for c in df.columns if c not in wavelength_cols]

    for col in other_cols:
        # Check if potential ID column
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            if df[col].nunique() == len(df):  # All unique
                result['id_candidates'].append(col)
            elif df[col].nunique() <= 50:  # Few unique values = potential categorical target
                result['target_candidates'].append(col)
                result['metadata_columns'].append(col)  # Also metadata
            else:
                result['metadata_columns'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric non-wavelength column = potential target AND metadata
            result['target_candidates'].append(col)
            result['metadata_columns'].append(col)  # Also show in metadata
        else:
            result['metadata_columns'].append(col)

    return result
