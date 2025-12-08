"""
Main backend engine for Spectral Predict v3.

Provides high-level API for data loading, merging, and analysis.
Delegates to v1 modules for proven functionality.
"""

import sys
from pathlib import Path
from typing import Union, Optional, Callable, List
import numpy as np
import pandas as pd

# V3 is now standalone - no V1 dependency

# Import v3 types and utilities
from .types import SpectralDataset, LoadResult, MergeResult
from . import io_utils


class Engine:
    """
    Main backend API for Spectral Predict v3.

    Provides unified interface for:
    - Loading spectral data from various formats
    - Merging with reference files
    - Running model search
    - Applying preprocessing

    Example
    -------
    >>> engine = Engine()
    >>> result = engine.load_file("data/spectra.csv")
    >>> merged = engine.merge_with_reference(
    ...     result.dataset,
    ...     "data/reference.csv",
    ...     id_column="Sample_ID",
    ...     target_column="Moisture"
    ... )
    """

    def __init__(self):
        """Initialize the engine."""
        self._search_module = None
        self._preprocess_module = None

    def load_file(self, path: Union[str, Path]) -> LoadResult:
        """
        Load spectral data from a file.

        Automatically detects format and uses appropriate reader.

        Parameters
        ----------
        path : str or Path
            Path to spectral file

        Returns
        -------
        LoadResult
            Contains SpectralDataset and loading metadata

        Raises
        ------
        ValueError
            If format cannot be detected or file cannot be read
        """
        path = Path(path)
        format_type = io_utils.detect_format(path)

        if format_type == 'csv':
            return io_utils.read_csv_spectra(path)
        elif format_type == 'excel':
            return self._load_excel(path)
        elif format_type == 'directory':
            return self._load_directory(path)
        elif format_type == 'asd':
            return self._load_asd(path)
        else:
            raise ValueError(
                f"Unsupported format: {format_type} for file {path}. "
                f"Supported: csv, excel, asd, directory"
            )

    def _load_excel(self, path: Path) -> LoadResult:
        """Load from Excel file."""
        df = pd.read_excel(path)

        # Detect columns
        col_info = io_utils.detect_columns(df)

        if not col_info['wavelength_columns']:
            raise ValueError(
                f"No wavelength columns detected in {path}. "
                f"Expected numeric column names in range 200-25000 nm."
            )

        # Use first ID candidate as index, or first column
        if col_info['id_candidates']:
            id_col = col_info['id_candidates'][0]
        else:
            id_col = df.columns[0]

        # Extract spectral data
        wl_cols = col_info['wavelength_columns']
        warnings = []

        # Convert to numeric, handling non-numeric values
        X_df = df[wl_cols].apply(pd.to_numeric, errors='coerce')

        # Count and warn about missing values BEFORE imputation
        nan_count = X_df.isna().sum().sum()
        if nan_count > 0:
            n_samples_affected = X_df.isna().any(axis=1).sum()
            warnings.append(
                f"Warning: {nan_count} missing/non-numeric values in {n_samples_affected} samples "
                f"were replaced with column means"
            )

        # Impute missing values
        X_df = X_df.fillna(X_df.mean()).fillna(0)
        X = X_df.values.astype(float)
        wavelengths = np.array([float(c) for c in wl_cols])
        sample_ids = list(df[id_col].astype(str))

        # Extract metadata columns
        metadata_columns = {}
        for col in col_info['metadata_columns'] + col_info['target_candidates']:
            if col != id_col:
                metadata_columns[col] = list(df[col])

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata={'file_format': 'excel'},
            metadata_columns=metadata_columns
        )

        return LoadResult(
            dataset=dataset,
            format_detected='excel',
            warnings=warnings
        )

    def _load_directory(self, path: Path) -> LoadResult:
        """Load spectral files from a directory."""
        from . import io as v3_io
        import pandas as pd

        # Use v3's read_spectra which handles directory detection
        try:
            df, metadata = v3_io.read_spectra(path)
            warnings = []

            # Convert to numeric, handling non-numeric values
            df_numeric = df.apply(pd.to_numeric, errors='coerce')

            # Count and warn about missing values BEFORE imputation
            nan_count = df_numeric.isna().sum().sum()
            if nan_count > 0:
                n_samples_affected = df_numeric.isna().any(axis=1).sum()
                warnings.append(
                    f"Warning: {nan_count} missing/non-numeric values in {n_samples_affected} samples "
                    f"were replaced with column means"
                )

            df_numeric = df_numeric.fillna(df_numeric.mean()).fillna(0)

            wavelengths = np.array([float(c) for c in df.columns])
            X = df_numeric.values.astype(float)
            sample_ids = list(df.index.astype(str))

            dataset = SpectralDataset(
                X=X,
                wavelengths=wavelengths,
                sample_ids=sample_ids,
                metadata=metadata
            )

            return LoadResult(
                dataset=dataset,
                format_detected=metadata.get('file_format', 'directory'),
                warnings=warnings
            )
        except Exception as e:
            raise ValueError(f"Could not load directory {path}: {e}")

    def _load_asd_directory(self, path: Path) -> LoadResult:
        """Load ASD files from directory using v3's io.read_asd_dir."""
        from . import io as v3_io
        import pandas as pd

        # Use v3's read_asd_dir which handles ASCII and binary ASD files
        df, metadata = v3_io.read_asd_dir(path)
        warnings = []

        # Convert to numeric, handling non-numeric values
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Count and warn about missing values BEFORE imputation
        nan_count = df_numeric.isna().sum().sum()
        if nan_count > 0:
            n_samples_affected = df_numeric.isna().any(axis=1).sum()
            warnings.append(
                f"Warning: {nan_count} missing/non-numeric values in {n_samples_affected} samples "
                f"were replaced with column means"
            )

        df_numeric = df_numeric.fillna(df_numeric.mean()).fillna(0)

        wavelengths = np.array([float(c) for c in df.columns])
        X = df_numeric.values.astype(float)
        sample_ids = list(df.index.astype(str))

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata={**metadata, 'file_format': 'asd'}
        )

        return LoadResult(
            dataset=dataset,
            format_detected='asd',
            warnings=warnings
        )

    def _load_asd(self, path: Path) -> LoadResult:
        """Load single ASD file."""
        # If single file, load via directory reader
        return self._load_asd_directory(path.parent)

    def _load_csv_directory(self, path: Path, csv_files: List[Path]) -> LoadResult:
        """Load and combine multiple CSV files."""
        datasets = []
        for csv_path in csv_files:
            result = io_utils.read_csv_spectra(csv_path)
            datasets.append(result.dataset)

        # Combine datasets
        if len(datasets) == 1:
            return LoadResult(
                dataset=datasets[0],
                format_detected='csv',
                warnings=[]
            )

        # Stack vertically (assumes same wavelengths)
        combined_X = np.vstack([d.X for d in datasets])
        combined_ids = []
        for d in datasets:
            combined_ids.extend(d.sample_ids)

        dataset = SpectralDataset(
            X=combined_X,
            wavelengths=datasets[0].wavelengths,
            sample_ids=combined_ids,
            metadata={'file_format': 'csv', 'n_files': len(csv_files)}
        )

        return LoadResult(
            dataset=dataset,
            format_detected='csv',
            warnings=[f"Combined {len(csv_files)} CSV files"]
        )

    def merge_with_reference(
        self,
        dataset: SpectralDataset,
        ref_path: Union[str, Path],
        id_column: str,
        target_column: str
    ) -> MergeResult:
        """
        Merge spectral data with reference file.

        Uses smart filename matching for flexible ID alignment.

        Parameters
        ----------
        dataset : SpectralDataset
            Spectral data to merge
        ref_path : str or Path
            Path to reference file (CSV or Excel)
        id_column : str
            Column in reference containing sample IDs
        target_column : str
            Column in reference containing target values

        Returns
        -------
        MergeResult
            Contains merged dataset and statistics
        """
        return io_utils.align_with_reference(
            dataset, ref_path, id_column, target_column
        )

    def find_reference(self, directory: Union[str, Path]) -> Optional[Path]:
        """
        Find reference file in a directory.

        Parameters
        ----------
        directory : str or Path
            Directory to search

        Returns
        -------
        Path or None
            Path to reference file if unambiguous match found
        """
        return io_utils.find_reference_file(directory)

    def run_search(
        self,
        dataset: SpectralDataset,
        task_type: str = "regression",
        folds: int = 5,
        tier: str = "standard",
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Run model search on dataset.

        Parameters
        ----------
        dataset : SpectralDataset
            Dataset with target values
        task_type : str
            'regression' or 'classification'
        folds : int
            Number of CV folds
        tier : str
            Search tier: 'quick', 'standard', 'comprehensive'
        progress_callback : callable, optional
            Function(progress_pct: float, message: str)

        Returns
        -------
        pd.DataFrame
            Results with columns: model, preprocessing, metrics, etc.
        """
        if not dataset.has_target:
            raise ValueError("Dataset must have target values for model search")

        # Lazy import v3 search
        if self._search_module is None:
            from . import search
            self._search_module = search

        # Convert to DataFrame format expected by search module
        X_df = pd.DataFrame(
            dataset.X,
            index=dataset.sample_ids,
            columns=dataset.wavelengths
        )
        y_series = pd.Series(dataset.y, index=dataset.sample_ids)

        # Wrap progress callback
        def v1_progress(info: dict):
            if progress_callback:
                pct = info.get('current', 0) / max(info.get('total', 1), 1)
                msg = info.get('message', '')
                progress_callback(pct, msg)

        # Run search
        results = self._search_module.run_search(
            X_df, y_series,
            task_type=task_type,
            folds=folds,
            tier=tier,
            progress_callback=v1_progress if progress_callback else None
        )

        return results

    def apply_preprocessing(
        self,
        X: np.ndarray,
        method: str,
        wavelengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        X : np.ndarray
            Spectral data (n_samples, n_wavelengths)
        method : str
            Preprocessing method: 'snv', 'sg1', 'sg2', 'msc', 'raw'
        wavelengths : np.ndarray, optional
            Wavelength axis (required for some methods)

        Returns
        -------
        np.ndarray
            Preprocessed data
        """
        # Lazy import v3 preprocess
        if self._preprocess_module is None:
            from . import preprocess
            self._preprocess_module = preprocess

        if method == 'raw' or method is None:
            return X

        if method == 'snv':
            transformer = self._preprocess_module.SNV()
            return transformer.fit_transform(X)

        if method in ['sg1', 'sg2']:
            deriv = 1 if method == 'sg1' else 2
            transformer = self._preprocess_module.SavgolDerivative(
                deriv=deriv, window=7
            )
            return transformer.fit_transform(X)

        raise ValueError(f"Unknown preprocessing method: {method}")
