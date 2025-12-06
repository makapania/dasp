"""Type definitions for Spectral Predict v3."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class SpectralDataset:
    """Container for loaded spectral data.

    Attributes
    ----------
    X : np.ndarray
        Spectral data matrix (n_samples, n_wavelengths)
    wavelengths : np.ndarray
        Wavelength axis (n_wavelengths,)
    sample_ids : list[str]
        Sample identifiers
    y : np.ndarray, optional
        Target values (n_samples,)
    target_name : str, optional
        Name of target variable
    metadata : dict
        Additional metadata (data_type, file_format, etc.)
    metadata_columns : dict[str, list]
        Additional metadata columns {column_name: [values]}
    """
    X: np.ndarray
    wavelengths: np.ndarray
    sample_ids: List[str]
    y: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metadata_columns: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return self.X.shape[1]

    @property
    def wavelength_range(self) -> tuple:
        """(min_wavelength, max_wavelength) in nm."""
        return (float(self.wavelengths.min()), float(self.wavelengths.max()))

    @property
    def has_target(self) -> bool:
        """Whether target values are present."""
        return self.y is not None and len(self.y) > 0

    def copy(self) -> 'SpectralDataset':
        """Create a deep copy."""
        return SpectralDataset(
            X=self.X.copy(),
            wavelengths=self.wavelengths.copy(),
            sample_ids=self.sample_ids.copy(),
            y=self.y.copy() if self.y is not None else None,
            target_name=self.target_name,
            metadata=self.metadata.copy(),
            metadata_columns={k: v.copy() for k, v in self.metadata_columns.items()}
        )


@dataclass
class LoadResult:
    """Result of loading a file.

    Attributes
    ----------
    dataset : SpectralDataset
        Loaded data
    format_detected : str
        Detected file format
    warnings : list[str]
        Any warnings during loading
    """
    dataset: SpectralDataset
    format_detected: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class MergeResult:
    """Result of merging spectral data with reference.

    Attributes
    ----------
    dataset : SpectralDataset
        Merged dataset
    n_matched : int
        Number of samples successfully matched
    n_unmatched_spectra : int
        Spectral samples with no reference match
    n_unmatched_reference : int
        Reference samples with no spectral match
    used_fuzzy_matching : bool
        Whether fuzzy filename matching was used
    warnings : list[str]
        Any warnings during merge
    """
    dataset: SpectralDataset
    n_matched: int
    n_unmatched_spectra: int
    n_unmatched_reference: int
    used_fuzzy_matching: bool
    warnings: List[str] = field(default_factory=list)
