"""
Spectral Library Search - Local library management and similarity search.

This module provides functionality for building and searching a local spectral
library that grows over time as users load new spectra. Key features:
- Persistent storage across sessions
- Automatic duplicate detection
- Multiple similarity metrics (HQI, SAM, Euclidean, etc.)
- Wavelength alignment for spectra from different instruments
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .similarity_metrics import (
    compute_similarity,
    compute_batch_similarity,
    hit_quality_index,
    METRICS,
)

logger = logging.getLogger(__name__)

# Default library storage location
try:
    from platformdirs import user_data_dir
    DEFAULT_LIBRARY_DIR = Path(user_data_dir("spectral_predict", "SpectralPredict")) / "library"
except ImportError:
    # Fallback if platformdirs not installed
    DEFAULT_LIBRARY_DIR = Path.home() / ".spectral_predict" / "library"


@dataclass
class LibraryEntry:
    """A single entry in the spectral library."""
    sample_id: str
    spectrum: np.ndarray
    wavelengths: np.ndarray
    source_file: str = ""
    date_added: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""  # Hash for duplicate detection

    def __post_init__(self):
        if not self.date_added:
            self.date_added = datetime.now().isoformat()
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute a hash fingerprint of the spectrum for duplicate detection."""
        # Use rounded spectrum values to handle floating point variations
        rounded = np.round(self.spectrum, decimals=6)
        return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


class SpectralLibrary:
    """
    A persistent spectral library that grows as users load new spectra.

    Features:
    - Automatic persistence to disk
    - Duplicate detection by sample ID and spectral fingerprint
    - Multiple similarity metrics for searching
    - Wavelength alignment for comparing spectra from different instruments

    Parameters
    ----------
    name : str
        Name of the library
    storage_path : Path, optional
        Path to store library files. Defaults to user data directory.
    auto_save : bool
        Whether to automatically save after modifications
    duplicate_threshold : float
        HQI threshold for spectral duplicate detection (default 0.9999)
    """

    def __init__(
        self,
        name: str = "local",
        storage_path: Optional[Path] = None,
        auto_save: bool = True,
        duplicate_threshold: float = 0.9999,
    ):
        self.name = name
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_LIBRARY_DIR
        self.auto_save = auto_save
        self.duplicate_threshold = duplicate_threshold

        # Internal storage
        self._entries: Dict[str, LibraryEntry] = {}  # sample_id -> entry
        self._fingerprints: Dict[str, str] = {}  # fingerprint -> sample_id
        self._wavelength_grid: Optional[np.ndarray] = None  # Common wavelength grid
        self._spectra_matrix: Optional[np.ndarray] = None  # Cached matrix for fast search
        self._matrix_dirty: bool = True  # Whether matrix needs rebuild

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Try to load existing library
        self._load()

    @property
    def size(self) -> int:
        """Number of entries in the library."""
        return len(self._entries)

    @property
    def sample_ids(self) -> List[str]:
        """List of all sample IDs in the library."""
        return list(self._entries.keys())

    @property
    def wavelengths(self) -> Optional[np.ndarray]:
        """Common wavelength grid for the library."""
        return self._wavelength_grid

    @property
    def categories(self) -> List[str]:
        """List of unique categories in the library."""
        cats = set()
        for entry in self._entries.values():
            if entry.category:
                cats.add(entry.category)
        return sorted(cats)

    def add_spectrum(
        self,
        sample_id: str,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        source_file: str = "",
        category: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Add a spectrum to the library if not a duplicate.

        Parameters
        ----------
        sample_id : str
            Unique identifier for the sample
        spectrum : np.ndarray
            Spectral intensity values
        wavelengths : np.ndarray
            Wavelength values corresponding to spectrum
        source_file : str
            Source file path for reference
        category : str
            Category/class label for the spectrum
        metadata : dict, optional
            Additional metadata
        force : bool
            If True, add even if duplicate detected

        Returns
        -------
        Tuple[bool, str]
            (success, message) - success is True if added, False if duplicate
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        wavelengths = np.asarray(wavelengths, dtype=np.float64)

        if len(spectrum) != len(wavelengths):
            return False, f"Spectrum length ({len(spectrum)}) != wavelengths length ({len(wavelengths)})"

        # Check for duplicate by sample ID
        if sample_id in self._entries and not force:
            return False, f"Sample ID '{sample_id}' already exists in library"

        # Create entry to get fingerprint
        entry = LibraryEntry(
            sample_id=sample_id,
            spectrum=spectrum,
            wavelengths=wavelengths,
            source_file=source_file,
            category=category,
            metadata=metadata or {},
        )

        # Check for duplicate by fingerprint
        if entry.fingerprint in self._fingerprints and not force:
            existing_id = self._fingerprints[entry.fingerprint]
            return False, f"Spectrum matches existing sample '{existing_id}' (fingerprint match)"

        # Check for near-duplicate by spectral similarity
        if self.size > 0 and not force:
            is_dup, similar_id, score = self._check_spectral_duplicate(spectrum, wavelengths)
            if is_dup:
                return False, f"Spectrum very similar to '{similar_id}' (HQI={score:.4f})"

        # Add to library
        self._entries[sample_id] = entry
        self._fingerprints[entry.fingerprint] = sample_id
        self._matrix_dirty = True

        # Update wavelength grid if needed
        if self._wavelength_grid is None:
            self._wavelength_grid = wavelengths.copy()
        elif not np.allclose(self._wavelength_grid, wavelengths, rtol=1e-5):
            # Different wavelength grid - will need interpolation for search
            logger.debug(f"New spectrum has different wavelength grid than library")

        if self.auto_save:
            self._save()

        return True, f"Added '{sample_id}' to library"

    def add_spectra_batch(
        self,
        df: pd.DataFrame,
        source_file: str = "",
        category: str = "",
    ) -> Tuple[int, int, List[str]]:
        """
        Add multiple spectra from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sample IDs as index and wavelengths as columns
        source_file : str
            Source file path
        category : str
            Category for all spectra

        Returns
        -------
        Tuple[int, int, List[str]]
            (added_count, skipped_count, messages)
        """
        added = 0
        skipped = 0
        messages = []

        wavelengths = df.columns.astype(float).values

        # Temporarily disable auto-save for batch operation
        old_auto_save = self.auto_save
        self.auto_save = False

        try:
            for sample_id in df.index:
                spectrum = df.loc[sample_id].values.astype(float)
                success, msg = self.add_spectrum(
                    sample_id=str(sample_id),
                    spectrum=spectrum,
                    wavelengths=wavelengths,
                    source_file=source_file,
                    category=category,
                )
                if success:
                    added += 1
                else:
                    skipped += 1
                    messages.append(msg)
        finally:
            self.auto_save = old_auto_save

        if self.auto_save:
            self._save()

        return added, skipped, messages

    def _check_spectral_duplicate(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if spectrum is a near-duplicate of existing entry.

        Returns (is_duplicate, similar_sample_id, similarity_score)
        """
        if self.size == 0:
            return False, None, 0.0

        # Align to common grid if needed
        aligned_spectrum = self._align_to_grid(spectrum, wavelengths)
        if aligned_spectrum is None:
            return False, None, 0.0

        # Check against all entries (could optimize with FAISS later)
        best_score = 0.0
        best_id = None

        for sample_id, entry in self._entries.items():
            entry_aligned = self._align_to_grid(entry.spectrum, entry.wavelengths)
            if entry_aligned is not None:
                score = hit_quality_index(aligned_spectrum, entry_aligned)
                if score > best_score:
                    best_score = score
                    best_id = sample_id

        is_dup = best_score >= self.duplicate_threshold
        return is_dup, best_id, best_score

    def _align_to_grid(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Align spectrum to the library's common wavelength grid."""
        if self._wavelength_grid is None:
            return spectrum

        # Check if wavelengths match (same length and values)
        if len(wavelengths) == len(self._wavelength_grid):
            if np.allclose(wavelengths, self._wavelength_grid, rtol=1e-5):
                return spectrum

        # Interpolate to common grid
        try:
            f = interp1d(
                wavelengths,
                spectrum,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate',
            )
            return f(self._wavelength_grid)
        except Exception as e:
            logger.warning(f"Failed to interpolate spectrum: {e}")
            return None

    def remove_spectrum(self, sample_id: str) -> bool:
        """Remove a spectrum from the library."""
        if sample_id not in self._entries:
            return False

        entry = self._entries.pop(sample_id)
        if entry.fingerprint in self._fingerprints:
            del self._fingerprints[entry.fingerprint]

        self._matrix_dirty = True

        if self.auto_save:
            self._save()

        return True

    def get_spectrum(self, sample_id: str) -> Optional[LibraryEntry]:
        """Get a library entry by sample ID."""
        return self._entries.get(sample_id)

    def search(
        self,
        query: np.ndarray,
        query_wavelengths: np.ndarray,
        metric: str = 'hqi',
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search the library for spectra most similar to the query.

        Parameters
        ----------
        query : np.ndarray
            Query spectrum
        query_wavelengths : np.ndarray
            Wavelengths for query spectrum
        metric : str
            Similarity metric: 'hqi', 'sam', 'euclidean', 'cosine',
            'deriv1_corr', 'deriv2_corr', 'sid'
        top_k : int
            Number of top matches to return
        category : str, optional
            Filter results by category

        Returns
        -------
        pd.DataFrame
            Top matches with columns: rank, sample_id, score, category, source_file
        """
        if self.size == 0:
            return pd.DataFrame(columns=['rank', 'sample_id', 'score', 'category', 'source_file'])

        # Align query to library grid
        aligned_query = self._align_to_grid(query, query_wavelengths)
        if aligned_query is None:
            logger.warning("Could not align query to library wavelength grid")
            return pd.DataFrame(columns=['rank', 'sample_id', 'score', 'category', 'source_file'])

        # Get entries to search (optionally filtered by category)
        entries_to_search = []
        for sample_id, entry in self._entries.items():
            if category is None or entry.category == category:
                entries_to_search.append((sample_id, entry))

        if not entries_to_search:
            return pd.DataFrame(columns=['rank', 'sample_id', 'score', 'category', 'source_file'])

        # Compute similarities
        results = []
        for sample_id, entry in entries_to_search:
            aligned_ref = self._align_to_grid(entry.spectrum, entry.wavelengths)
            if aligned_ref is not None:
                score = compute_similarity(aligned_query, aligned_ref, metric, normalize=True)
                results.append({
                    'sample_id': sample_id,
                    'score': score,
                    'category': entry.category,
                    'source_file': entry.source_file,
                })

        if not results:
            return pd.DataFrame(columns=['rank', 'sample_id', 'score', 'category', 'source_file'])

        # Sort by score (higher is better after normalization)
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False).head(top_k)
        df.insert(0, 'rank', range(1, len(df) + 1))
        df = df.reset_index(drop=True)

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        if self.size == 0:
            return {
                'total_entries': 0,
                'categories': [],
                'wavelength_range': None,
                'date_range': None,
            }

        categories = {}
        dates = []
        wl_ranges = []

        for entry in self._entries.values():
            cat = entry.category or 'uncategorized'
            categories[cat] = categories.get(cat, 0) + 1
            if entry.date_added:
                dates.append(entry.date_added)
            wl_ranges.append((entry.wavelengths.min(), entry.wavelengths.max()))

        return {
            'total_entries': self.size,
            'categories': categories,
            'wavelength_range': (
                min(r[0] for r in wl_ranges),
                max(r[1] for r in wl_ranges),
            ) if wl_ranges else None,
            'date_range': (min(dates), max(dates)) if dates else None,
        }

    def export_to_csv(self, filepath: Union[str, Path]) -> None:
        """Export library to CSV file."""
        if self.size == 0:
            raise ValueError("Library is empty")

        # Build DataFrame
        data = {}
        wavelengths = None

        for sample_id, entry in self._entries.items():
            if wavelengths is None:
                wavelengths = entry.wavelengths
            # Align to first spectrum's wavelengths
            aligned = self._align_to_grid(entry.spectrum, entry.wavelengths)
            if aligned is not None:
                data[sample_id] = aligned

        df = pd.DataFrame(data, index=wavelengths).T
        df.index.name = 'sample_id'
        df.to_csv(filepath)

    def clear(self) -> None:
        """Clear all entries from the library."""
        self._entries.clear()
        self._fingerprints.clear()
        self._matrix_dirty = True
        self._wavelength_grid = None
        self._spectra_matrix = None

        if self.auto_save:
            self._save()

    def _save(self) -> None:
        """Save library to disk."""
        library_file = self.storage_path / f"{self.name}_library.json"
        spectra_file = self.storage_path / f"{self.name}_spectra.npz"

        # Save metadata
        metadata = {
            'name': self.name,
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'entries': {},
        }

        # Collect spectra for numpy save
        spectra_dict = {}
        wavelengths_dict = {}

        for sample_id, entry in self._entries.items():
            metadata['entries'][sample_id] = {
                'source_file': entry.source_file,
                'date_added': entry.date_added,
                'category': entry.category,
                'metadata': entry.metadata,
                'fingerprint': entry.fingerprint,
            }
            spectra_dict[sample_id] = entry.spectrum
            wavelengths_dict[sample_id] = entry.wavelengths

        # Save common wavelength grid
        if self._wavelength_grid is not None:
            wavelengths_dict['__grid__'] = self._wavelength_grid

        # Write files
        with open(library_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        np.savez_compressed(spectra_file, **spectra_dict, **{f'wl_{k}': v for k, v in wavelengths_dict.items()})

        logger.info(f"Saved library '{self.name}' with {self.size} entries")

    def _load(self) -> None:
        """Load library from disk."""
        library_file = self.storage_path / f"{self.name}_library.json"
        spectra_file = self.storage_path / f"{self.name}_spectra.npz"

        if not library_file.exists() or not spectra_file.exists():
            logger.info(f"No existing library found at {self.storage_path}")
            return

        try:
            with open(library_file, 'r') as f:
                metadata = json.load(f)

            data = np.load(spectra_file, allow_pickle=True)

            # Load wavelength grid
            if 'wl___grid__' in data:
                self._wavelength_grid = data['wl___grid__']

            # Load entries
            for sample_id, entry_meta in metadata.get('entries', {}).items():
                if sample_id in data:
                    wl_key = f'wl_{sample_id}'
                    wavelengths = data[wl_key] if wl_key in data else self._wavelength_grid

                    entry = LibraryEntry(
                        sample_id=sample_id,
                        spectrum=data[sample_id],
                        wavelengths=wavelengths,
                        source_file=entry_meta.get('source_file', ''),
                        date_added=entry_meta.get('date_added', ''),
                        category=entry_meta.get('category', ''),
                        metadata=entry_meta.get('metadata', {}),
                        fingerprint=entry_meta.get('fingerprint', ''),
                    )
                    self._entries[sample_id] = entry
                    self._fingerprints[entry.fingerprint] = sample_id

            logger.info(f"Loaded library '{self.name}' with {self.size} entries")

        except Exception as e:
            logger.error(f"Failed to load library: {e}")

    def save(self) -> None:
        """Manually save the library."""
        self._save()


# Global library instance for the application
_global_library: Optional[SpectralLibrary] = None


def get_library(name: str = "local", **kwargs) -> SpectralLibrary:
    """
    Get or create the global spectral library.

    Parameters
    ----------
    name : str
        Library name
    **kwargs
        Additional arguments passed to SpectralLibrary constructor

    Returns
    -------
    SpectralLibrary
        The global library instance
    """
    global _global_library

    if _global_library is None or _global_library.name != name:
        _global_library = SpectralLibrary(name=name, **kwargs)

    return _global_library


def add_to_library(
    df: pd.DataFrame,
    source_file: str = "",
    category: str = "",
    library_name: str = "local",
) -> Tuple[int, int]:
    """
    Convenience function to add spectra from a DataFrame to the global library.

    Parameters
    ----------
    df : pd.DataFrame
        Spectral data (samples as rows, wavelengths as columns)
    source_file : str
        Source file path
    category : str
        Category label
    library_name : str
        Library name

    Returns
    -------
    Tuple[int, int]
        (added_count, skipped_count)
    """
    library = get_library(library_name)
    added, skipped, _ = library.add_spectra_batch(df, source_file, category)
    return added, skipped


def search_library(
    query: np.ndarray,
    wavelengths: np.ndarray,
    metric: str = 'hqi',
    top_k: int = 10,
    library_name: str = "local",
) -> pd.DataFrame:
    """
    Convenience function to search the global library.

    Parameters
    ----------
    query : np.ndarray
        Query spectrum
    wavelengths : np.ndarray
        Query wavelengths
    metric : str
        Similarity metric
    top_k : int
        Number of results
    library_name : str
        Library name

    Returns
    -------
    pd.DataFrame
        Search results
    """
    library = get_library(library_name)
    return library.search(query, wavelengths, metric, top_k)
