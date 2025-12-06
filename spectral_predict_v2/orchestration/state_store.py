"""
State Store - Single source of truth for application state.

All UI components read from and write to this store, ensuring consistency
across modes (Explore, Build, Predict).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum, auto
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal


class AppMode(Enum):
    """Application modes."""
    EXPLORE = auto()
    BUILD = auto()
    PREDICT = auto()


class TaskType(Enum):
    """Analysis task types."""
    REGRESSION = auto()
    CLASSIFICATION = auto()


@dataclass
class DataState:
    """Current dataset state."""
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    X: Optional[np.ndarray] = None  # Spectral data (samples x wavelengths)
    y: Optional[np.ndarray] = None  # Target values
    wavelengths: Optional[np.ndarray] = None  # Wavelength axis
    sample_ids: Optional[list] = None  # Sample identifiers
    target_column: Optional[str] = None  # Name of target column
    n_samples: int = 0
    n_wavelengths: int = 0
    wavelength_min: float = 0.0
    wavelength_max: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0

    # Quality flags
    outlier_indices: list = field(default_factory=list)
    missing_count: int = 0

    # Preprocessing applied
    preprocessing_applied: Optional[str] = None  # e.g., "SNV", "SG1"
    X_preprocessed: Optional[np.ndarray] = None  # Preprocessed spectra


@dataclass
class AnalysisState:
    """Current analysis state."""
    is_running: bool = False
    progress: float = 0.0  # 0.0 to 1.0
    current_stage: str = ""
    estimated_time_remaining: Optional[float] = None  # seconds

    # Results
    results_df: Optional[pd.DataFrame] = None
    n_models_evaluated: int = 0
    best_model_name: Optional[str] = None
    best_score: Optional[float] = None


@dataclass
class ModelState:
    """Current model state (for Build/Predict modes)."""
    selected_model: Optional[Any] = None  # Trained model object
    model_name: Optional[str] = None
    model_config: dict = field(default_factory=dict)
    preprocessing: Optional[str] = None
    variable_mask: Optional[np.ndarray] = None  # Boolean mask for selected variables
    metrics: dict = field(default_factory=dict)  # RMSECV, R2, etc.
    is_saved: bool = False
    save_path: Optional[str] = None


class StateStore(QObject):
    """
    Central state management for the application.

    Emits signals when state changes so UI components can update.
    """

    # Signals for state changes
    mode_changed = Signal(AppMode)
    data_changed = Signal()
    analysis_started = Signal()
    analysis_progress = Signal(float, str)  # progress, stage
    analysis_completed = Signal()
    model_changed = Signal()
    preset_changed = Signal(str)

    def __init__(self):
        super().__init__()

        # Core state
        self._mode = AppMode.EXPLORE
        self._task_type = TaskType.REGRESSION
        self._data = DataState()
        self._analysis = AnalysisState()
        self._model = ModelState()
        self._current_preset: Optional[str] = None

        # Pinned models for comparison (up to 4)
        self._pinned_model_indices: list[int] = []

    # --- Mode ---

    @property
    def mode(self) -> AppMode:
        return self._mode

    def set_mode(self, mode: AppMode):
        if self._mode != mode:
            self._mode = mode
            self.mode_changed.emit(mode)

    # --- Task Type ---

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def set_task_type(self, task_type: TaskType):
        self._task_type = task_type

    # --- Data ---

    @property
    def data(self) -> DataState:
        return self._data

    @property
    def has_data(self) -> bool:
        return self._data.X is not None

    def load_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        file_path: str,
        target_column: str,
        sample_ids: Optional[list] = None
    ):
        """Load a new dataset into the store."""
        import os

        # Handle case when y is None (spectral files without targets)
        if y is not None:
            target_min = float(np.nanmin(y))
            target_max = float(np.nanmax(y))
        else:
            target_min = 0.0
            target_max = 0.0

        # Generate sample_ids if not provided
        if sample_ids is None:
            sample_ids = [f"Sample_{i+1}" for i in range(X.shape[0])]

        self._data = DataState(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            X=X,
            y=y,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            target_column=target_column,
            n_samples=X.shape[0],
            n_wavelengths=X.shape[1],
            wavelength_min=float(wavelengths.min()),
            wavelength_max=float(wavelengths.max()),
            target_min=target_min,
            target_max=target_max,
        )

        # Clear previous analysis results
        self._analysis = AnalysisState()
        self._model = ModelState()
        self._pinned_model_indices = []

        self.data_changed.emit()

    def set_outliers(self, indices: list[int]):
        """Mark samples as outliers."""
        self._data.outlier_indices = indices
        self.data_changed.emit()

    def apply_preprocessing(self, name: str, X_preprocessed: np.ndarray):
        """Store preprocessed data."""
        self._data.preprocessing_applied = name
        self._data.X_preprocessed = X_preprocessed
        self.data_changed.emit()

    def get_active_X(self) -> Optional[np.ndarray]:
        """Get the active spectral data (preprocessed if available)."""
        if self._data.X_preprocessed is not None:
            return self._data.X_preprocessed
        return self._data.X

    # --- Analysis ---

    @property
    def analysis(self) -> AnalysisState:
        return self._analysis

    def start_analysis(self):
        """Mark analysis as started."""
        self._analysis.is_running = True
        self._analysis.progress = 0.0
        self._analysis.current_stage = "Initializing..."
        self.analysis_started.emit()

    def update_progress(self, progress: float, stage: str, time_remaining: Optional[float] = None):
        """Update analysis progress."""
        self._analysis.progress = progress
        self._analysis.current_stage = stage
        self._analysis.estimated_time_remaining = time_remaining
        self.analysis_progress.emit(progress, stage)

    def complete_analysis(self, results_df: pd.DataFrame):
        """Mark analysis as complete with results."""
        self._analysis.is_running = False
        self._analysis.progress = 1.0
        self._analysis.results_df = results_df
        self._analysis.n_models_evaluated = len(results_df)

        if len(results_df) > 0:
            # Assuming results are sorted by score descending
            best = results_df.iloc[0]
            self._analysis.best_model_name = str(best.get("model", "Unknown"))
            self._analysis.best_score = float(best.get("composite_score", 0))

        self.analysis_completed.emit()

    # --- Model ---

    @property
    def model(self) -> ModelState:
        return self._model

    def set_model(
        self,
        model: Any,
        name: str,
        config: dict,
        preprocessing: Optional[str] = None,
        variable_mask: Optional[np.ndarray] = None,
        metrics: Optional[dict] = None
    ):
        """Set the current working model."""
        self._model = ModelState(
            selected_model=model,
            model_name=name,
            model_config=config,
            preprocessing=preprocessing,
            variable_mask=variable_mask,
            metrics=metrics or {},
        )
        self.model_changed.emit()

    def mark_model_saved(self, path: str):
        """Mark the current model as saved."""
        self._model.is_saved = True
        self._model.save_path = path
        self.model_changed.emit()

    # --- Pinned Models for Comparison ---

    @property
    def pinned_indices(self) -> list[int]:
        return self._pinned_model_indices

    def toggle_pinned(self, index: int) -> bool:
        """Toggle a model's pinned state. Returns True if now pinned."""
        if index in self._pinned_model_indices:
            self._pinned_model_indices.remove(index)
            return False
        elif len(self._pinned_model_indices) < 4:
            self._pinned_model_indices.append(index)
            return True
        return False  # Already at max

    def clear_pinned(self):
        """Clear all pinned models."""
        self._pinned_model_indices = []

    # --- Presets ---

    @property
    def current_preset(self) -> Optional[str]:
        return self._current_preset

    def set_preset(self, preset_name: str):
        """Set the current preset."""
        self._current_preset = preset_name
        self.preset_changed.emit(preset_name)

    # --- Utility ---

    def get_data_summary(self) -> dict:
        """Get a summary of current data for display."""
        if not self.has_data:
            return {}

        return {
            "file_name": self._data.file_name,
            "n_samples": self._data.n_samples,
            "n_wavelengths": self._data.n_wavelengths,
            "wavelength_range": f"{self._data.wavelength_min:.0f}-{self._data.wavelength_max:.0f}",
            "target": self._data.target_column,
            "target_range": f"{self._data.target_min:.2f}-{self._data.target_max:.2f}",
            "n_outliers": len(self._data.outlier_indices),
            "preprocessing": self._data.preprocessing_applied,
        }

    def reset(self):
        """Reset all state."""
        self._mode = AppMode.EXPLORE
        self._task_type = TaskType.REGRESSION
        self._data = DataState()
        self._analysis = AnalysisState()
        self._model = ModelState()
        self._current_preset = None
        self._pinned_model_indices = []
        self.data_changed.emit()
