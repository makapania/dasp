"""
Selection manager for bidirectional sync between plots and grid.

Allows selecting samples in PCA plot, spectra plot, or data grid
and synchronizing the selection across all views.
"""

from typing import Set, List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class SelectionState:
    """Current selection state."""
    selected_indices: Set[int] = field(default_factory=set)
    selection_source: str = "none"  # "grid", "spectra_plot", "pca_plot"


class SelectionManager:
    """
    Manages selection sync between plots and grid.

    Example
    -------
    >>> manager = SelectionManager()
    >>> manager.on_selection_change(lambda indices, source: print(f"Selected: {indices}"))
    >>> manager.select([0, 1, 2], source="pca_plot")
    Selected: {0, 1, 2}
    """

    def __init__(self):
        self._state = SelectionState()
        self._listeners: List[Callable[[Set[int], str], None]] = []

    @property
    def selected_indices(self) -> Set[int]:
        """Get currently selected indices."""
        return self._state.selected_indices.copy()

    @property
    def selection_source(self) -> str:
        """Get source of current selection."""
        return self._state.selection_source

    @property
    def selection_count(self) -> int:
        """Get number of selected items."""
        return len(self._state.selected_indices)

    def on_selection_change(self, callback: Callable[[Set[int], str], None]):
        """
        Register callback for selection changes.

        Parameters
        ----------
        callback : callable
            Called with (selected_indices, source) when selection changes
        """
        self._listeners.append(callback)

    def select(self, indices: List[int], source: str = "code", add: bool = False):
        """
        Select samples by index.

        Parameters
        ----------
        indices : list of int
            Sample indices to select
        source : str
            Source of selection ("grid", "spectra_plot", "pca_plot", "code")
        add : bool
            If True, add to existing selection. If False, replace.
        """
        if add:
            new_selection = self._state.selected_indices | set(indices)
        else:
            new_selection = set(indices)

        # Only notify if selection changed
        if new_selection != self._state.selected_indices or source != self._state.selection_source:
            self._state.selected_indices = new_selection
            self._state.selection_source = source
            self._notify()

    def toggle(self, index: int, source: str = "code"):
        """
        Toggle selection of a single sample.

        Parameters
        ----------
        index : int
            Sample index to toggle
        source : str
            Source of selection
        """
        new_selection = self._state.selected_indices.copy()
        if index in new_selection:
            new_selection.discard(index)
        else:
            new_selection.add(index)

        self._state.selected_indices = new_selection
        self._state.selection_source = source
        self._notify()

    def clear(self, source: str = "code"):
        """Clear all selection."""
        if self._state.selected_indices:
            self._state.selected_indices = set()
            self._state.selection_source = source
            self._notify()

    def select_all(self, n_samples: int, source: str = "code"):
        """Select all samples."""
        new_selection = set(range(n_samples))
        if new_selection != self._state.selected_indices:
            self._state.selected_indices = new_selection
            self._state.selection_source = source
            self._notify()

    def is_selected(self, index: int) -> bool:
        """Check if a sample is selected."""
        return index in self._state.selected_indices

    def _notify(self):
        """Notify all listeners of selection change."""
        for callback in self._listeners:
            try:
                callback(self._state.selected_indices, self._state.selection_source)
            except Exception as e:
                print(f"Selection callback error: {e}")
