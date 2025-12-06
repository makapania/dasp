"""
Progress bars and cancellation support for Spectral Predict v3.

This module provides thread-safe progress tracking with:
- Progress bars with percentage display
- Cancel buttons for long operations
- Estimated time remaining
- Thread-safe state management
"""

import dearpygui.dearpygui as dpg
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ProgressState:
    """Thread-safe progress state."""
    is_running: bool = False
    is_cancelled: bool = False
    current: int = 0
    total: int = 100
    message: str = ""
    start_time: float = 0.0
    lock: threading.Lock = threading.Lock()


class ProgressTracker:
    """
    Thread-safe progress tracker with DPG UI integration.

    Allows background threads to update progress while the UI remains responsive.
    """

    def __init__(self, tag: str, parent: int, width: int = 600):
        """
        Initialize progress tracker.

        Parameters
        ----------
        tag : str
            Unique tag for the progress UI
        parent : int
            Parent DPG item ID
        width : int
            Width of the progress bar in pixels
        """
        self.tag = tag
        self.parent = parent
        self.width = width
        self.state = ProgressState()
        self.state.lock = threading.Lock()

        # Create UI elements
        self._create_ui()

    def _create_ui(self):
        """Create the progress bar UI elements."""
        with dpg.group(tag=self.tag, parent=self.parent, horizontal=False, show=False):
            # Status message
            dpg.add_text(
                "Starting...",
                tag=f"{self.tag}_message",
                color=(255, 255, 255)
            )

            dpg.add_spacer(height=5)

            # Progress bar
            dpg.add_progress_bar(
                tag=f"{self.tag}_bar",
                default_value=0.0,
                width=self.width,
                overlay="0%"
            )

            dpg.add_spacer(height=5)

            # Info line (ETA, items processed)
            dpg.add_text(
                "",
                tag=f"{self.tag}_info",
                color=(150, 150, 150)
            )

            dpg.add_spacer(height=10)

            # Cancel button
            dpg.add_button(
                label="Cancel",
                tag=f"{self.tag}_cancel",
                callback=self._on_cancel,
                width=100
            )

    def _on_cancel(self):
        """Handle cancel button click."""
        with self.state.lock:
            self.state.is_cancelled = True
            dpg.set_value(f"{self.tag}_message", "Cancelling...")
            dpg.configure_item(f"{self.tag}_cancel", enabled=False)

    def start(self, total: int, message: str = "Processing..."):
        """
        Start tracking progress.

        Parameters
        ----------
        total : int
            Total number of items to process
        message : str
            Initial status message
        """
        with self.state.lock:
            self.state.is_running = True
            self.state.is_cancelled = False
            self.state.current = 0
            self.state.total = max(1, total)  # Avoid division by zero
            self.state.message = message
            self.state.start_time = time.time()

        # Update UI
        dpg.configure_item(self.tag, show=True)
        dpg.set_value(f"{self.tag}_message", message)
        dpg.set_value(f"{self.tag}_bar", 0.0)
        dpg.configure_item(f"{self.tag}_bar", overlay="0%")
        dpg.set_value(f"{self.tag}_info", "Starting...")
        dpg.configure_item(f"{self.tag}_cancel", enabled=True)

    def update(self, current: int, message: Optional[str] = None):
        """
        Update progress.

        Parameters
        ----------
        current : int
            Current progress (number of items processed)
        message : str, optional
            Updated status message
        """
        with self.state.lock:
            self.state.current = current
            if message is not None:
                self.state.message = message

            # Calculate progress percentage
            progress = min(1.0, current / self.state.total)
            percentage = int(progress * 100)

            # Calculate ETA
            elapsed = time.time() - self.state.start_time
            if current > 0:
                items_per_sec = current / elapsed
                remaining_items = self.state.total - current
                eta_seconds = remaining_items / items_per_sec if items_per_sec > 0 else 0
            else:
                eta_seconds = 0

        # Update UI (outside lock to avoid deadlock)
        dpg.set_value(f"{self.tag}_bar", progress)
        dpg.configure_item(f"{self.tag}_bar", overlay=f"{percentage}%")

        if message is not None:
            dpg.set_value(f"{self.tag}_message", message)

        # Format info string
        if eta_seconds > 0:
            eta_str = self._format_time(eta_seconds)
            info = f"Processed {current}/{self.state.total} items | ETA: {eta_str}"
        else:
            info = f"Processed {current}/{self.state.total} items"

        dpg.set_value(f"{self.tag}_info", info)

    def finish(self, message: str = "Complete!"):
        """
        Mark progress as complete.

        Parameters
        ----------
        message : str
            Completion message
        """
        with self.state.lock:
            self.state.is_running = False
            self.state.current = self.state.total

        # Update UI
        dpg.set_value(f"{self.tag}_bar", 1.0)
        dpg.configure_item(f"{self.tag}_bar", overlay="100%")
        dpg.set_value(f"{self.tag}_message", message)

        elapsed = time.time() - self.state.start_time
        dpg.set_value(f"{self.tag}_info", f"Completed in {self._format_time(elapsed)}")

        dpg.configure_item(f"{self.tag}_cancel", enabled=False)

        # Hide after a delay
        threading.Timer(2.0, lambda: dpg.configure_item(self.tag, show=False)).start()

    def hide(self):
        """Hide the progress bar immediately."""
        dpg.configure_item(self.tag, show=False)

    def is_cancelled(self) -> bool:
        """
        Check if operation was cancelled.

        Returns
        -------
        bool
            True if user clicked cancel
        """
        with self.state.lock:
            return self.state.is_cancelled

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds into human-readable time.

        Parameters
        ----------
        seconds : float
            Time in seconds

        Returns
        -------
        str
            Formatted time string (e.g., "2m 30s", "45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


def create_progress_tracker(tag: str, parent: int, width: int = 600) -> ProgressTracker:
    """
    Factory function to create a progress tracker.

    Parameters
    ----------
    tag : str
        Unique tag for the progress UI
    parent : int
        Parent DPG item ID
    width : int
        Width of the progress bar in pixels

    Returns
    -------
    ProgressTracker
        Configured progress tracker instance
    """
    return ProgressTracker(tag=tag, parent=parent, width=width)


class SimpleProgressBar:
    """
    Simpler progress bar without ETA calculation (for lighter use cases).
    """

    def __init__(self, tag: str, parent: int, width: int = 400):
        """
        Initialize simple progress bar.

        Parameters
        ----------
        tag : str
            Unique tag
        parent : int
            Parent DPG item ID
        width : int
            Width in pixels
        """
        self.tag = tag
        self.parent = parent
        self.width = width
        self._create_ui()

    def _create_ui(self):
        """Create simple progress bar UI."""
        with dpg.group(tag=self.tag, parent=self.parent, horizontal=False, show=False):
            dpg.add_progress_bar(
                tag=f"{self.tag}_bar",
                default_value=0.0,
                width=self.width,
                overlay="0%"
            )

    def show(self):
        """Show the progress bar."""
        dpg.configure_item(self.tag, show=True)

    def hide(self):
        """Hide the progress bar."""
        dpg.configure_item(self.tag, show=False)

    def set_progress(self, value: float):
        """
        Set progress value.

        Parameters
        ----------
        value : float
            Progress from 0.0 to 1.0
        """
        value = max(0.0, min(1.0, value))
        percentage = int(value * 100)

        dpg.set_value(f"{self.tag}_bar", value)
        dpg.configure_item(f"{self.tag}_bar", overlay=f"{percentage}%")


# Example usage in a background thread
def example_long_operation(progress_tracker: ProgressTracker):
    """
    Example of using ProgressTracker in a background operation.

    Parameters
    ----------
    progress_tracker : ProgressTracker
        Progress tracker instance
    """
    n_items = 100

    progress_tracker.start(total=n_items, message="Processing items...")

    for i in range(n_items):
        # Check for cancellation
        if progress_tracker.is_cancelled():
            progress_tracker.finish(message="Cancelled by user")
            return False

        # Simulate work
        time.sleep(0.05)

        # Update progress
        progress_tracker.update(
            current=i + 1,
            message=f"Processing item {i + 1}/{n_items}..."
        )

    progress_tracker.finish(message="All items processed successfully!")
    return True
