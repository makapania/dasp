"""Thread-safe controller for search pause/resume/end functionality."""

import threading


class SearchController:
    """Thread-safe controller for search pause/resume/end.

    Uses threading.Event objects for safe cross-thread signaling.
    The search loop calls check_and_wait() at natural checkpoints.

    Usage:
        controller = SearchController()

        # In GUI thread:
        controller.pause()   # Pause search
        controller.resume()  # Resume search
        controller.end()     # End search immediately

        # In search thread:
        if not controller.check_and_wait():
            break  # End requested
    """

    def __init__(self):
        self._pause_event = threading.Event()
        self._end_event = threading.Event()
        self._pause_event.set()  # Start in running state (not paused)

    def pause(self):
        """Pause the search. Search will block at next checkpoint."""
        self._pause_event.clear()

    def resume(self):
        """Resume the search from paused state."""
        self._pause_event.set()

    def end(self):
        """End the search immediately. Also unblocks if paused."""
        self._end_event.set()
        self._pause_event.set()  # Unblock if waiting on pause

    def stop(self):
        """Alias for end() - stops the search immediately."""
        self.end()

    def reset(self):
        """Reset controller to initial state for a new search."""
        self._pause_event.set()
        self._end_event.clear()

    def check_and_wait(self) -> bool:
        """Check for end signal and wait if paused.

        Call this at natural checkpoints in the search loop.

        Returns:
            True to continue searching, False to end immediately.
        """
        if self._end_event.is_set():
            return False
        self._pause_event.wait()  # Blocks if paused
        return not self._end_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return not self._pause_event.is_set()

    @property
    def is_ended(self) -> bool:
        """Check if end was requested."""
        return self._end_event.is_set()
