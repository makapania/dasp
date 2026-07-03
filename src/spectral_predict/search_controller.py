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
        # T-11 B: separate event so the GUI can tell whether the worker has
        # *actually* reached check_and_wait() and is blocked. Set when the
        # worker is currently inside `_pause_event.wait()`; cleared otherwise.
        self._actually_paused = threading.Event()
        self._pause_event.set()  # Start in running state (not paused)
        # T-17: explicit stop-requested flag (set by stop()/end(), cleared by
        # reset()). Lets callers distinguish "a stop was requested" from the
        # end-event internals, and gives the multi-target cancel path a
        # plain attribute to assert against.
        self._stop_requested = False

    def pause(self):
        """Request a pause. The worker blocks at the next checkpoint.

        Pause is asynchronous — the GUI should poll `is_actually_paused` to
        know when the worker is genuinely blocked (per T-11 B).
        """
        self._pause_event.clear()

    def resume(self):
        """Resume the search from paused state."""
        self._pause_event.set()

    def end(self):
        """End the search immediately. Also unblocks if paused."""
        self._end_event.set()
        self._stop_requested = True
        self._pause_event.set()  # Unblock if waiting on pause

    def stop(self):
        """Alias for end() - stops the search immediately."""
        self.end()

    def reset(self):
        """Reset controller to initial state for a new search."""
        self._pause_event.set()
        self._end_event.clear()
        self._actually_paused.clear()
        self._stop_requested = False

    def check_and_wait(self) -> bool:
        """Check for end signal and wait if paused.

        Call this at natural checkpoints in the search loop.

        Sets `_actually_paused` while blocked so the GUI can transition
        from "Pausing — please wait" to "Paused" only when the worker has
        genuinely stopped (T-11 B).

        Returns:
            True to continue searching, False to end immediately.
        """
        if self._end_event.is_set():
            return False
        if not self._pause_event.is_set():
            # About to block — tell the GUI "we're really paused now."
            self._actually_paused.set()
            try:
                self._pause_event.wait()
            finally:
                self._actually_paused.clear()
        return not self._end_event.is_set()

    @property
    def is_paused(self) -> bool:
        """True if a pause has been requested (regardless of worker state)."""
        return not self._pause_event.is_set()

    @property
    def is_actually_paused(self) -> bool:
        """True only when the worker is currently blocked at a checkpoint.

        The GUI uses this to distinguish "pause requested but trial still
        running" from "worker has acknowledged the pause." T-11 B.
        """
        return self._actually_paused.is_set()

    def is_ended(self) -> bool:
        """Check if end was requested.

        Note: unlike ``is_paused`` / ``is_actually_paused`` (which are
        properties), this is a regular method so callers can distinguish
        "query the flag" (``ctrl.is_ended()``) from the boolean value at
        call sites — required by the T-17 multi-target cancel path which
        asserts ``ctrl.is_ended() or ctrl._stop_requested``.
        """
        return self._end_event.is_set()
