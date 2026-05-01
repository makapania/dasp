"""T-11 B regression tests for SearchController pause-acknowledgment semantics."""
from __future__ import annotations

import threading
import time

import pytest

from spectral_predict.search_controller import SearchController


def test_initial_state_running():
    c = SearchController()
    assert not c.is_paused
    assert not c.is_actually_paused
    assert not c.is_ended


def test_pause_requested_before_acknowledged():
    """T-11 B: pause() flips `is_paused` immediately but `is_actually_paused`
    only becomes true once a worker reaches check_and_wait()."""
    c = SearchController()
    c.pause()
    assert c.is_paused
    # No worker has reached check_and_wait() yet — UI must NOT report
    # actual pause acknowledgment.
    assert not c.is_actually_paused


def test_actually_paused_true_during_check_and_wait_block():
    c = SearchController()
    seen_actual_paused = threading.Event()
    worker_resumed = threading.Event()

    def worker():
        # First call should pass through (not paused yet).
        assert c.check_and_wait()
        c.pause()  # Self-pause to simulate the GUI thread asking for pause.
        # Spin off a second thread that observes the actual-paused state
        # while the worker blocks.
        def observer():
            # Give the worker a moment to enter the wait
            time.sleep(0.05)
            if c.is_actually_paused:
                seen_actual_paused.set()
            time.sleep(0.05)
            c.resume()
        threading.Thread(target=observer, daemon=True).start()
        # This call blocks until observer resumes.
        assert c.check_and_wait()
        worker_resumed.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert seen_actual_paused.is_set(), "Observer must see is_actually_paused=True while worker blocks"
    assert worker_resumed.is_set(), "Worker must resume cleanly"
    # After the worker exits the wait, actual-paused must clear.
    assert not c.is_actually_paused


def test_resume_clears_blocked_worker():
    c = SearchController()
    c.pause()

    blocked = threading.Event()
    unblocked = threading.Event()

    def worker():
        blocked.set()
        c.check_and_wait()
        unblocked.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    blocked.wait(1.0)
    # Wait briefly so the worker actually enters the wait.
    time.sleep(0.05)
    assert c.is_actually_paused
    c.resume()
    unblocked.wait(1.0)
    assert unblocked.is_set()
    assert not c.is_actually_paused


def test_end_unblocks_paused_worker():
    c = SearchController()
    c.pause()

    unblocked = threading.Event()
    rc_holder = {}

    def worker():
        rc_holder["rc"] = c.check_and_wait()
        unblocked.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05)
    c.end()
    unblocked.wait(1.0)
    assert unblocked.is_set()
    assert rc_holder["rc"] is False  # end signaled


def test_reset_clears_actually_paused():
    c = SearchController()
    c.pause()
    # Manually fake an actually-paused state then reset.
    c._actually_paused.set()
    c.reset()
    assert not c.is_paused
    assert not c.is_actually_paused
    assert not c.is_ended


def test_check_and_wait_when_not_paused_does_not_set_actually_paused():
    """A pass-through check_and_wait (no pause requested) must not set
    is_actually_paused — that flag is only for "worker is actually blocked"
    semantics."""
    c = SearchController()
    assert c.check_and_wait()
    assert not c.is_actually_paused
