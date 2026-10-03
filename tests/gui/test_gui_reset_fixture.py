"""The shared-app reset in tests/gui/conftest.py: callbacks, windows, data."""

from __future__ import annotations

import tkinter as tk

from tests.gui.conftest import _cancel_pending_after


def test_cancelling_widget_owned_callbacks_keeps_owner_teardown_working(session_app):
    """A canvas-scheduled callback (like the running-figure animation) is cancelled
    without deleting the canvas's command, so destroying the canvas later is clean."""
    _app, root = session_app
    canvas = tk.Canvas(root)
    fired = []
    canvas.after(60_000, lambda: fired.append(1))
    canvas.after_idle(lambda: fired.append(2))

    _cancel_pending_after(root)

    assert root.tk.splitlist(root.tk.call("after", "info")) == ()
    canvas.destroy()  # raised "can't delete Tcl command" with root.after_cancel
    root.update()
    assert fired == []


def test_reset_removes_data_and_windows_a_test_added(gui_app, request):
    root = gui_app.root
    gui_app.X = object()
    gui_app._scratch_from_a_test = [1, 2, 3]
    dialog = tk.Toplevel(root)

    unsettled = gui_app._test_baseline.restore(gui_app)

    assert unsettled == []
    assert gui_app.X is None
    assert not hasattr(gui_app, "_scratch_from_a_test")
    assert not dialog.winfo_exists()
