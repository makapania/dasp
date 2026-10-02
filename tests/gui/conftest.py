"""
GUI test fixtures and configuration for Spectral Predict V1.

Usage:
    pytest tests/gui/ -v                    # Run headless (default)
    pytest tests/gui/ -v --visible          # Run with visible window
    pytest tests/gui/ -v --data-path=path   # Use different test data
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no plot windows

import pytest
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent.parent


def pytest_addoption(parser):
    """Add custom command line options for GUI tests."""
    parser.addoption(
        "--visible",
        action="store_true",
        default=False,
        help="Show GUI window during tests (for debugging)"
    )
    parser.addoption(
        "--data-path",
        default=str(PROJECT_ROOT / "example"),
        help="Path to test data folder (default: example/)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "comprehensive: mark test as comprehensive comparison test")


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_path(request):
    """Return the test data directory path."""
    return Path(request.config.getoption("--data-path"))


@pytest.fixture(scope="session")
def gui_visible(request):
    """Return whether GUI should be visible during tests."""
    return request.config.getoption("--visible")


def _toplevels(widget):
    """Every Toplevel below ``widget`` (dialogs can be parented to any widget)."""
    for child in widget.winfo_children():
        if isinstance(child, tk.Toplevel):
            yield child
        yield from _toplevels(child)


def _withdraw_all(root) -> None:
    """Hide ``root`` and every Toplevel under it."""
    for window in (root, *_toplevels(root)):
        try:
            window.withdraw()
        except tk.TclError:
            pass


@pytest.fixture(scope="session", autouse=True)
def _keep_tk_windows_hidden(gui_visible):
    """Headless runs: no app window or dialog may ever appear on screen.

    Withdrawing the root before building the app is not enough: the app maximises
    itself (``root.state('zoomed')``), dialogs ``deiconify()``, and a new Toplevel is
    mapped as soon as it is created. A visible window that is busy in a test shows
    up on the user's desktop as "ASP ... (Not Responding)". Applies to every root
    built under tests/gui (session app, module-level apps). ``--visible`` disables it.
    """
    if gui_visible:
        yield
        return
    mp = pytest.MonkeyPatch()
    original_toplevel_init = tk.Toplevel.__init__
    original_state = tk.Wm.wm_state
    original_grab_set = tk.Misc.grab_set

    def hidden_toplevel_init(self, *args, **kwargs):
        original_toplevel_init(self, *args, **kwargs)
        try:
            self.withdraw()
        except tk.TclError:
            pass

    def state_without_mapping(self, newstate=None):
        if newstate in ("normal", "zoomed", "iconic"):
            return None  # these would map the window
        return original_state(self, newstate)

    def grab_set_unmapped(self):
        try:
            return original_grab_set(self)
        except tk.TclError:
            return None  # "grab failed: window not viewable" on a withdrawn dialog

    mp.setattr(tk.Toplevel, "__init__", hidden_toplevel_init)
    for name in ("deiconify", "wm_deiconify"):
        mp.setattr(tk.Wm, name, lambda self: None)
    for name in ("state", "wm_state"):
        mp.setattr(tk.Wm, name, state_without_mapping)
    mp.setattr(tk.Misc, "grab_set", grab_set_unmapped)
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def session_app(gui_visible):
    """
    Create a single SpectralPredictApp instance for the entire test session.

    Session-scoped to avoid Tkinter menu resource exhaustion.
    All messagebox dialogs are auto-dismissed to prevent blocking in headless mode.
    """
    from spectral_predict_gui_optimized import SpectralPredictApp

    root = tk.Tk()
    root.title("GUI Test Session")

    if not gui_visible:
        root.withdraw()

    app = SpectralPredictApp(root)
    if not gui_visible:
        _withdraw_all(root)  # belt and braces: startup re-shows the root (zoomed)

    # Initialize tier (required by app)
    if hasattr(app, '_on_tier_changed'):
        app._on_tier_changed()
    root.update_idletasks()

    # Fresh-launch state that gui_app restores before every test.
    app._test_baseline = _AppStateSnapshot(app, root)

    yield app, root

    # Cleanup at end of session: pending callbacks first, so none fires on a dead root.
    _cancel_pending_after(root)
    try:
        root.quit()
        root.destroy()
    except tk.TclError:
        pass
    import matplotlib.pyplot as plt

    plt.close('all')


# ---------------------------------------------------------------------------
# Shared-app state reset
# ---------------------------------------------------------------------------


def _cancel_pending_after(root) -> None:
    """Cancel every pending ``after``/``after_idle`` callback on ``root``'s interpreter.

    The freshly launched app has none, so anything pending belongs to an earlier test
    (debounced filters, progress polls) and would otherwise fire inside the next one.

    Cancels the raw Tcl timer only. ``root.after_cancel`` would also delete the
    callback's Tcl command, but that command belongs to the widget that scheduled it
    (e.g. the running-figure canvas uses ``canvas.after``); the owner later deletes
    it again on destroy and raises ``TclError: can't delete Tcl command``. Leaving the
    command registered costs nothing: its owner cleans it up when destroyed.
    """
    try:
        pending = root.tk.splitlist(root.tk.call('after', 'info'))
    except tk.TclError:
        return
    for after_id in pending:
        try:
            root.tk.call('after', 'cancel', after_id)
        except tk.TclError:
            pass


def _holds_gui_objects(value, _depth: int = 0) -> bool:
    """True if ``value`` is, or contains, a widget, Tk variable or figure.

    Such attributes are wiring (lazily built pages, canvases, widget registries), not
    data; resetting them would orphan live widgets, so the reset leaves them alone.
    """
    if isinstance(value, (tk.Misc, tk.Variable)):
        return True
    module = type(value).__module__ or ''
    if module.startswith(('matplotlib', 'tkinter', 'threading')):
        return True
    if _depth > 3:
        return False
    if isinstance(value, dict):
        return any(_holds_gui_objects(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_gui_objects(v, _depth + 1) for v in value)
    if not _is_plain_data(value) and hasattr(value, '__dict__'):
        # Helper objects (progress monitors, controllers) that keep widget references
        # directly. One level only: deeper chains reach the app itself.
        return any(
            isinstance(v, (tk.Misc, tk.Variable))
            or (type(v).__module__ or '').startswith(('matplotlib', 'tkinter'))
            for v in vars(value).values()
        )
    return False


_PLAIN_LEAVES = (type(None), bool, int, float, complex, str, bytes)


def _is_plain_data(value, _depth: int = 0) -> bool:
    """True for scalars, strings, numpy/pandas data and containers of them only."""
    import numpy as np
    import pandas as pd

    if isinstance(value, _PLAIN_LEAVES + (np.generic, np.ndarray, pd.DataFrame, pd.Series, pd.Index)):
        return True
    if _depth > 4:
        return False
    if isinstance(value, dict):
        return all(
            _is_plain_data(k, _depth + 1) and _is_plain_data(v, _depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_is_plain_data(v, _depth + 1) for v in value)
    return False


class _AppStateSnapshot:
    """The session app's state right after launch, and how to put it back.

    Covers (a) every Tk variable held by the app, directly or in a dict/list
    attribute (~820), by raw Tcl value; (b) every data attribute (data, validation
    arrays, exclusions, results, transfer/contamination state ...) whose launch value
    holds no GUI objects, and removal of data attributes created since launch; (c) the
    DataSourceManager's contents; (d) variable traces,
    Toplevel windows and pending ``after`` callbacks added since launch.
    """

    _MAX_VAR_PASSES = 5

    def __init__(self, app, root):
        import copy

        self.root = root
        self.var_values: dict[str, str] = {}
        self.var_traces: dict[str, set] = {}
        self.vars: dict[str, tk.Variable] = {}
        for value in vars(app).values():
            for var in self._tk_vars_in(value):
                name = str(var)
                self.vars[name] = var
                self.var_values[name] = str(root.getvar(name))
                self.var_traces[name] = set(var.trace_info())
        # Only attributes whose launch value is plain data. Helper objects built at
        # launch (fonts, styles, sidebar, tooltips) hold Tk handles and are wiring.
        self.data: dict[str, object] = {
            name: copy.deepcopy(value)
            for name, value in vars(app).items()
            if _is_plain_data(value)
        }
        dsm = getattr(app, 'data_source_manager', None)
        self.dsm_state = None
        if dsm is not None and _is_plain_data(list(vars(dsm).values())):
            self.dsm_state = copy.deepcopy(vars(dsm))
        self.toplevels = {str(w) for w in root.winfo_children() if isinstance(w, tk.Toplevel)}
        # Everything present at launch, plus the snapshot itself (set right after).
        self.launch_attrs = set(vars(app)) | {'_test_baseline'}

    @staticmethod
    def _tk_vars_in(value):
        if isinstance(value, tk.Variable):
            yield value
        elif isinstance(value, dict):
            yield from (v for v in value.values() if isinstance(v, tk.Variable))
        elif isinstance(value, (list, tuple)):
            yield from (v for v in value if isinstance(v, tk.Variable))

    def restore(self, app) -> list[str]:
        """Put the app back to launch state. Returns Tk variables that would not settle."""
        import copy

        root = self.root
        _cancel_pending_after(root)
        for widget in root.winfo_children():
            if isinstance(widget, tk.Toplevel) and str(widget) not in self.toplevels:
                try:
                    widget.destroy()
                except tk.TclError:
                    pass

        for name, value in self.data.items():
            current = getattr(app, name, None)
            if _holds_gui_objects(current):
                continue  # now wired to live widgets (e.g. a lazily built page)
            setattr(app, name, copy.deepcopy(value))
        # Attributes created since launch (backups such as X_before_contam_correction,
        # caches) did not exist in a fresh app; drop the data ones, keep widget wiring.
        for name in set(vars(app)) - self.launch_attrs:
            if not _holds_gui_objects(vars(app)[name]):
                delattr(app, name)
        dsm = getattr(app, 'data_source_manager', None)
        if dsm is not None and self.dsm_state is not None:
            vars(dsm).clear()
            vars(dsm).update(copy.deepcopy(self.dsm_state))

        # Remove traces tests added, before restoring values, so they do not fire.
        for name, var in self.vars.items():
            try:
                for mode, callback in set(var.trace_info()) - self.var_traces[name]:
                    var.trace_remove(mode, callback)
            except tk.TclError:
                pass

        # Writes fire the app's own traces, which rewrite other variables (task_type
        # rewrites imbalance_method; model boxes rewrite model_tier). Re-apply until
        # nothing differs from launch, so restore order does not matter.
        unsettled: list[str] = []
        for _ in range(self._MAX_VAR_PASSES):
            unsettled = []
            for name, value in self.var_values.items():
                try:
                    if str(root.getvar(name)) != value:
                        unsettled.append(name)
                        root.setvar(name, value)
                except tk.TclError:
                    pass
            if not unsettled:
                break
        unsettled = [
            name for name, value in self.var_values.items() if str(root.getvar(name)) != value
        ]
        # Traces fired above may have queued debounced work against the reset state.
        _cancel_pending_after(root)
        import matplotlib.pyplot as plt

        plt.close('all')
        return unsettled


@pytest.fixture(autouse=True)
def _suppress_dialogs():
    """Auto-dismiss all tkinter dialogs to prevent blocking in headless tests."""
    with patch('tkinter.messagebox.showinfo', return_value=None), \
         patch('tkinter.messagebox.showwarning', return_value=None), \
         patch('tkinter.messagebox.showerror', return_value=None), \
         patch('tkinter.messagebox.askyesno', return_value=True), \
         patch('tkinter.messagebox.askokcancel', return_value=True), \
         patch('tkinter.messagebox.askquestion', return_value='yes'), \
         patch('tkinter.messagebox.askretrycancel', return_value=True), \
         patch('tkinter.messagebox.askyesnocancel', return_value=True), \
         patch('tkinter.filedialog.askopenfilename', return_value=''), \
         patch('tkinter.filedialog.asksaveasfilename', return_value=''), \
         patch('tkinter.filedialog.askdirectory', return_value=''):
        yield


@pytest.fixture
def gui_app(session_app, request):
    """
    Provide the session app with reset state for each test.

    Reuses the same app instance but restores its launch state first: every Tk
    variable, every data attribute (data, validation arrays, exclusions, results,
    data sources ...), and no leftover traces, dialogs or pending callbacks. See
    ``_AppStateSnapshot``.
    """
    app, root = session_app

    unsettled = app._test_baseline.restore(app)
    if not request.config.getoption("--visible"):
        _withdraw_all(root)
    if unsettled:
        # Fail here, at the reset, rather than letting a later assertion trip over
        # state a previous test left behind.
        pytest.fail(
            f"GUI reset: Tk variables did not settle to launch values: {unsettled[:10]}",
            pytrace=False,
        )

    yield app

    # T-51 PR D: the session app is shared, so a ticked bundle or a bad startup
    # value left by one test must not leak into the next launch test.
    from spectral_predict.run_gui_settings import LEGACY_DEFAULTS

    for name, value in LEGACY_DEFAULTS.items():
        var = getattr(app, name, None)
        if var is not None:
            var.set(value)


@pytest.fixture
def gui_harness(gui_app, gui_visible):
    """
    Create a GUITestHarness wrapping the app.

    This is the main fixture for most GUI tests.
    """
    from tests.gui.harness import GUITestHarness
    return GUITestHarness(gui_app, visible=gui_visible)


@pytest.fixture
def example_csv_path(data_path):
    """Path to the BoneCollagen.csv example file."""
    csv_path = data_path / "BoneCollagen.csv"
    if not csv_path.exists():
        pytest.skip(f"Example data not found: {csv_path}")
    return csv_path


@pytest.fixture
def example_asd_dir(data_path):
    """Path to the directory containing ASD spectral files."""
    asd_files = list(data_path.glob("*.asd"))
    if not asd_files:
        pytest.skip(f"No ASD files found in: {data_path}")
    return data_path


# Cache loaded data to avoid reloading for each test
_cached_spectral_data = None
_cached_reference_data = None


def _load_example_data(data_path):
    """Load and cache example data."""
    global _cached_spectral_data, _cached_reference_data

    if _cached_spectral_data is not None:
        return _cached_spectral_data.copy(), _cached_reference_data.copy()

    import pandas as pd
    from spectral_predict.io import read_asd_dir

    # Load reference data
    csv_path = data_path / "BoneCollagen.csv"
    ref_df = pd.read_csv(csv_path)

    # Read spectral data
    result = read_asd_dir(str(data_path))
    X = result[0] if isinstance(result, tuple) else result

    # Adjust index to match reference format
    new_index = [idx.replace("Spectrum", "Spectrum ") if idx.startswith("Spectrum") else idx
                 for idx in X.index]
    X.index = new_index

    # Match with reference data
    ref_df['File Number'] = ref_df['File Number'].str.strip()
    X.index = X.index.str.strip()

    common_ids = X.index.intersection(ref_df.set_index('File Number').index)
    X = X.loc[common_ids]
    ref_subset = ref_df.set_index('File Number').loc[common_ids]

    # Cache the data
    _cached_spectral_data = X
    _cached_reference_data = ref_subset

    return X.copy(), ref_subset.copy()


@pytest.fixture
def loaded_regression_data(gui_harness, data_path):
    """
    GUI harness with BoneCollagen data loaded for regression.

    Uses %Collagen as target (continuous variable).
    """
    try:
        X, ref_subset = _load_example_data(data_path)
    except Exception as e:
        pytest.skip(f"Could not load example data: {e}")

    y = ref_subset['%Collagen']

    # Set data in app
    gui_harness.app.X = X
    gui_harness.app.X_original = X.copy()
    gui_harness.app.y = y
    gui_harness.app.task_type.set("regression")

    gui_harness.wait_for_idle()

    return gui_harness


@pytest.fixture
def loaded_classification_data(gui_harness, data_path):
    """
    GUI harness with BoneCollagen data loaded for classification.

    Uses CollagenCat as target (Low/Medium/High categories).
    """
    try:
        X, ref_subset = _load_example_data(data_path)
    except Exception as e:
        pytest.skip(f"Could not load example data: {e}")

    y = ref_subset['CollagenCat']

    # Set data in app
    gui_harness.app.X = X
    gui_harness.app.X_original = X.copy()
    gui_harness.app.y = y
    gui_harness.app.task_type.set("classification")

    gui_harness.wait_for_idle()

    return gui_harness
