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

    # Initialize tier (required by app)
    if hasattr(app, '_on_tier_changed'):
        app._on_tier_changed()

    yield app, root

    # Cleanup at end of session
    try:
        root.quit()
        root.destroy()
    except tk.TclError:
        pass


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
def gui_app(session_app):
    """
    Provide the session app with reset state for each test.

    Resets data between tests but reuses the same app instance.
    """
    app, root = session_app

    # Reset app state before test
    app.X = None
    app.X_original = None
    app.y = None
    app.results_df = None

    yield app

    # State will be reset for next test


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
