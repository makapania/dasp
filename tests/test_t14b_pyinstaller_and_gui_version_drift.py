"""T-14b: pin every PyInstaller-chain and GUI version-display surface to the
canonical ``spectral_predict.__version__`` source, so a future bump cannot
silently leave the bundled .exe metadata, the Inno Setup output filename, or
the GUI title bar pointing at a stale version string.

T-14 closed five drift sites in src/. T-14b extends the same pattern to the
build chain (whose drift the user sees in the .exe filename and Windows file
properties) and the GUI (whose drift the user sees in the OS title bar).
"""

from __future__ import annotations

import re
from pathlib import Path

from spectral_predict import __version__

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    return (PROJECT_ROOT / relpath).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GUI title bar + in-canvas version label
# ---------------------------------------------------------------------------


def test_gui_has_no_hardcoded_b1_literal():
    """The GUI must not contain a hardcoded ``0.5.0b1`` (or any prior beta)
    literal anywhere — both display sites should drive from ``__version__``."""
    src = _read("spectral_predict_gui_optimized.py")
    for stale in ("0.5.0b1", "0.4.0", "0.3.0", "0.2.0"):
        assert stale not in src, (
            f"GUI source contains stale version literal {stale!r}; "
            "version drift has regressed (T-14b)"
        )


def test_gui_imports_dasp_version():
    """GUI must import ``__version__`` from the package so the title bar and
    version label render the live value."""
    src = _read("spectral_predict_gui_optimized.py")
    assert "from spectral_predict import __version__" in src, (
        "GUI must import __version__ from spectral_predict to avoid drift"
    )
    # Both display sites must use the imported value.
    assert "self.root.title(f\"ASP - Advanced Spectral Prediction" in src
    assert "_DASP_VERSION" in src


# ---------------------------------------------------------------------------
# version_info.txt — Windows .exe file properties (right-click → Properties)
# ---------------------------------------------------------------------------


def test_version_info_txt_strings_match_dasp_version():
    """``FileVersion`` and ``ProductVersion`` strings in version_info.txt must
    equal ``spectral_predict.__version__``. Drift here means the bundled .exe
    shows a stale version under Windows file properties even when the running
    app reports the new one."""
    text = _read("version_info.txt")
    file_version = re.search(r"FileVersion'?,\s*u?'([^']+)'", text)
    product_version = re.search(r"ProductVersion'?,\s*u?'([^']+)'", text)
    assert file_version, "FileVersion not found in version_info.txt"
    assert product_version, "ProductVersion not found in version_info.txt"
    assert file_version.group(1) == __version__, (
        f"FileVersion in version_info.txt is {file_version.group(1)!r}; "
        f"must match __version__ ({__version__!r})"
    )
    assert product_version.group(1) == __version__, (
        f"ProductVersion in version_info.txt is {product_version.group(1)!r}; "
        f"must match __version__ ({__version__!r})"
    )


def test_version_info_txt_tuple_trailing_element_tracks_beta_number():
    """Windows file-version tuples must be 4 ints. The codebase convention
    encodes the beta number in the trailing tuple element (so 0.5.0b2 →
    (0, 5, 0, 2)). Pin that mapping so a string bump that forgets the tuple
    fails the test."""
    text = _read("version_info.txt")

    # Extract beta number from __version__ when present (e.g. "0.5.0b2" → 2).
    m = re.match(r"(\d+)\.(\d+)\.(\d+)b(\d+)$", __version__)
    if not m:
        # Non-beta version — no tuple/string drift contract to enforce here.
        return
    major, minor, patch, beta = (int(g) for g in m.groups())
    expected_tuple = f"({major}, {minor}, {patch}, {beta})"

    filevers = re.search(r"filevers=\(([^)]+)\)", text)
    prodvers = re.search(r"prodvers=\(([^)]+)\)", text)
    assert filevers, "filevers tuple not found in version_info.txt"
    assert prodvers, "prodvers tuple not found in version_info.txt"
    assert f"({filevers.group(1)})" == expected_tuple, (
        f"filevers tuple is ({filevers.group(1)}); expected {expected_tuple} "
        f"to match __version__ {__version__!r}"
    )
    assert f"({prodvers.group(1)})" == expected_tuple, (
        f"prodvers tuple is ({prodvers.group(1)}); expected {expected_tuple} "
        f"to match __version__ {__version__!r}"
    )


# ---------------------------------------------------------------------------
# Inno Setup script — output filename + AppVersion in installer
# ---------------------------------------------------------------------------


def test_iss_my_app_version_matches_dasp_version():
    """The Inno Setup script's ``MyAppVersion`` define drives the installer's
    output filename (``SpectralPredict_Setup_py312_<VERSION>.exe``) and the
    AppVersion shown in Add/Remove Programs. Must equal ``__version__``."""
    text = _read("installer/spectral_predict_py312.iss")
    m = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', text)
    assert m, "MyAppVersion #define not found in installer/spectral_predict_py312.iss"
    assert m.group(1) == __version__, (
        f"MyAppVersion in .iss is {m.group(1)!r}; must match "
        f"__version__ ({__version__!r}) — installer filename will drift otherwise"
    )


# ---------------------------------------------------------------------------
# build_installer_py312.py — drives output filename through VERSION constant
# ---------------------------------------------------------------------------


def test_build_installer_imports_version_from_package():
    """``build_installer_py312.py`` must source its VERSION from the package's
    ``__version__`` rather than redefining a literal — otherwise the installer
    output filename can drift from what the running app reports."""
    text = _read("build_installer_py312.py")
    assert "from spectral_predict import __version__ as VERSION" in text, (
        "build_installer_py312.py must derive VERSION from "
        "spectral_predict.__version__"
    )
    # No hardcoded literal at module scope.
    assert not re.search(r'^VERSION\s*=\s*"[^"]+"', text, flags=re.MULTILINE), (
        "build_installer_py312.py still contains a hardcoded VERSION literal"
    )
