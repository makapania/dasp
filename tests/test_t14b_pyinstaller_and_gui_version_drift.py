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


def test_version_info_txt_tuple_structure_and_semver_match():
    """Always-on contract: ``filevers`` and ``prodvers`` must each be 4-int
    tuples whose first three elements match major/minor/patch parsed from
    ``__version__``. Pinning the structural contract independently of the
    pre-release suffix means the test still does work after the project
    cuts a non-beta release (a previous version of this test silently
    no-op'd on non-beta versions — fix-of-fixes from cross-family review)."""
    text = _read("version_info.txt")

    sem = re.match(r"(\d+)\.(\d+)\.(\d+)", __version__)
    assert sem, f"Cannot parse __version__ {__version__!r} as semver"
    major, minor, patch = (int(g) for g in sem.groups())

    for name in ("filevers", "prodvers"):
        m = re.search(rf"{name}=\(([^)]+)\)", text)
        assert m, f"{name} tuple not found in version_info.txt"
        parts = [p.strip() for p in m.group(1).split(",")]
        assert len(parts) == 4, (
            f"{name} must be a 4-int tuple; got ({m.group(1)})"
        )
        for p in parts:
            assert p.isdigit(), (
                f"{name} elements must be integers; got {p!r} in "
                f"({m.group(1)})"
            )
        ints = [int(p) for p in parts]
        assert ints[:3] == [major, minor, patch], (
            f"{name} first three elements {ints[:3]} must match "
            f"__version__ major.minor.patch ({major}.{minor}.{patch})"
        )


def test_version_info_txt_tuple_trailing_element_matches_beta_number():
    """Beta-specific contract: when ``__version__`` carries a ``bN`` suffix,
    the trailing tuple element in ``filevers``/``prodvers`` must equal N.
    Pinning the convention so a beta-string bump that forgets the tuple
    bump fails the test. Skipped (with explicit asserts in
    test_version_info_txt_tuple_structure_and_semver_match) for non-beta
    versions — that release will pick its own tuple-trailing convention."""
    beta_m = re.match(r"\d+\.\d+\.\d+b(\d+)$", __version__)
    if beta_m is None:
        import pytest
        pytest.skip(f"__version__ {__version__!r} is not a beta release")

    beta = int(beta_m.group(1))
    text = _read("version_info.txt")
    for name in ("filevers", "prodvers"):
        m = re.search(rf"{name}=\(([^)]+)\)", text)
        assert m, f"{name} tuple not found in version_info.txt"
        parts = [p.strip() for p in m.group(1).split(",")]
        assert int(parts[3]) == beta, (
            f"{name} trailing element {parts[3]} must equal beta number "
            f"{beta} parsed from __version__ {__version__!r}"
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
# pyproject.toml — wheel metadata version (consumed by setuptools / pip show)
# ---------------------------------------------------------------------------


def test_pyproject_toml_version_matches_dasp_version():
    """``pyproject.toml`` carries an independent ``[project].version`` that
    setuptools uses for wheel metadata. ``pip show spectral-predict`` reads
    from there, not from ``__init__.__version__``. Drift here means an
    install-from-source surface reports a stale version.

    Both DeepSeek V4 Pro Max and GLM 5.1 flagged this as an unpinned drift
    site during T-14b review. Closing the pin here prevents the same drift
    pattern T-14 closed elsewhere."""
    text = _read("pyproject.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert m, "version field not found in pyproject.toml [project] section"
    assert m.group(1) == __version__, (
        f"pyproject.toml version is {m.group(1)!r}; must match "
        f"__version__ ({__version__!r})"
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
