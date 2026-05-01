"""
Resource path utilities for frozen applications (Nuitka/PyInstaller).

This module provides utilities to locate resources (logos, data files, etc.)
in both development mode and when bundled with Nuitka or PyInstaller.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """
    Check if running as a frozen application (Nuitka or PyInstaller).

    Nuitka sets __compiled__ and __nuitka_binary_dir.
    PyInstaller sets sys.frozen and sys._MEIPASS.
    """
    # Nuitka standalone mode
    if "__compiled__" in globals():
        return True
    # PyInstaller
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return True
    return False


def get_base_path() -> Path:
    """
    Get the base path for resource files.

    In development: Returns the project root directory
    In Nuitka frozen mode: Returns the directory containing the executable
    In PyInstaller frozen mode: Returns _MEIPASS (bundle directory)
    """
    # Nuitka standalone mode - resources are in the same directory as the executable
    if "__compiled__" in globals():
        return Path(sys.executable).parent

    # PyInstaller frozen mode
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)

    # Development mode: return the directory containing this file's parent (project root)
    return Path(__file__).parent.parent.parent


def get_resource_path(relative_path: str | Path) -> Path:
    """
    Get the absolute path to a resource file.

    Args:
        relative_path: Path relative to the project root (e.g., "asp_logo_final.png")

    Returns:
        Absolute path to the resource, valid in both development and frozen modes.
    """
    base = get_base_path()
    return base / relative_path


def get_src_path() -> Path:
    """Get the path to the src directory."""
    # Nuitka standalone mode
    if "__compiled__" in globals():
        return Path(sys.executable).parent / "src"

    # PyInstaller frozen mode
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / "src"

    # Development mode
    return Path(__file__).parent.parent


def get_example_path() -> Path:
    """Get the path to the example data directory."""
    return get_base_path() / "example"


def get_user_data_dir() -> Path:
    """
    Per-user writeable directory for runtime state (logs, Optuna SQLite stores).

    Returns the platform-conventional user-data location and creates it if
    missing. Bundled installs of dasp typically land under read-only
    `Program Files\\dasp\\` on Windows, so runtime artifacts must live
    elsewhere.

    - Windows: `%LOCALAPPDATA%\\dasp\\` (defaults to `~\\AppData\\Local\\dasp`)
    - macOS:   `~/Library/Application Support/dasp/`
    - Linux:   `$XDG_DATA_HOME/dasp/` if set, else `~/.local/share/dasp/`
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        root = Path(xdg) if xdg else (Path.home() / ".local" / "share")

    user_dir = root / "dasp"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_log_dir() -> Path:
    """Per-run log file directory under the user data dir. Created on demand."""
    log_dir = get_user_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_user_optuna_dir() -> Path:
    """Per-run Optuna SQLite storage directory under the user data dir."""
    optuna_dir = get_user_data_dir() / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return optuna_dir
