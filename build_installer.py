#!/usr/bin/env python
"""
Build automation script for Spectral Predict Windows installer.

This script automates the entire build process:
1. Creates Windows icon from PNG
2. Runs Nuitka to create the standalone application
3. Runs Inno Setup to create the installer

Prerequisites:
    - Python 3.11+ (3.12 recommended)
    - pip install nuitka ordered-set zstandard pillow
    - C compiler (MSVC or MinGW64 - Nuitka can auto-download MinGW64)
    - Inno Setup 6 installed (https://jrsoftware.org/isdl.php)

Usage:
    python build_installer.py

Output:
    dist/installer/SpectralPredict_Setup_0.1.0.exe
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Configuration
VERSION = "0.1.0"
APP_NAME = "SpectralPredict"
PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
INSTALLER_DIR = DIST_DIR / "installer"
# Nuitka output folder name (based on input script name)
NUITKA_OUTPUT_DIR = DIST_DIR / "spectral_predict_gui_optimized.dist"


def print_step(message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}\n")


def create_icon() -> bool:
    """Create Windows ICO file from PNG with multiple resolutions."""
    print_step("Step 1: Creating Windows icon")

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install pillow")
        return False

    png_path = PROJECT_ROOT / "asp_logo_final.png"
    ico_path = PROJECT_ROOT / "asp_logo.ico"

    if not png_path.exists():
        print(f"ERROR: Source PNG not found: {png_path}")
        return False

    try:
        # Load the source image
        img = Image.open(png_path)

        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Create multiple resolutions for Windows icon
        sizes = [16, 32, 48, 64, 128, 256]
        icons = []

        for size in sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            icons.append(resized)

        # Save as ICO with all sizes
        icons[0].save(
            ico_path,
            format='ICO',
            sizes=[(s, s) for s in sizes],
            append_images=icons[1:]
        )

        print(f"Created: {ico_path}")
        print(f"Sizes: {sizes}")
        return True

    except Exception as e:
        print(f"ERROR creating icon: {e}")
        return False


def copy_vcruntime_dlls() -> None:
    """Copy missing VC++ runtime DLLs to the dist folder for standalone distribution."""
    # DLLs needed for OpenMP and other VC++ features
    vcruntime_dlls = ["vcomp140.dll"]
    system32 = Path(r"C:\Windows\System32")

    for dll_name in vcruntime_dlls:
        dest = NUITKA_OUTPUT_DIR / dll_name
        if dest.exists():
            continue  # Already present

        source = system32 / dll_name
        if source.exists():
            shutil.copy2(str(source), str(dest))
            print(f"Copied {dll_name} for standalone distribution")
        else:
            print(f"WARNING: {dll_name} not found in System32")


def run_nuitka() -> bool:
    """Run Nuitka to create the standalone application."""
    print_step("Step 2: Running Nuitka")

    main_script = PROJECT_ROOT / "spectral_predict_gui_optimized.py"
    if not main_script.exists():
        print(f"ERROR: Main script not found: {main_script}")
        return False

    # Check for required icon
    ico_path = PROJECT_ROOT / "asp_logo.ico"
    if not ico_path.exists():
        print(f"ERROR: Icon file not found: {ico_path}")
        print("Run create_icon() first")
        return False

    # Packages that may need explicit inclusion (only if installed)
    optional_packages = [
        "scipy", "sklearn", "xgboost", "lightgbm", "catboost",
        "optuna", "pymoo", "imblearn", "PIL", "pandas", "joblib",
    ]

    # Check which optional packages are installed
    installed_packages = []
    for pkg in optional_packages:
        try:
            __import__(pkg)
            installed_packages.append(pkg)
        except ImportError:
            pass

    # Build Nuitka command
    nuitka_cmd = [
        sys.executable, "-m", "nuitka",
        # Core options
        "--standalone",
        "--assume-yes-for-downloads",  # Auto-download MinGW64 if needed

        # Plugins for specific libraries (matplotlib is auto-enabled, numpy plugin deprecated)
        "--enable-plugin=tk-inter",

        # Windows-specific options
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={ico_path}",

        # Exclude packages that crash Nuitka 4.0 (shap is optional, guarded by try/except)
        "--nofollow-import-to=shap",
    ]

    # Add installed packages
    for pkg in installed_packages:
        nuitka_cmd.append(f"--include-package={pkg}")

    # Always include spectral_predict from src
    nuitka_cmd.append("--include-package=spectral_predict")

    # Include package data for packages that need data files
    # imblearn needs VERSION.txt at runtime
    nuitka_cmd.append("--include-package-data=imblearn")

    # Data files and directories
    nuitka_cmd.extend([
        # Include data files
        f"--include-data-files={PROJECT_ROOT / 'asp_logo_final.png'}=asp_logo_final.png",

        # Include entire directories
        f"--include-data-dir={PROJECT_ROOT / 'example'}=example",
        f"--include-data-dir={PROJECT_ROOT / 'src' / 'spectral_predict'}=src/spectral_predict",

        # Output configuration
        f"--output-dir={DIST_DIR}",
        f"--output-filename={APP_NAME}.exe",

        # The main script
        str(main_script),
    ])

    print("Running Nuitka with options:")
    for opt in nuitka_cmd[3:]:  # Skip python -m nuitka
        print(f"  {opt}")

    try:
        result = subprocess.run(
            nuitka_cmd,
            cwd=str(PROJECT_ROOT),
            check=True
        )

        # Verify output
        exe_path = NUITKA_OUTPUT_DIR / f"{APP_NAME}.exe"
        if exe_path.exists():
            print(f"Created: {exe_path}")
            # Copy missing VC++ runtime DLLs for standalone distribution
            copy_vcruntime_dlls()
            return True
        else:
            # Try alternate naming (Nuitka might use original script name)
            alt_exe = NUITKA_OUTPUT_DIR / "spectral_predict_gui_optimized.exe"
            if alt_exe.exists():
                # Rename to expected name
                target = NUITKA_OUTPUT_DIR / f"{APP_NAME}.exe"
                shutil.move(str(alt_exe), str(target))
                print(f"Created and renamed: {target}")
                # Copy missing VC++ runtime DLLs for standalone distribution
                copy_vcruntime_dlls()
                return True
            print(f"ERROR: Expected output not found: {exe_path}")
            print(f"Contents of {NUITKA_OUTPUT_DIR}:")
            if NUITKA_OUTPUT_DIR.exists():
                for item in NUITKA_OUTPUT_DIR.iterdir():
                    print(f"  {item.name}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Nuitka failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print("ERROR: Nuitka not installed. Run: pip install nuitka ordered-set zstandard")
        return False


def find_inno_setup() -> Path | None:
    """Find Inno Setup installation."""
    # Common installation paths
    candidates = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Inno Setup 6" / "ISCC.exe",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Try to find via PATH
    try:
        result = subprocess.run(
            ["where", "ISCC.exe"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip().split('\n')[0])
    except Exception:
        pass

    return None


def run_inno_setup() -> bool:
    """Run Inno Setup to create the installer."""
    print_step("Step 3: Running Inno Setup")

    iscc = find_inno_setup()
    if iscc is None:
        print("ERROR: Inno Setup not found")
        print("Please install from: https://jrsoftware.org/isdl.php")
        print("Or add ISCC.exe to your PATH")
        return False

    print(f"Found Inno Setup: {iscc}")

    iss_file = PROJECT_ROOT / "installer" / "spectral_predict.iss"
    if not iss_file.exists():
        print(f"ERROR: ISS file not found: {iss_file}")
        return False

    # Create installer output directory
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [str(iscc), str(iss_file)],
            cwd=str(PROJECT_ROOT / "installer"),
            check=True
        )

        # Verify output
        installer_path = INSTALLER_DIR / f"SpectralPredict_Setup_{VERSION}.exe"
        if installer_path.exists():
            size_mb = installer_path.stat().st_size / (1024 * 1024)
            print(f"Created: {installer_path}")
            print(f"Size: {size_mb:.1f} MB")
            return True
        else:
            print(f"ERROR: Expected installer not found: {installer_path}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Inno Setup failed with code {e.returncode}")
        return False


def main() -> int:
    """Main build process."""
    print(f"Building {APP_NAME} v{VERSION} Windows Installer (Nuitka)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python version: {sys.version}")

    # Step 1: Create icon
    if not create_icon():
        print("\nBuild failed at icon creation step")
        return 1

    # Step 2: Run Nuitka
    if not run_nuitka():
        print("\nBuild failed at Nuitka step")
        return 1

    # Step 3: Run Inno Setup
    if not run_inno_setup():
        print("\nBuild failed at Inno Setup step")
        print("\nNote: Nuitka bundle was created successfully.")
        print(f"You can run the app directly: {NUITKA_OUTPUT_DIR / APP_NAME}.exe")
        return 1

    print_step("Build Complete!")
    print(f"Installer: dist\\installer\\SpectralPredict_Setup_{VERSION}.exe")
    print(f"\nTo test the standalone app:")
    print(f"  {NUITKA_OUTPUT_DIR / APP_NAME}.exe")

    return 0


if __name__ == "__main__":
    sys.exit(main())
