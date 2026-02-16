#!/usr/bin/env python
"""
Cross-platform build script for Spectral Predict (Windows + macOS).

Detects the current platform and runs the appropriate PyInstaller build:
  - Windows: PyInstaller with spectral_predict.spec -> Inno Setup installer
  - macOS:   PyInstaller with spectral_predict_mac.spec -> .app bundle

Prerequisites:
    pip install pyinstaller pillow

Usage:
    python build_installer.py
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Configuration
VERSION = "0.4.0"
APP_NAME = "SpectralPredict"
PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"


def print_step(message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}\n")


# ============================================================================
# Icon creation
# ============================================================================

def create_icon_windows() -> bool:
    """Create Windows ICO file from PNG with multiple resolutions."""
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
        img = Image.open(png_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        sizes = [(s, s) for s in [16, 32, 48, 64, 128, 256]]
        # Save from the largest resolution and let Pillow create sub-sizes
        img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_256.save(ico_path, format='ICO', sizes=sizes)
        ico_size = ico_path.stat().st_size
        print(f"Created: {ico_path} ({ico_size:,} bytes)")
        return True
    except Exception as e:
        print(f"ERROR creating icon: {e}")
        return False


def create_icon_macos() -> bool:
    """Create macOS ICNS file from PNG using sips + iconutil."""
    png_path = PROJECT_ROOT / "asp_logo_final.png"
    icns_path = PROJECT_ROOT / "asp_logo.icns"

    if not png_path.exists():
        print(f"ERROR: Source PNG not found: {png_path}")
        return False

    iconset_dir = PROJECT_ROOT / "asp_logo.iconset"
    try:
        # Create iconset directory
        if iconset_dir.exists():
            shutil.rmtree(iconset_dir)
        iconset_dir.mkdir()

        # macOS icon sizes: name -> (width, scale)
        icon_sizes = {
            "icon_16x16.png": 16,
            "icon_16x16@2x.png": 32,
            "icon_32x32.png": 32,
            "icon_32x32@2x.png": 64,
            "icon_128x128.png": 128,
            "icon_128x128@2x.png": 256,
            "icon_256x256.png": 256,
            "icon_256x256@2x.png": 512,
            "icon_512x512.png": 512,
            "icon_512x512@2x.png": 1024,
        }

        for name, size in icon_sizes.items():
            dest = iconset_dir / name
            subprocess.run(
                ["sips", "-z", str(size), str(size), str(png_path),
                 "--out", str(dest)],
                capture_output=True,
                check=True,
            )

        # Convert iconset to icns
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)],
            check=True,
        )
        print(f"Created: {icns_path}")
        return True

    except FileNotFoundError:
        print("WARNING: sips/iconutil not found (not on macOS?)")
        print("Attempting Pillow fallback...")
        return _create_icns_pillow(png_path, icns_path)
    except subprocess.CalledProcessError as e:
        print(f"ERROR creating icns: {e}")
        return False
    finally:
        # Clean up iconset directory
        if iconset_dir.exists():
            shutil.rmtree(iconset_dir)


def _create_icns_pillow(png_path: Path, icns_path: Path) -> bool:
    """Fallback: create a basic ICNS using Pillow (less optimal but works)."""
    try:
        from PIL import Image

        img = Image.open(png_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Pillow can save ICNS directly on some platforms
        img.save(icns_path, format='ICNS')
        print(f"Created (Pillow fallback): {icns_path}")
        return True
    except Exception as e:
        print(f"WARNING: Could not create .icns icon: {e}")
        print("Build will proceed without a custom icon.")
        return True  # Non-fatal — app will use default icon


def create_icon() -> bool:
    """Create platform-appropriate icon."""
    print_step("Step 1: Creating application icon")

    if IS_MACOS:
        return create_icon_macos()
    else:
        return create_icon_windows()


# ============================================================================
# PyInstaller build
# ============================================================================

def _find_build_python() -> str:
    """Find the .venv311 Python interpreter for building.

    Always uses .venv311 which has PyInstaller and all dependencies.
    Fails if the venv is missing or broken.
    """
    if IS_WINDOWS:
        venv_python = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv311" / "bin" / "python"

    if not venv_python.exists():
        raise FileNotFoundError(
            f".venv311 not found at {venv_python.parent.parent}\n"
            "Create it with: python3.11 -m venv .venv311 && .venv311/Scripts/pip install -r requirements.txt"
        )

    return str(venv_python)


def run_pyinstaller() -> bool:
    """Run PyInstaller with the platform-appropriate spec file."""
    print_step("Step 2: Running PyInstaller")

    if IS_MACOS:
        spec_file = PROJECT_ROOT / "spectral_predict_mac.spec"
    else:
        spec_file = PROJECT_ROOT / "spectral_predict.spec"

    if not spec_file.exists():
        print(f"ERROR: Spec file not found: {spec_file}")
        return False

    main_script = PROJECT_ROOT / "spectral_predict_gui_optimized.py"
    if not main_script.exists():
        print(f"ERROR: Main script not found: {main_script}")
        return False

    # Clean slate: nuke dist/ and build/ to prevent stale/corrupt artifacts
    build_dir = PROJECT_ROOT / "build"
    for d in (DIST_DIR, build_dir):
        if d.exists():
            print(f"Removing {d} ...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    print("Clean build directories ready.")

    python_exe = _find_build_python()
    cmd = [
        python_exe, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]

    print(f"Spec file: {spec_file.name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: PyInstaller failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print("ERROR: PyInstaller not installed. Run: pip install pyinstaller")
        return False

    # Verify output
    if IS_MACOS:
        app_path = DIST_DIR / f"{APP_NAME}.app"
        if app_path.exists():
            print(f"Created: {app_path}")
            return True
        # Might be inside a subfolder
        app_path_alt = DIST_DIR / APP_NAME / f"{APP_NAME}.app"
        if app_path_alt.exists():
            print(f"Created: {app_path_alt}")
            return True
        print(f"ERROR: Expected output not found: {app_path}")
        return False
    else:
        exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
        if not exe_path.exists():
            print(f"ERROR: Expected output not found: {exe_path}")
            if (DIST_DIR / APP_NAME).exists():
                print(f"Contents of {DIST_DIR / APP_NAME}:")
                for item in sorted((DIST_DIR / APP_NAME).iterdir())[:20]:
                    print(f"  {item.name}")
            return False

        print(f"Created: {exe_path}")

        # Post-build verification: check critical files exist
        critical_files = [
            exe_path,
            DIST_DIR / APP_NAME / "_internal" / "python311.dll",
            DIST_DIR / APP_NAME / "_internal" / "python3.dll",
        ]
        all_ok = True
        print("\nPost-build verification:")
        for f in critical_files:
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  PASS  {f.relative_to(DIST_DIR)}  ({size_mb:.1f} MB)")
            else:
                print(f"  FAIL  {f.relative_to(DIST_DIR)}  — MISSING")
                all_ok = False

        if not all_ok:
            print("\nERROR: Post-build verification FAILED — critical files missing.")
            print("The built exe will likely crash on launch.")
            return False

        print("\nAll critical files verified.")
        return True


# ============================================================================
# Post-build: Windows (Inno Setup)
# ============================================================================

def find_inno_setup() -> Path | None:
    """Find Inno Setup installation."""
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
            text=True,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip().split('\n')[0])
    except Exception:
        pass

    return None


def run_inno_setup() -> bool:
    """Run Inno Setup to create the Windows installer."""
    print_step("Step 3: Running Inno Setup")

    iscc = find_inno_setup()
    if iscc is None:
        print("WARNING: Inno Setup not found — skipping installer creation.")
        print("Install from: https://jrsoftware.org/isdl.php")
        print(f"\nYou can still run the app directly:")
        print(f"  {DIST_DIR / APP_NAME / APP_NAME}.exe")
        return True  # Non-fatal

    print(f"Found Inno Setup: {iscc}")

    iss_file = PROJECT_ROOT / "installer" / "spectral_predict.iss"
    if not iss_file.exists():
        print(f"WARNING: ISS file not found: {iss_file}")
        print("Skipping installer creation.")
        return True  # Non-fatal

    installer_dir = DIST_DIR / "installer"
    installer_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [str(iscc), str(iss_file)],
            cwd=str(PROJECT_ROOT / "installer"),
            check=True,
        )

        installer_path = installer_dir / f"SpectralPredict_Setup_{VERSION}.exe"
        if installer_path.exists():
            size_mb = installer_path.stat().st_size / (1024 * 1024)
            print(f"Created: {installer_path}")
            print(f"Size: {size_mb:.1f} MB")
        else:
            print("WARNING: Installer exe not found at expected path.")

        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Inno Setup failed with code {e.returncode}")
        print("PyInstaller bundle was created successfully.")
        return True  # Non-fatal


# ============================================================================
# Post-build: macOS (codesign)
# ============================================================================

def codesign_macos() -> bool:
    """Ad-hoc codesign the .app bundle for local use."""
    print_step("Step 3: Ad-hoc codesigning .app bundle")

    # Find the .app
    app_path = DIST_DIR / f"{APP_NAME}.app"
    if not app_path.exists():
        app_path = DIST_DIR / APP_NAME / f"{APP_NAME}.app"
    if not app_path.exists():
        print("WARNING: .app bundle not found for codesigning.")
        return True  # Non-fatal

    try:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app_path)],
            check=True,
        )
        print(f"Signed: {app_path}")
        return True
    except FileNotFoundError:
        print("WARNING: codesign not found — skipping.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: codesign failed (code {e.returncode}) — app may still work.")
        return True  # Non-fatal


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Main build process."""
    platform_name = "macOS" if IS_MACOS else "Windows"
    arch = platform.machine()
    print(f"Building {APP_NAME} v{VERSION} for {platform_name} ({arch})")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python version: {sys.version}")

    if not IS_MACOS and not IS_WINDOWS:
        print(f"WARNING: Untested platform ({sys.platform}). Proceeding anyway.")

    # Step 1: Create icon
    if not create_icon():
        print("\nBuild failed at icon creation step.")
        return 1

    # Step 2: Run PyInstaller
    if not run_pyinstaller():
        print("\nBuild failed at PyInstaller step.")
        return 1

    # Step 3: Platform-specific post-build
    if IS_MACOS:
        codesign_macos()

        app_path = DIST_DIR / f"{APP_NAME}.app"
        if not app_path.exists():
            app_path = DIST_DIR / APP_NAME / f"{APP_NAME}.app"

        print_step("Build Complete!")
        print(f"App bundle: {app_path}")
        print(f"\nTo run:")
        print(f"  open {app_path}")
        print(f"  # or: {app_path}/Contents/MacOS/{APP_NAME}")
        print(f"\nIf macOS blocks it (unsigned):")
        print(f"  Right-click > Open, or:")
        print(f"  System Settings > Privacy & Security > 'Open Anyway'")
    else:
        run_inno_setup()

        print_step("Build Complete!")
        print(f"Standalone app: {DIST_DIR / APP_NAME / APP_NAME}.exe")
        installer_path = DIST_DIR / "installer" / f"SpectralPredict_Setup_{VERSION}.exe"
        if installer_path.exists():
            print(f"Installer: {installer_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
