#!/usr/bin/env python
"""
PyInstaller build script for Spectral Predict — Python 3.12 build.

PARALLEL to build_installer.py (the legacy 3.11 build). This script never
modifies the 3.11 build path; it only produces the 3.12 bundle in:
    dist/SpectralPredict-py312/SpectralPredict-py312.exe

Wins over the 3.11 build:
  - Python 3.12 + PyInstaller 6.x (newer wheels, fewer workarounds)
  - Pick up newly-required deps (Pillow, shap, jcamp, pybaselines, vendor formats)
  - Cleaner dependency closure, fewer hidden-import patches

NOT a win — the 3.12 bundle did NOT recover real multiprocessing:
  src/spectral_predict/search.py:_frozen_needs_threading_fallback() returns
  True for ANY frozen build regardless of Python version. The fork-bomb /
  argv-parse crash in PyInstaller's spawned-child runtime hook is not Python-
  3.11-specific, so the loky→threading fallback still applies in the 3.12
  bundle. Practical impact: numpy/sklearn/lightgbm/xgboost training still gets
  thread-parallel speedup (those C extensions release the GIL), but pure-
  Python parallel loops (pymoo NSGA-II, GA-PLS evaluation) fall back to
  single-core. Recovering true multiprocessing in the bundle would require
  fixing the runtime hook itself, not bumping Python.

Prerequisites:
    .venv312\\Scripts\\pip install pyinstaller

Usage:
    python build_installer_py312.py

This is now the shipped path. The 3.11 build remains in-repo only as a
fallback during the beta soak (see docs/PROJECT_STATUS.md for retirement plan).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Read VERSION from the package's __version__ so the build artefact filenames
# can never drift from the runtime version (T-14 / T-14b).
sys.path.insert(0, str(Path(__file__).parent / "src"))
from spectral_predict import __version__ as VERSION  # noqa: E402

APP_NAME = "SpectralPredict-py312"
PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"


def print_step(message: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}\n")


# ============================================================================
# Icon creation (reused from 3.11 build approach)
# ============================================================================

def create_icon_windows() -> bool:
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed in current Python. Run from .venv312:")
        print("  .venv312\\Scripts\\pip install pillow")
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
        img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_256.save(ico_path, format='ICO', sizes=sizes)
        print(f"Created: {ico_path} ({ico_path.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        print(f"ERROR creating icon: {e}")
        return False


def create_icon() -> bool:
    print_step("Step 1: Creating application icon")
    if IS_MACOS:
        print("INFO: macOS 3.12 build path not implemented yet — skipping icon.")
        return True
    return create_icon_windows()


# ============================================================================
# PyInstaller build
# ============================================================================

def _find_build_python() -> str:
    """Locate the .venv312 Python interpreter."""
    if IS_WINDOWS:
        venv_python = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv312" / "bin" / "python"

    if not venv_python.exists():
        raise FileNotFoundError(
            f".venv312 not found at {venv_python.parent.parent}\n"
            "Create it with: install.bat (or install.sh on mac/linux)\n"
            "Then install PyInstaller: .venv312\\Scripts\\pip install pyinstaller"
        )
    return str(venv_python)


def run_pyinstaller() -> bool:
    print_step("Step 2: Running PyInstaller (Python 3.12)")

    spec_file = PROJECT_ROOT / "spectral_predict_py312.spec"
    if not spec_file.exists():
        print(f"ERROR: Spec file not found: {spec_file}")
        return False

    main_script = PROJECT_ROOT / "spectral_predict_gui_optimized.py"
    if not main_script.exists():
        print(f"ERROR: Main script not found: {main_script}")
        return False

    # Clean previous 3.12 build artifacts only — leave the 3.11 dist/ alone.
    py312_dist = DIST_DIR / APP_NAME
    py312_build = PROJECT_ROOT / "build" / APP_NAME
    for d in (py312_dist, py312_build):
        if d.exists():
            print(f"Removing {d} ...")
            shutil.rmtree(d)

    python_exe = _find_build_python()
    cmd = [
        python_exe, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]
    print(f"Spec file: {spec_file.name}")
    print(f"Python: {python_exe}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: PyInstaller failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print("ERROR: PyInstaller not installed in .venv312.")
        print("  .venv312\\Scripts\\pip install pyinstaller")
        return False

    if IS_MACOS:
        print("INFO: macOS .app verification not implemented for the 3.12 build yet.")
        return True

    exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
    if not exe_path.exists():
        print(f"ERROR: Expected output not found: {exe_path}")
        if (DIST_DIR / APP_NAME).exists():
            print(f"Contents of {DIST_DIR / APP_NAME}:")
            for item in sorted((DIST_DIR / APP_NAME).iterdir())[:20]:
                print(f"  {item.name}")
        return False

    print(f"Created: {exe_path}")

    # Post-build verification
    critical_files = [
        exe_path,
        DIST_DIR / APP_NAME / "_internal" / "python312.dll",
        DIST_DIR / APP_NAME / "_internal" / "python3.dll",
    ]
    all_ok = True
    print("\nPost-build verification:")
    for f in critical_files:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  PASS  {f.relative_to(DIST_DIR)}  ({size_mb:.1f} MB)")
        else:
            print(f"  FAIL  {f.relative_to(DIST_DIR)}  -- MISSING")
            all_ok = False

    if not all_ok:
        print("\nERROR: Post-build verification FAILED -- critical files missing.")
        return False

    print("\nAll critical files verified.")

    # Post-build: strip torch artifacts that leak through binary dependency
    # analysis.  Filtering a.binaries / a.datas inside the spec file causes
    # PyInstaller TOC reordering which corrupts unrelated data files (pandas).
    # Doing it here after COLLECT is safe and achieves the same size savings.
    _internal = DIST_DIR / APP_NAME / "_internal"
    torch_dirs = ["torch", "torchvision", "torchaudio"]
    removed_bytes = 0
    for td in torch_dirs:
        target = _internal / td
        if target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target)
            removed_bytes += size
            print(f"  Removed {td}/ ({size / (1024*1024):.0f} MB)")
    for dd in sorted(_internal.glob("torch*")):
        if dd.is_dir() and any(dd.name.startswith(t) for t in torch_dirs):
            size = sum(f.stat().st_size for f in dd.rglob("*") if f.is_file())
            shutil.rmtree(dd)
            removed_bytes += size
            print(f"  Removed {dd.name}/ ({size / (1024*1024):.0f} MB)")
    if removed_bytes:
        print(f"  Total torch savings: {removed_bytes / (1024*1024):.0f} MB")

    # Post-build repair: PyInstaller occasionally overwrites pandas/util/__init__.py
    # with vendored packaging/_structures.py content due to a dist-info collection
    # collision. This is intermittent (hits roughly 1 in 3 rebuilds in testing).
    # The symptom is ImportError: cannot import name 'capitalize_first_letter'
    # from 'pandas.util' at bundle launch. Detect by byte-comparing against the
    # source venv and restore when mismatched.
    print("\nVerifying critical bundled files against venv ...")
    venv_site_pkgs = PROJECT_ROOT / ".venv312" / "Lib" / "site-packages"
    repair_targets = [
        Path("pandas") / "util" / "__init__.py",
    ]
    repaired = 0
    for target in repair_targets:
        venv_file = venv_site_pkgs / target
        bundle_file = _internal / target
        if not venv_file.exists() or not bundle_file.exists():
            continue
        if venv_file.read_bytes() != bundle_file.read_bytes():
            shutil.copy2(venv_file, bundle_file)
            print(f"  [REPAIR] Restored {target} from venv (TOC collision detected)")
            repaired += 1
    if repaired == 0:
        print(f"  All {len(repair_targets)} checked file(s) match venv — no repair needed.")

    return True


# ============================================================================
# Inno Setup (optional — produces a single-file installer.exe)
# ============================================================================

def find_inno_setup() -> Path | None:
    """Locate Inno Setup 6's ISCC.exe; return None if not installed."""
    candidates = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Inno Setup 6" / "ISCC.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    try:
        result = subprocess.run(["where", "ISCC.exe"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def run_inno_setup() -> bool:
    """Run Inno Setup on the 3.12 .iss script. Non-fatal if ISCC is missing."""
    print_step("Step 3: Running Inno Setup (optional)")

    iscc = find_inno_setup()
    if iscc is None:
        print("INFO: Inno Setup not found — skipping installer creation.")
        print("Install from: https://jrsoftware.org/isdl.php")
        print(f"\nThe bundled app is ready at: {DIST_DIR / APP_NAME / APP_NAME}.exe")
        return True  # non-fatal

    print(f"Found Inno Setup: {iscc}")

    iss_file = PROJECT_ROOT / "installer" / "spectral_predict_py312.iss"
    if not iss_file.exists():
        print(f"WARNING: ISS file not found: {iss_file}")
        print("Skipping installer creation.")
        return True  # non-fatal

    installer_dir = DIST_DIR / "installer"
    installer_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [str(iscc), str(iss_file)],
            cwd=str(PROJECT_ROOT / "installer"),
            check=True,
        )
        installer_path = installer_dir / f"SpectralPredict_Setup_py312_{VERSION}.exe"
        if installer_path.exists():
            size_mb = installer_path.stat().st_size / (1024 * 1024)
            print(f"Created: {installer_path}")
            print(f"Size: {size_mb:.1f} MB")
        else:
            print("WARNING: installer exe not found at expected path — check Inno Setup output above.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Inno Setup failed with code {e.returncode}")
        print("The bundled app folder is still usable; only the installer wrapper failed.")
        return True  # non-fatal — don't fail the build


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    platform_name = "macOS" if IS_MACOS else ("Windows" if IS_WINDOWS else sys.platform)
    arch = platform.machine()
    print(f"Building {APP_NAME} v{VERSION} for {platform_name} ({arch})")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python version (build orchestrator): {sys.version}")

    if not create_icon():
        print("\nBuild failed at icon creation step.")
        return 1

    if not run_pyinstaller():
        print("\nBuild failed at PyInstaller step.")
        return 1

    if IS_WINDOWS:
        run_inno_setup()

    print_step("3.12 Bundle Build Complete")
    print(f"Standalone app: {DIST_DIR / APP_NAME / APP_NAME}.exe")
    installer_path = DIST_DIR / "installer" / f"SpectralPredict_Setup_py312_{VERSION}.exe"
    if installer_path.exists():
        print(f"Installer:      {installer_path}")
    print()
    print("Next: smoke-test the bundle (--test flag) and launch it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
