#!/usr/bin/env python
"""
PyInstaller build script for Spectral Predict.

This is the ONLY build path. The legacy 3.11 script (build_installer.py) and its
spec no longer exist, so nothing here is parallel to anything. Output:
    dist/SpectralPredict-py312/SpectralPredict-py312.exe

The "py312" in those filenames is a STABLE ARTIFACT IDENTITY, not a claim about
the Python version — it is frozen so existing installations upgrade in place
rather than appearing as a second application. The interpreter actually used is
chosen by BUILD_PYTHON_VERSION below (override with the DASP_BUILD_PYTHON
environment variable), and every version-dependent path derives from it.

NOT a win — the frozen bundle did NOT recover real multiprocessing:
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
    The build venv must exist and contain PyInstaller. Both are pinned in
    requirements-lock.txt:
        <build venv>\\Scripts\\pip install -r requirements-lock.txt

Usage:
    python build_installer_py312.py                  # uses BUILD_PYTHON_VERSION
    DASP_BUILD_PYTHON=312 DASP_ALLOW_LOCK_DRIFT=1 python build_installer_py312.py   # roll back to 3.12

The build refuses to run if the build venv differs from requirements-lock.txt
(scripts/check_env_lock.py); DASP_ALLOW_LOCK_DRIFT=1 overrides that.
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

# APP_NAME is a STABLE ARTIFACT IDENTITY, deliberately decoupled from the Python
# version. Existing installations upgrade in place only while this is unchanged,
# so it stays "py312" regardless of which interpreter builds the bundle.
APP_NAME = "SpectralPredict-py312"
PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"

# The ONE place the build interpreter is chosen. Everything version-dependent
# (venv directory, pythonXY.dll, site-packages for the pandas repair) derives
# from the interpreter this selects, so switching Python versions -- or rolling
# back -- is a single edit here rather than a rename across three build files.
BUILD_PYTHON_VERSION = os.environ.get("DASP_BUILD_PYTHON", "314")
BUILD_VENV = PROJECT_ROOT / f".venv{BUILD_PYTHON_VERSION}"

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
        print(f"ERROR: Pillow not installed in current Python. Run from {BUILD_VENV.name}:")
        print(f"  {BUILD_VENV.name}\\Scripts\\pip install pillow")
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
    """Locate the build interpreter selected by BUILD_PYTHON_VERSION."""
    if IS_WINDOWS:
        venv_python = BUILD_VENV / "Scripts" / "python.exe"
    else:
        venv_python = BUILD_VENV / "bin" / "python"

    if not venv_python.exists():
        raise FileNotFoundError(
            f"{BUILD_VENV.name} not found at {BUILD_VENV}\n"
            "Create it with: install.bat (or install.sh on mac/linux)\n"
            f"Then install PyInstaller: {BUILD_VENV.name}\\Scripts\\pip install pyinstaller\n"
            "Set DASP_BUILD_PYTHON to build against a different interpreter."
        )
    return str(venv_python)


def _query_build_python(python_exe: str) -> tuple[str, Path]:
    """Ask the build interpreter what it actually is.

    Returns (version_tag, site_packages) where version_tag is like "314".

    The value is queried rather than assumed: BUILD_PYTHON_VERSION names a venv
    directory, and a directory name is not evidence of the interpreter inside
    it. A .venv314 built from the wrong Python would otherwise produce a bundle
    verified against the wrong pythonXY.dll.
    """
    out = subprocess.run(
        [
            python_exe,
            "-c",
            "import sys,sysconfig;"
            "print(f'{sys.version_info.major}{sys.version_info.minor}');"
            "print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()  # Preserve spaces inside the site-packages path.
    return out[0], Path(out[1])


def run_pyinstaller() -> bool:
    print_step("Step 2: Running PyInstaller")

    spec_file = PROJECT_ROOT / "spectral_predict_py312.spec"
    if not spec_file.exists():
        print(f"ERROR: Spec file not found: {spec_file}")
        return False

    main_script = PROJECT_ROOT / "spectral_predict_gui_optimized.py"
    if not main_script.exists():
        print(f"ERROR: Main script not found: {main_script}")
        return False

    # Clean this bundle's previous build artifacts.
    prev_dist = DIST_DIR / APP_NAME
    prev_build = PROJECT_ROOT / "build" / APP_NAME
    for d in (prev_dist, prev_build):
        if d.exists():
            print(f"Removing {d} ...")
            shutil.rmtree(d)

    python_exe = _find_build_python()
    py_tag, venv_site_pkgs = _query_build_python(python_exe)
    print(f"Build interpreter: {python_exe} (Python {py_tag[0]}.{py_tag[1:]})")

    # The bundle freezes whatever the build venv holds, so a venv that missed a
    # lockfile update would ship the stale package to every user.
    lock_check = PROJECT_ROOT / "scripts" / "check_env_lock.py"
    if subprocess.run([python_exe, str(lock_check)], cwd=str(PROJECT_ROOT)).returncode != 0:
        if os.environ.get("DASP_ALLOW_LOCK_DRIFT") != "1":
            print(f"ERROR: {BUILD_VENV.name} does not match requirements-lock.txt. Run:")
            print(f"  {BUILD_VENV.name}\\Scripts\\python -m pip install -r requirements-lock.txt")
            print("Set DASP_ALLOW_LOCK_DRIFT=1 to build anyway (e.g. a 3.12 rollback build).")
            return False
        print("WARNING: building from a venv that differs from requirements-lock.txt.")
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
        print(f"ERROR: PyInstaller not installed in {BUILD_VENV.name}.")
        print(f"  {BUILD_VENV.name}\\Scripts\\pip install pyinstaller")
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
        DIST_DIR / APP_NAME / "_internal" / f"python{py_tag}.dll",
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
    repair_targets = [
        Path("pandas") / "util" / "__init__.py",
    ]
    repaired = 0
    for target in repair_targets:
        venv_file = venv_site_pkgs / target
        bundle_file = _internal / target
        if not venv_file.is_file():
            print(f"ERROR: Cannot verify bundled {target}: source file missing: {venv_file}")
            return False
        if not bundle_file.is_file():
            print(f"ERROR: Critical bundled file missing: {bundle_file}")
            return False
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
        # winget installs Inno Setup per-user by default, which is NOT under
        # either Program Files. Omitting this is how a machine with Inno Setup
        # correctly installed still reports "Inno Setup not found".
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
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
    """Run Inno Setup on the .iss script and verify THIS run produced an installer.

    Previously every failure path here returned True and main() discarded the
    result, so a build that produced no installer -- or that failed outright --
    still printed "Build Complete". Worse, stale output was never cleared, so the
    completion banner could advertise an installer left over from an earlier
    build. For a project whose only distribution channel is this installer, that
    is the most expensive possible thing to get wrong silently.

    Set DASP_SKIP_INSTALLER=1 to deliberately build the bundle only.
    """
    print_step("Step 3: Running Inno Setup")

    installer_dir = DIST_DIR / "installer"
    installer_path = installer_dir / f"SpectralPredict_Setup_py312_{VERSION}.exe"

    # Remove stale output FIRST, so nothing downstream can mistake a previous
    # build's installer for this one.
    if installer_path.exists():
        print(f"Removing stale installer: {installer_path}")
        installer_path.unlink()

    if os.environ.get("DASP_SKIP_INSTALLER"):
        print("DASP_SKIP_INSTALLER set — skipping installer creation deliberately.")
        print(f"The bundled app is ready at: {DIST_DIR / APP_NAME / APP_NAME}.exe")
        return True

    iscc = find_inno_setup()
    if iscc is None:
        print("ERROR: Inno Setup not found — cannot create the installer.")
        print("Install from: https://jrsoftware.org/isdl.php")
        print("Or set DASP_SKIP_INSTALLER=1 to build the bundle only.")
        return False

    print(f"Found Inno Setup: {iscc}")

    iss_file = PROJECT_ROOT / "installer" / "spectral_predict_py312.iss"
    if not iss_file.exists():
        print(f"ERROR: ISS file not found: {iss_file}")
        return False

    installer_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [str(iscc), str(iss_file)],
            cwd=str(PROJECT_ROOT / "installer"),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Inno Setup failed with code {e.returncode}")
        return False

    if not installer_path.exists():
        print(f"ERROR: Inno Setup reported success but produced no installer at {installer_path}")
        print("Check the Inno Setup output above — OutputBaseFilename may have drifted.")
        return False

    size_mb = installer_path.stat().st_size / (1024 * 1024)
    print(f"Created: {installer_path}")
    print(f"Size: {size_mb:.1f} MB")
    return True


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

    if IS_WINDOWS and not run_inno_setup():
        print("\nBuild failed at Inno Setup step.")
        return 1

    print_step("Bundle Build Complete")
    print(f"Standalone app: {DIST_DIR / APP_NAME / APP_NAME}.exe")
    # run_inno_setup() deleted any stale installer before building, so reaching
    # here with the file present means THIS run produced it.
    installer_path = DIST_DIR / "installer" / f"SpectralPredict_Setup_py312_{VERSION}.exe"
    if installer_path.exists():
        print(f"Installer:      {installer_path}")
    print()
    print("Next: smoke-test the bundle (--test flag) and launch it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
