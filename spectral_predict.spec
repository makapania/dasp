# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Spectral Predict.

Build with:
    .venv311\Scripts\pyinstaller spectral_predict.spec

Test with:
    dist\SpectralPredict\_internal\python.exe test_bundled_app.py

Output:
    dist/SpectralPredict/SpectralPredict.exe
"""

import sys
import glob
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Project root
project_root = Path(SPECPATH)

# Virtual environment site-packages
venv_path = project_root / '.venv311'
site_packages = venv_path / 'Lib' / 'site-packages'

print(f"Project root: {project_root}")
print(f"Site packages: {site_packages}")

# ============================================================================
# BRUTE-FORCE COLLECTION
# Instead of relying on collect_all() which fails silently, we explicitly
# find and include ALL compiled extensions, DLLs, and metadata.
# ============================================================================

all_datas = []
all_binaries = []
all_hiddenimports = []

# --- 1. Collect using PyInstaller's collect_all for known packages ---
packages_to_collect = [
    'matplotlib',
    'catboost',
    'xgboost',
    'lightgbm',
    'imblearn',
    'sklearn',
    'scipy',
    'numpy',
    'numba',
    'llvmlite',
    'pandas',
    'PIL',
    'optuna',
    'sqlalchemy',
    'joblib',
    'platformdirs',
    'openpyxl',
    'cma',
    'shap',
    'tabulate',  # Required by pandas.to_markdown()
]

for pkg in packages_to_collect:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hiddenimports.extend(hiddenimports)
        print(f"[OK] Collected {pkg}: {len(datas)} datas, {len(binaries)} binaries")
    except Exception as e:
        print(f"[WARN] Warning: collect_all failed for {pkg}: {e}")

# --- 2. MANUALLY collect pymoo (collect_all crashes on it) ---
pymoo_path = site_packages / 'pymoo'
if pymoo_path.exists():
    # Recursively add ALL pymoo files
    for root, dirs, files in os.walk(pymoo_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(site_packages)
        for f in files:
            src = root_path / f
            all_datas.append((str(src), str(rel_path)))
    print(f"[OK] Manually collected pymoo from {pymoo_path}")
else:
    print(f"[WARN] Warning: pymoo not found at {pymoo_path}")

# --- 3. MANUALLY collect moocore (pymoo dependency) ---
moocore_path = site_packages / 'moocore'
if moocore_path.exists():
    for root, dirs, files in os.walk(moocore_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(site_packages)
        for f in files:
            src = root_path / f
            all_datas.append((str(src), str(rel_path)))
    print(f"[OK] Manually collected moocore from {moocore_path}")

# --- 4. Find ALL .pyd files (compiled extensions) ---
print("\n--- Finding all .pyd files ---")
pyd_count = 0
for pyd_file in site_packages.glob('**/*.pyd'):
    rel_dir = pyd_file.parent.relative_to(site_packages)
    all_binaries.append((str(pyd_file), str(rel_dir)))
    pyd_count += 1
print(f"[OK] Found {pyd_count} .pyd files")

# --- 5. Find ALL DLLs in package lib folders ---
print("\n--- Finding package DLLs ---")
dll_count = 0
# Common DLL locations
dll_patterns = [
    '*/lib/*.dll',
    '*/*.libs/*.dll',
    '*/.libs/*.dll',
    '*/libs/*.dll',
]
for pattern in dll_patterns:
    for dll_file in site_packages.glob(pattern):
        rel_dir = dll_file.parent.relative_to(site_packages)
        all_binaries.append((str(dll_file), str(rel_dir)))
        dll_count += 1
        print(f"  Found DLL: {dll_file.name} -> {rel_dir}")
print(f"[OK] Found {dll_count} package DLLs")

# --- 6. Copy ALL .dist-info folders (for importlib.metadata) ---
print("\n--- Collecting dist-info metadata ---")
dist_info_count = 0
for dist_info in site_packages.glob('*.dist-info'):
    # Add the entire dist-info folder
    for f in dist_info.iterdir():
        if f.is_file():
            all_datas.append((str(f), dist_info.name))
    dist_info_count += 1
print(f"[OK] Found {dist_info_count} dist-info folders")

# --- 7. Explicitly add VERSION.txt files (common missing files) ---
print("\n--- Finding VERSION.txt files ---")
version_files = list(site_packages.glob('**/VERSION.txt')) + \
                list(site_packages.glob('**/VERSION')) + \
                list(site_packages.glob('**/version.txt'))
for vf in version_files:
    rel_dir = vf.parent.relative_to(site_packages)
    all_datas.append((str(vf), str(rel_dir)))
    print(f"  Found: {vf.relative_to(site_packages)}")
print(f"[OK] Found {len(version_files)} version files")

# --- 8. Collect all submodules for problematic packages ---
print("\n--- Collecting submodules ---")
submodule_packages = ['pymoo', 'imblearn', 'moocore', 'shap']
for pkg in submodule_packages:
    try:
        submods = collect_submodules(pkg)
        all_hiddenimports.extend(submods)
        print(f"[OK] Collected {len(submods)} submodules for {pkg}")
    except Exception as e:
        print(f"[WARN] Warning: Could not collect submodules for {pkg}: {e}")

# ============================================================================
# HIDDEN IMPORTS
# ============================================================================

explicit_hiddenimports = [
    # Core scientific libraries
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'pandas',
    'scipy',
    'scipy.signal',
    'scipy.special',
    'scipy.optimize',
    'scipy.stats',
    'scipy.linalg',
    'scipy.sparse',
    'scipy.sparse.linalg',
    'scipy.sparse.csgraph',
    'scipy.spatial',
    'scipy.interpolate',
    'scipy.fftpack',
    'scipy.ndimage',
    'scipy.integrate',
    # sklearn modules
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'sklearn.cross_decomposition',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.svm',
    'sklearn.neural_network',
    'sklearn.pipeline',
    'sklearn.decomposition',
    # ML boosting libraries
    'lightgbm',
    'xgboost',
    'catboost',
    # imbalanced-learn
    'imblearn',
    'imblearn.pipeline',
    'imblearn.over_sampling',
    'imblearn.under_sampling',
    # Multi-objective optimization - EXPLICIT
    'pymoo',
    'pymoo.core',
    'pymoo.core.problem',
    'pymoo.core.algorithm',
    'pymoo.core.population',
    'pymoo.core.result',
    'pymoo.algorithms',
    'pymoo.algorithms.moo',
    'pymoo.algorithms.moo.nsga2',
    'pymoo.optimize',
    'pymoo.util',
    'pymoo.util.nds',
    'pymoo.util.nds.non_dominated_sorting',
    'pymoo.operators',
    'pymoo.operators.crossover',
    'pymoo.operators.mutation',
    'pymoo.operators.sampling',
    'pymoo.operators.selection',
    'pymoo.termination',
    'moocore',
    # Model interpretability
    'shap',
    'shap.explainers',
    'shap.plots',
    # Utilities
    'joblib',
    'numba',
    'llvmlite',
    'platformdirs',
    # GUI and plotting
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageOps',
    'PIL.ImageFont',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backend_bases',
    # Standard library extensions
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    # Excel support
    'openpyxl',
    # Database
    'sqlalchemy',
    'sqlalchemy.pool',
]

all_hiddenimports.extend(explicit_hiddenimports)
# Remove duplicates
all_hiddenimports = list(set(all_hiddenimports))

print(f"\n--- Summary ---")
print(f"Total datas: {len(all_datas)}")
print(f"Total binaries: {len(all_binaries)}")
print(f"Total hidden imports: {len(all_hiddenimports)}")

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    ['spectral_predict_gui_optimized.py'],
    pathex=[str(project_root)],
    binaries=all_binaries,
    # Disable bytecode optimization to fix scipy.stats compatibility
    optimize=0,
    datas=[
        # Bundle the source modules
        ('src/spectral_predict', 'src/spectral_predict'),
        # Bundle logo file
        ('asp_logo_final.png', '.'),
        # Bundle example data (for testing)
        ('example/BoneCollagen.csv', 'example'),
    ] + all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude test modules
        'pytest',
        # Exclude development tools
        'IPython',
        'jupyter',
        'notebook',
        # Exclude unnecessary backends
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True,  # Don't archive .pyc files - fixes scipy compatibility
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectralPredict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # TEMP: Enable console to see errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='asp_logo.ico',
    version='version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    # Exclude scientific libraries from UPX compression to prevent corruption
    upx_exclude=[
        'scipy',
        'numpy',
        'sklearn',
        'xgboost',
        'lightgbm',
        'catboost',
        'numba',
        'llvmlite',
    ],
    name='SpectralPredict',
)
