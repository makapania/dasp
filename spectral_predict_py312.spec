# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Spectral Predict — Python 3.12 build (experimental).

This is a PARALLEL build to the production 3.11 spec (spectral_predict.spec).
Goals vs 3.11 build:
  - Use Python 3.12 (newer wheels, fewer workarounds)
  - Recover real multiprocessing (loky backend instead of threading fallback)
  - Pick up newly-required deps (Pillow, shap, jcamp, pybaselines, vendor formats)

Build with:
    .venv312\\Scripts\\pyinstaller spectral_predict_py312.spec

Output:
    dist/SpectralPredict-py312/SpectralPredict-py312.exe

Do NOT modify the 3.11 spec to mirror this — keep them independent so the
production build path is never destabilized by 3.12 experiments.
"""

import sys
import glob
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

project_root = Path(SPECPATH)
venv_path = project_root / '.venv312'
site_packages = venv_path / 'Lib' / 'site-packages'

print(f"[py312] Project root: {project_root}")
print(f"[py312] Site packages: {site_packages}")

all_datas = []
all_binaries = []
all_hiddenimports = []

# --- 1. Collect using PyInstaller's collect_all for known packages ---
# Includes the new required deps from the 2026-04-17 audit:
#   jcamp, pybaselines, specdal, brukeropus, specio, spectrochempy_omnic
packages_to_collect = [
    'matplotlib',
    'catboost',
    'xgboost',
    'lightgbm',
    'imblearn',
    'sklearn',
    'scipy',
    'numpy',
    'pandas',
    'PIL',
    'optuna',
    'sqlalchemy',
    'joblib',
    'platformdirs',
    'openpyxl',
    'cma',
    'shap',
    'numba',
    'llvmlite',
    'tabulate',
    # New required deps in pyproject (2026-04-17 audit)
    'jcamp',
    'pybaselines',
    'spc_io',
    'specdal',
    'brukeropus',
    'specio_py310',
    'spectrochempy_omnic',
    'tksheet',
    'requests',
]

for pkg in packages_to_collect:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hiddenimports.extend(hiddenimports)
        print(f"[OK] Collected {pkg}: {len(datas)} datas, {len(binaries)} binaries")
    except Exception as e:
        print(f"[WARN] collect_all failed for {pkg}: {e}")

# --- 2. MANUALLY collect pymoo (collect_all crashes on it) ---
pymoo_path = site_packages / 'pymoo'
if pymoo_path.exists():
    for root, dirs, files in os.walk(pymoo_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(site_packages)
        for f in files:
            src = root_path / f
            all_datas.append((str(src), str(rel_path)))
    print(f"[OK] Manually collected pymoo from {pymoo_path}")
else:
    print(f"[WARN] pymoo not found at {pymoo_path}")

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
    if any(part in {'torch', 'torchvision', 'torchaudio'} for part in pyd_file.parts):
        continue
    rel_dir = pyd_file.parent.relative_to(site_packages)
    all_binaries.append((str(pyd_file), str(rel_dir)))
    pyd_count += 1
print(f"[OK] Found {pyd_count} .pyd files")

# --- 5. Find ALL DLLs in package lib folders ---
print("\n--- Finding package DLLs ---")
dll_count = 0
dll_patterns = [
    '*/lib/*.dll',
    '*/*.libs/*.dll',
    '*/.libs/*.dll',
    '*/libs/*.dll',
]
for pattern in dll_patterns:
    for dll_file in site_packages.glob(pattern):
        if any(part in {'torch', 'torchvision', 'torchaudio'} for part in dll_file.parts):
            continue
        rel_dir = dll_file.parent.relative_to(site_packages)
        all_binaries.append((str(dll_file), str(rel_dir)))
        dll_count += 1
        print(f"  Found DLL: {dll_file.name} -> {rel_dir}")
print(f"[OK] Found {dll_count} package DLLs")

# --- 6. Copy ALL .dist-info folders (for importlib.metadata) ---
print("\n--- Collecting dist-info metadata ---")
dist_info_count = 0
for dist_info in site_packages.glob('*.dist-info'):
    for f in dist_info.iterdir():
        if f.is_file():
            all_datas.append((str(f), dist_info.name))
    dist_info_count += 1
print(f"[OK] Found {dist_info_count} dist-info folders")

# --- 7. Explicitly add VERSION.txt files ---
print("\n--- Finding VERSION.txt files ---")
version_files = list(site_packages.glob('**/VERSION.txt')) + \
                list(site_packages.glob('**/VERSION')) + \
                list(site_packages.glob('**/version.txt'))
for vf in version_files:
    rel_dir = vf.parent.relative_to(site_packages)
    all_datas.append((str(vf), str(rel_dir)))
print(f"[OK] Found {len(version_files)} version files")

# --- 8. Collect all submodules for problematic packages ---
print("\n--- Collecting submodules ---")
submodule_packages = ['pymoo', 'imblearn', 'moocore', 'shap', 'sklearn', 'scipy']
for pkg in submodule_packages:
    try:
        submods = collect_submodules(pkg)
        all_hiddenimports.extend(submods)
        print(f"[OK] Collected {len(submods)} submodules for {pkg}")
    except Exception as e:
        print(f"[WARN] Could not collect submodules for {pkg}: {e}")

# ============================================================================
# HIDDEN IMPORTS
# ============================================================================

explicit_hiddenimports = [
    # Core scientific libraries
    'numpy', 'numpy.core._methods', 'numpy.lib.format',
    'pandas',
    'scipy', 'scipy.signal', 'scipy.special', 'scipy.optimize', 'scipy.stats',
    'scipy.linalg', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.sparse.csgraph',
    'scipy.spatial', 'scipy.interpolate', 'scipy.fftpack', 'scipy.ndimage',
    'scipy.integrate',
    # sklearn modules
    'sklearn',
    'sklearn.utils._cython_blas', 'sklearn.utils._typedefs', 'sklearn.utils._heap',
    'sklearn.utils._sorting', 'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._quad_tree', 'sklearn.tree._utils',
    'sklearn.cross_decomposition', 'sklearn.preprocessing',
    'sklearn.model_selection', 'sklearn.metrics', 'sklearn.linear_model',
    'sklearn.ensemble', 'sklearn.svm', 'sklearn.neural_network',
    'sklearn.pipeline', 'sklearn.decomposition',
    # One-class specific (added 2026-04-17 audit)
    'sklearn.covariance', 'sklearn.neighbors',
    # ML boosting libraries
    'lightgbm', 'xgboost', 'catboost',
    # imbalanced-learn
    'imblearn', 'imblearn.pipeline', 'imblearn.over_sampling', 'imblearn.under_sampling',
    # Multi-objective optimization
    'pymoo', 'pymoo.core', 'pymoo.core.problem', 'pymoo.core.algorithm',
    'pymoo.core.population', 'pymoo.core.result', 'pymoo.algorithms',
    'pymoo.algorithms.moo', 'pymoo.algorithms.moo.nsga2', 'pymoo.optimize',
    'pymoo.util', 'pymoo.util.nds', 'pymoo.util.nds.non_dominated_sorting',
    'pymoo.operators', 'pymoo.operators.crossover', 'pymoo.operators.mutation',
    'pymoo.operators.sampling', 'pymoo.operators.selection', 'pymoo.termination',
    'moocore',
    # Model interpretability
    'shap', 'shap.explainers', 'shap.plots',
    # Utilities
    'joblib', 'platformdirs', 'threadpoolctl',
    # GUI and plotting
    'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageDraw', 'PIL.ImageOps', 'PIL.ImageFont',
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_pdf', 'matplotlib.backend_bases',
    # Standard library extensions
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
    # Excel support
    'openpyxl', 'xlsxwriter', 'xlrd',
    # Database
    'sqlalchemy', 'sqlalchemy.pool',
    # File-format readers
    'jcamp', 'spc_io', 'specdal', 'brukeropus', 'specio_py310', 'spectrochempy_omnic',
    'pybaselines', 'pybaselines.whittaker', 'pybaselines.polynomial',
    # GUI table widget
    'tksheet',
    # HTTP (used by some vendor format readers)
    'requests',
    # Optuna optimization framework
    'optuna', 'optuna.samplers', 'optuna.pruners',
]

all_hiddenimports.extend(explicit_hiddenimports)
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
    optimize=0,
    datas=[
        ('src/spectral_predict', 'src/spectral_predict'),
        ('asp_logo_final.png', '.'),
        ('asp_logo.ico', '.'),
        ('example/BoneCollagen.csv', 'example'),
    ] + all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'IPython', 'jupyter', 'notebook',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        # Exclude torch — no in-tree module imports it (T-38 deleted the
        # last importer, learned_preprocessing.py). Bundling torch would
        # add ~800MB for nothing.
        'torch', 'torchvision', 'torchaudio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectralPredict-py312',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    upx_exclude=[
        'scipy', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 'catboost',
        'numba', 'llvmlite',
    ],
    name='SpectralPredict-py312',
)
