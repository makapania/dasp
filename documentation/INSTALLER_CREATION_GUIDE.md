# Installer Creation Guide (Without Code Signing)
## Complete Step-by-Step Instructions for Windows, macOS, and Linux

**Date:** November 5, 2025
**Purpose:** Create installers for Spectral Predict GUI without code signing certificates
**Target Audience:** AI agents or developers building installers
**Cost:** $0 (no code signing)

**⚠️ IMPORTANT:** Without code signing, users will see security warnings:
- **Windows:** "Windows protected your PC" / "Unknown publisher"
- **macOS:** "Cannot be opened because the developer cannot be verified"
- **Both:** Users must explicitly bypass warnings to run

This is acceptable for:
- Internal/academic use
- Small user base who trust the source
- Beta testing
- Situations where $300-600/year for certificates isn't justified

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Windows Installer (PyInstaller + Inno Setup)](#windows-installer)
3. [macOS Installer (py2app + DMG)](#macos-installer)
4. [Linux Installer (AppImage)](#linux-installer)
5. [Distribution via GitHub Releases](#distribution-via-github-releases)
6. [User Installation Instructions](#user-installation-instructions)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

**All Platforms:**
- Python 3.8+ with pip
- Git (for version control)
- Your Spectral Predict codebase

**Platform-Specific:**
- **Windows:** Inno Setup 6+ ([download](http://www.jrsoftware.org/isinfo.php))
- **macOS:** Xcode Command Line Tools (`xcode-select --install`)
- **Linux:** appimagetool ([download](https://appimage.github.io/))

### Python Dependencies

**Install globally or in build environment:**

```bash
pip install pyinstaller==6.16.0  # Latest as of 2025
pip install py2app  # macOS only
```

### Project Structure

Ensure your project looks like this:

```
dasp/
├── spectral_predict_gui_optimized.py  # Main GUI script
├── src/
│   └── spectral_predict/              # Python package
│       ├── __init__.py
│       ├── search.py
│       ├── models.py
│       └── ...
├── julia_port/                        # Julia backend
│   └── SpectralPredict/
│       ├── Project.toml
│       └── src/
├── documentation/                     # User docs
├── README.md
└── requirements.txt                   # Python dependencies
```

---

## Windows Installer

### Overview

**Tools:** PyInstaller → Inno Setup → `.exe` installer

**Steps:**
1. Create PyInstaller spec file
2. Bundle Python + dependencies + Julia
3. Create Inno Setup script
4. Compile installer

**Expected Output:** `SpectralPredict_Setup_v1.0.0_Windows.exe` (~150-300 MB)

---

### Step 1: Prepare Build Environment

**Create dedicated build directory:**

```powershell
# In your project root
mkdir build_windows
cd build_windows

# Copy necessary files
cp ../spectral_predict_gui_optimized.py .
cp -r ../src .
cp -r ../documentation .
cp ../README.md .
cp ../requirements.txt .
```

**Install dependencies:**

```powershell
pip install -r requirements.txt
pip install pyinstaller
```

---

### Step 2: Create PyInstaller Spec File

**File:** `SpectralPredict.spec`

```python
# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for Spectral Predict GUI.

Bundles Python, dependencies, and Julia runtime into standalone executable.

Build with:
    pyinstaller SpectralPredict.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all submodules for sklearn (has many dynamic imports)
sklearn_hidden_imports = collect_submodules('sklearn')

# Collect all submodules for other key packages
hidden_imports = [
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.skiplist',
] + sklearn_hidden_imports

# Data files to include
datas = [
    ('src/spectral_predict', 'spectral_predict'),  # Python package
    ('documentation', 'documentation'),            # User docs
    ('README.md', '.'),                            # README
]

# Add Julia runtime if bundling (see Step 3)
# datas += [('julia_runtime', 'julia_runtime')]

a = Analysis(
    ['spectral_predict_gui_optimized.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',  # Don't include test frameworks
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    upx=True,                    # Compress with UPX (reduces size ~30%)
    console=False,               # No console window (GUI only)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,      # No code signing
    entitlements_file=None,
    icon='logo.ico'              # Add icon if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpectralPredict'
)
```

**Notes:**
- `console=False`: Hides console window (GUI only)
- `upx=True`: Compresses executables (smaller size)
- `exclude_binaries=True` + `COLLECT`: Creates folder with .exe + dependencies (better than `--onefile` which triggers Windows Defender)

---

### Step 3: Bundle Julia Runtime (Optional)

**Option A: Include Full Julia Runtime**

**Download Julia portable:**

```powershell
# Download Julia 1.11+ Windows binary (portable)
# From: https://julialang.org/downloads/

# Example for Julia 1.11.1:
Invoke-WebRequest -Uri "https://julialang-s3.julialang.org/bin/winnt/x64/1.11/julia-1.11.1-win64.zip" -OutFile "julia.zip"

# Extract
Expand-Archive julia.zip -DestinationPath julia_runtime

# Directory structure:
# julia_runtime/
#   ├── bin/
#   │   └── julia.exe
#   ├── lib/
#   └── share/
```

**Add to spec file:**

```python
# In SpectralPredict.spec, modify datas:
datas += [('julia_runtime', 'julia_runtime')]
```

**Update Python bridge to use bundled Julia:**

**File:** `src/spectral_predict/julia_bridge_config.py` (NEW)

```python
"""Configuration for Julia bridge - detects bundled Julia runtime."""

import sys
import os
from pathlib import Path

def get_julia_exe():
    """Get path to Julia executable (bundled or system)."""

    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        julia_exe = bundle_dir / "julia_runtime" / "bin" / "julia.exe"

        if julia_exe.exists():
            return str(julia_exe)

    # Fallback to system Julia
    return "julia"  # Assumes julia is in PATH

def get_julia_project():
    """Get path to Julia project."""

    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS)
        project_dir = bundle_dir / "julia_runtime" / "SpectralPredict"

        if project_dir.exists():
            return str(project_dir)

    # Fallback
    return None  # Will use system Julia project
```

**Modify `spectral_predict_julia_bridge.py`:**

```python
# At top of file
from .julia_bridge_config import get_julia_exe, get_julia_project

# Update default paths
JULIA_EXE = get_julia_exe()
JULIA_PROJECT = get_julia_project()
```

**⚠️ WARNING:** Including full Julia runtime adds ~400-500 MB to installer.

---

**Option B: Require User to Install Julia Separately**

**Pros:** Smaller installer (~150 MB)
**Cons:** Users must install Julia manually

**Skip bundling Julia.** Instead, in GUI startup:

```python
# In spectral_predict_gui_optimized.py
def check_julia_available():
    """Check if Julia is installed."""
    import subprocess

    try:
        result = subprocess.run(['julia', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except:
        return False

# At GUI startup
if not check_julia_available():
    messagebox.showwarning(
        "Julia Not Found",
        "Julia is not installed. Some features will be unavailable.\n\n"
        "Download from: https://julialang.org/downloads/"
    )
```

---

### Step 4: Build with PyInstaller

```powershell
# Build
pyinstaller SpectralPredict.spec

# Output location:
# dist/SpectralPredict/
#   ├── SpectralPredict.exe          # Main executable
#   ├── *.dll                         # Dependencies
#   └── [other files]
```

**Test the executable:**

```powershell
cd dist\SpectralPredict
.\SpectralPredict.exe
```

**Verify:**
- GUI launches
- Can load data
- Can run analysis (if Julia bundled)
- No errors in console (run from PowerShell to see errors)

---

### Step 5: Create Inno Setup Script

**File:** `installer_windows.iss`

```innosetup
; Inno Setup Script for Spectral Predict
; Creates Windows installer (.exe) from PyInstaller output

[Setup]
; Basic app info
AppName=Spectral Predict
AppVersion=1.0.0
AppPublisher=Your Name/Organization
AppPublisherURL=https://github.com/yourusername/spectralpredict
AppSupportURL=https://github.com/yourusername/spectralpredict/issues
AppUpdatesURL=https://github.com/yourusername/spectralpredict/releases

; Default installation directory
DefaultDirName={autopf}\SpectralPredict
DefaultGroupName=Spectral Predict

; Output settings
OutputDir=output
OutputBaseFilename=SpectralPredict_Setup_v1.0.0_Windows
SetupIconFile=logo.ico  ; Optional: add your icon

; Compression
Compression=lzma2
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Privileges
PrivilegesRequired=lowest  ; Don't require admin (installs to user directory)

; Uninstall
UninstallDisplayName=Spectral Predict
UninstallDisplayIcon={app}\SpectralPredict.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application files (from PyInstaller dist folder)
Source: "dist\SpectralPredict\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: isreadme

[Icons]
; Start menu shortcut
Name: "{group}\Spectral Predict"; Filename: "{app}\SpectralPredict.exe"

; Desktop shortcut (optional)
Name: "{autodesktop}\Spectral Predict"; Filename: "{app}\SpectralPredict.exe"; Tasks: desktopicon

; Uninstaller
Name: "{group}\Uninstall Spectral Predict"; Filename: "{uninstallexe}"

[Run]
; Launch after installation (optional)
Filename: "{app}\SpectralPredict.exe"; Description: "{cm:LaunchProgram,Spectral Predict}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any files created by the app
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\cache"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('This installer will install Spectral Predict on your computer.'#13#10#13#10 +
         'NOTE: Windows may show a security warning because this software is not code-signed. ' +
         'This is normal and safe to bypass.',
         mbInformation, MB_OK);
end;
```

---

### Step 6: Compile Installer

**Using Inno Setup GUI:**

1. Open Inno Setup Compiler
2. File → Open → Select `installer_windows.iss`
3. Build → Compile
4. Output: `output/SpectralPredict_Setup_v1.0.0_Windows.exe`

**Using command line:**

```powershell
# Find Inno Setup compiler
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

# Compile
& $iscc installer_windows.iss

# Output: output\SpectralPredict_Setup_v1.0.0_Windows.exe
```

---

### Step 7: Test Installer

**On a clean Windows 10/11 VM or PC:**

1. **Run installer:**
   ```powershell
   .\SpectralPredict_Setup_v1.0.0_Windows.exe
   ```

2. **Expect Windows SmartScreen warning:**
   ```
   Windows protected your PC
   Microsoft Defender SmartScreen prevented an unrecognized app from starting.
   Running this app might put your PC at risk.
   ```

3. **Bypass warning:**
   - Click "More info"
   - Click "Run anyway"

4. **Complete installation:**
   - Follow wizard
   - Choose install location
   - Finish

5. **Test application:**
   - Launch from Start Menu
   - Load sample data
   - Run analysis
   - Verify all features work

---

### Step 8: Optimize Installer Size

**If installer is too large (>300 MB), reduce size:**

**Option 1: UPX Compression (already enabled in spec)**
- Already done with `upx=True`
- Reduces size by ~30%

**Option 2: Exclude Unnecessary Libraries**

```python
# In SpectralPredict.spec, add to excludes:
excludes=[
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'tkinter.test',  # Test modules
    'test',
    'unittest',
    # Matplotlib backends you don't use
    'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_qt5',
    # If not using certain sklearn models
    'sklearn.ensemble.tests',
    'sklearn.datasets',  # Sample datasets
]
```

**Option 3: Don't Bundle Julia**

See "Option B" above - let users install Julia separately.

**Rebuild after changes:**

```powershell
pyinstaller --clean SpectralPredict.spec
```

---

## macOS Installer

### Overview

**Tools:** py2app → DMG

**Steps:**
1. Create setup.py for py2app
2. Bundle Python + dependencies + Julia
3. Create DMG with drag-and-drop installer
4. (Optional but recommended) Sign locally for Gatekeeper (self-signed certificate)

**Expected Output:** `SpectralPredict_v1.0.0_macOS.dmg` (~150-300 MB)

**⚠️ CRITICAL:** macOS Gatekeeper will block unsigned apps since macOS Catalina (2019). Users must right-click → Open to bypass.

---

### Step 1: Prepare Build Environment

**Requires macOS machine (or VM).**

```bash
# In your project root
mkdir build_macos
cd build_macos

# Copy necessary files
cp ../spectral_predict_gui_optimized.py .
cp -r ../src .
cp -r ../documentation .
cp ../README.md .
cp ../requirements.txt .
```

**Install dependencies:**

```bash
pip3 install -r requirements.txt
pip3 install py2app
```

---

### Step 2: Create setup.py for py2app

**File:** `setup.py`

```python
"""
py2app setup script for Spectral Predict GUI.

Build with:
    python3 setup.py py2app
"""

from setuptools import setup

APP = ['spectral_predict_gui_optimized.py']

DATA_FILES = [
    ('spectral_predict', ['src/spectral_predict']),
    ('documentation', ['documentation']),
    ('', ['README.md']),
]

# Add Julia runtime if bundling
# DATA_FILES += [('julia_runtime', ['julia_runtime'])]

OPTIONS = {
    'argv_emulation': True,  # Enable drag-and-drop file opening
    'packages': [
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'PIL',
        'tkinter',
    ],
    'includes': [
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.tree',
    ],
    'excludes': [
        'pytest',
        'IPython',
        'jupyter',
        'sphinx',
        'test',
        'unittest',
    ],
    'iconfile': 'logo.icns',  # Optional: .icns icon file
    'plist': {
        'CFBundleName': 'Spectral Predict',
        'CFBundleDisplayName': 'Spectral Predict',
        'CFBundleIdentifier': 'com.yourorg.spectralpredict',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleInfoDictionaryVersion': '6.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.14',  # macOS Mojave or later
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
    },
    'semi_standalone': False,  # Include Python framework
    'site_packages': True,
}

setup(
    app=APP,
    name='Spectral Predict',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
```

---

### Step 3: Bundle Julia Runtime (Optional)

**Option A: Include Full Julia Runtime**

```bash
# Download Julia macOS binary
# From: https://julialang.org/downloads/

# Example for Julia 1.11.1 (Intel):
curl -L https://julialang-s3.julialang.org/bin/mac/x64/1.11/julia-1.11.1-mac64.tar.gz -o julia.tar.gz

# OR for Apple Silicon (M1/M2):
curl -L https://julialang-s3.julialang.org/bin/mac/aarch64/1.11/julia-1.11.1-macaarch64.tar.gz -o julia.tar.gz

# Extract
tar -xzf julia.tar.gz
mv julia-1.11.1 julia_runtime

# Directory structure:
# julia_runtime/
#   ├── bin/
#   │   └── julia
#   ├── lib/
#   └── share/
```

**Add to setup.py:**

```python
DATA_FILES += [('julia_runtime', ['julia_runtime'])]
```

**Update Python bridge** (same as Windows - see Step 3 in Windows section)

**⚠️ WARNING:** Julia runtime adds ~400-500 MB.

---

**Option B: Require User to Install Julia**

Skip bundling. Recommend users install via Homebrew:

```bash
brew install julia
```

---

### Step 4: Build with py2app

```bash
# Clean previous builds
rm -rf build dist

# Build
python3 setup.py py2app

# Output location:
# dist/Spectral Predict.app
```

**Test the app:**

```bash
open "dist/Spectral Predict.app"
```

**Verify:**
- GUI launches
- Can load data
- Can run analysis
- Check console for errors: `Console.app` → filter for "Spectral Predict"

---

### Step 5: Handle Gatekeeper (Without Notarization)

**Problem:** macOS Gatekeeper blocks unsigned apps.

**User will see:**
```
"Spectral Predict.app" cannot be opened because the developer cannot be verified.
```

**Solutions:**

**Option 1: Users Right-Click → Open**

Instruct users:
1. Right-click (or Control-click) on `Spectral Predict.app`
2. Click "Open"
3. Click "Open" again in the confirmation dialog

This adds a one-time exception to Gatekeeper.

**Option 2: Self-Sign Locally (Ad Hoc Signature)**

Even without Apple Developer account, you can create a local signature:

```bash
# Sign with ad-hoc signature (no certificate)
codesign --force --deep --sign - "dist/Spectral Predict.app"

# Verify signature
codesign --verify --deep --verbose "dist/Spectral Predict.app"
```

**This does NOT bypass Gatekeeper**, but:
- Ensures app integrity (detects tampering)
- Slightly better than completely unsigned
- Still requires users to right-click → Open

---

### Step 6: Create DMG Installer

**Tool:** `create-dmg` (Homebrew)

**Install:**

```bash
brew install create-dmg
```

**Create DMG:**

```bash
create-dmg \
  --volname "Spectral Predict" \
  --volicon "logo.icns" \
  --window-pos 200 120 \
  --window-size 800 450 \
  --icon-size 100 \
  --icon "Spectral Predict.app" 200 190 \
  --hide-extension "Spectral Predict.app" \
  --app-drop-link 600 185 \
  --background "installer_background.png" \
  "SpectralPredict_v1.0.0_macOS.dmg" \
  "dist/"
```

**Explanation:**
- `--volname`: Volume name when mounted
- `--window-size`: DMG window dimensions
- `--icon`: Position of app icon
- `--app-drop-link`: Creates "Applications" folder shortcut at position (600, 185)
- `--background`: Custom background image (optional)

**Output:** `SpectralPredict_v1.0.0_macOS.dmg`

---

**Alternative: Simple DMG (No fancy UI)**

```bash
# Create simple DMG
hdiutil create -volname "Spectral Predict" \
               -srcfolder "dist/" \
               -ov \
               -format UDZO \
               "SpectralPredict_v1.0.0_macOS.dmg"
```

---

### Step 7: Test DMG

**On a clean macOS VM or Mac:**

1. **Mount DMG:**
   ```bash
   open SpectralPredict_v1.0.0_macOS.dmg
   ```

2. **Try to open app directly from DMG:**
   - Double-click `Spectral Predict.app`
   - **Expect Gatekeeper error:**
     ```
     "Spectral Predict.app" cannot be opened because the developer cannot be verified.
     ```

3. **Bypass Gatekeeper:**
   - Right-click → Open
   - Click "Open" in dialog

4. **Drag app to Applications:**
   - Drag `Spectral Predict.app` to Applications folder shortcut
   - Launch from Applications

5. **Test features:**
   - Load data
   - Run analysis
   - Verify Julia backend works (if bundled)

---

### Step 8: Build for Multiple Architectures

**If distributing to both Intel and Apple Silicon Macs:**

**Option 1: Universal Binary (Recommended)**

py2app doesn't natively support universal binaries. Instead:

1. Build on Intel Mac → `SpectralPredict_v1.0.0_macOS_Intel.dmg`
2. Build on M1/M2 Mac → `SpectralPredict_v1.0.0_macOS_ARM64.dmg`
3. Distribute both

**Option 2: Specify Architecture**

```bash
# On Intel Mac, build for Intel
arch -x86_64 python3 setup.py py2app

# On M1/M2 Mac, build for ARM64
arch -arm64 python3 setup.py py2app
```

---

### Step 9: Optimize DMG Size

**Same strategies as Windows:**

1. **Exclude unnecessary packages:**

```python
# In setup.py OPTIONS
'excludes': [
    'pytest', 'IPython', 'jupyter', 'sphinx', 'test',
    'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_qt5',
    'sklearn.datasets',
],
```

2. **Compress DMG:**

```bash
# Already done with `-format UDZO` (zlib compression)
# For maximum compression:
hdiutil create -format UDBZ ...  # bzip2 (slower but smaller)
```

3. **Don't bundle Julia:**

Let users install Julia separately.

---

## Linux Installer

### Overview

**Tool:** AppImage

**Steps:**
1. Build with PyInstaller
2. Create AppDir structure
3. Bundle Julia (optional)
4. Build AppImage

**Expected Output:** `SpectralPredict-x86_64.AppImage` (~150-300 MB)

**Benefits of AppImage:**
- Single file, no installation required
- Works on all major distros (Ubuntu, Fedora, Arch, etc.)
- Sandboxed execution
- Portable (can run from USB drive)

---

### Step 1: Prepare Build Environment

**Requires Linux machine (or VM).**

```bash
# In your project root
mkdir build_linux
cd build_linux

# Copy necessary files
cp ../spectral_predict_gui_optimized.py .
cp -r ../src .
cp -r ../documentation .
cp ../README.md .
cp ../requirements.txt .
```

**Install dependencies:**

```bash
pip3 install -r requirements.txt
pip3 install pyinstaller
```

---

### Step 2: Build with PyInstaller

**Create spec file (similar to Windows):**

**File:** `SpectralPredict_Linux.spec`

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

hidden_imports = [
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.tree',
    'pandas._libs.tslibs.timedeltas',
]

datas = [
    ('src/spectral_predict', 'spectral_predict'),
    ('documentation', 'documentation'),
    ('README.md', '.'),
]

a = Analysis(
    ['spectral_predict_gui_optimized.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['pytest', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    console=False,  # No terminal window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='SpectralPredict'
)
```

**Build:**

```bash
pyinstaller SpectralPredict_Linux.spec

# Output: dist/SpectralPredict/
```

---

### Step 3: Create AppDir Structure

**AppDir is the standard directory structure for AppImages:**

```bash
# Create AppDir
mkdir -p SpectralPredict.AppDir/usr/{bin,lib,share}

# Copy PyInstaller output
cp -r dist/SpectralPredict/* SpectralPredict.AppDir/usr/bin/

# Create launcher script
cat > SpectralPredict.AppDir/AppRun << 'EOF'
#!/bin/bash
# AppRun launcher script

# Get directory
HERE="$(dirname "$(readlink -f "${0}")")"

# Set library path
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"

# Launch application
exec "${HERE}/usr/bin/SpectralPredict" "$@"
EOF

chmod +x SpectralPredict.AppDir/AppRun
```

**Create .desktop file:**

```bash
cat > SpectralPredict.AppDir/SpectralPredict.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Spectral Predict
Comment=Spectral analysis and chemometrics
Exec=SpectralPredict
Icon=spectralpredict
Categories=Science;Education;DataVisualization;
Terminal=false
EOF
```

**Add icon (if you have one):**

```bash
# Copy icon to multiple sizes
cp logo.png SpectralPredict.AppDir/spectralpredict.png
cp logo.png SpectralPredict.AppDir/.DirIcon
```

---

### Step 4: Bundle Julia (Optional)

**Download Julia Linux binary:**

```bash
# Julia 1.11.1 for Linux x86_64
curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.1-linux-x86_64.tar.gz -o julia.tar.gz

# Extract
tar -xzf julia.tar.gz
mv julia-1.11.1 SpectralPredict.AppDir/usr/lib/julia_runtime

# Update launcher to include Julia
cat > SpectralPredict.AppDir/AppRun << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"

export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
export PATH="${HERE}/usr/lib/julia_runtime/bin:${PATH}"

exec "${HERE}/usr/bin/SpectralPredict" "$@"
EOF

chmod +x SpectralPredict.AppDir/AppRun
```

---

### Step 5: Build AppImage

**Download appimagetool:**

```bash
# Get appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage
```

**Build AppImage:**

```bash
./appimagetool-x86_64.AppImage SpectralPredict.AppDir SpectralPredict-x86_64.AppImage

# Output: SpectralPredict-x86_64.AppImage
```

**Make executable:**

```bash
chmod +x SpectralPredict-x86_64.AppImage
```

---

### Step 6: Test AppImage

```bash
# Run AppImage
./SpectralPredict-x86_64.AppImage

# Test features:
# - Load data
# - Run analysis
# - Verify Julia backend (if bundled)
```

**Test on different distros:**
- Ubuntu 22.04+
- Fedora 38+
- Arch Linux

---

### Step 7: Optimize Size

**Same strategies:**

1. Exclude unnecessary packages
2. Don't bundle Julia (let users install separately)
3. Use UPX compression

```bash
# Compress libraries
upx --best --lzma SpectralPredict.AppDir/usr/bin/*
```

---

## Distribution via GitHub Releases

### Step 1: Create Release on GitHub

```bash
# Tag version
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

**On GitHub:**

1. Go to repository → Releases
2. Click "Draft a new release"
3. Choose tag: `v1.0.0`
4. Release title: "Spectral Predict v1.0.0"

---

### Step 2: Upload Installers

**Drag and drop files:**

- `SpectralPredict_Setup_v1.0.0_Windows.exe` (~150-300 MB)
- `SpectralPredict_v1.0.0_macOS_Intel.dmg` (~150-300 MB)
- `SpectralPredict_v1.0.0_macOS_ARM64.dmg` (~150-300 MB)
- `SpectralPredict-x86_64.AppImage` (~150-300 MB)

**Also include:**
- `SHA256SUMS.txt` - File checksums for verification
- `INSTALL.md` - Installation instructions
- Source code (automatically added by GitHub)

---

### Step 3: Write Release Notes

**Example:**

```markdown
# Spectral Predict v1.0.0

First stable release of Spectral Predict GUI!

## Features

- 🔬 Multiple preprocessing methods (SNV, MSC, Savitzky-Golay derivatives)
- 🤖 6 regression models (PLS, Ridge, Lasso, Random Forest, MLP, Neural Boosted)
- 📊 Variable selection (Importance, SPA, UVE, iPLS, UVE-SPA)
- 📈 Model diagnostics (residuals, leverage, prediction intervals)
- 💾 Model save/load (.dasp format)
- 📁 CSV export for external validation

## Downloads

**Windows:**
- [SpectralPredict_Setup_v1.0.0_Windows.exe](link)
  - **Note:** Windows will show "Unknown publisher" warning. Click "More info" → "Run anyway".

**macOS:**
- [Intel Macs (x64)](link)
- [Apple Silicon (M1/M2)](link)
  - **Note:** Right-click → Open to bypass Gatekeeper.

**Linux:**
- [SpectralPredict-x86_64.AppImage](link)
  - Make executable: `chmod +x SpectralPredict-x86_64.AppImage`

## Installation

See [INSTALL.md](link) for detailed instructions.

## System Requirements

- **Windows:** 10 or 11 (64-bit)
- **macOS:** 10.14 (Mojave) or later
- **Linux:** Recent distro with glibc 2.27+
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk:** 500 MB free space

## Known Issues

- Windows Defender may flag installer (false positive, no code signing)
- macOS Gatekeeper blocks app (requires right-click → Open)
- Julia not bundled: install separately from https://julialang.org

## Documentation

- [User Guide](link)
- [Troubleshooting](link)
- [GitHub Issues](link)
```

---

### Step 4: Create SHA256SUMS.txt

```bash
# Windows
sha256sum SpectralPredict_Setup_v1.0.0_Windows.exe > SHA256SUMS.txt

# macOS
shasum -a 256 SpectralPredict_v1.0.0_macOS_Intel.dmg >> SHA256SUMS.txt
shasum -a 256 SpectralPredict_v1.0.0_macOS_ARM64.dmg >> SHA256SUMS.txt

# Linux
sha256sum SpectralPredict-x86_64.AppImage >> SHA256SUMS.txt
```

**Upload `SHA256SUMS.txt` to release.**

---

## User Installation Instructions

### Create INSTALL.md

**File:** `INSTALL.md`

```markdown
# Installation Instructions

## Windows

### Step 1: Download Installer

Download `SpectralPredict_Setup_v1.0.0_Windows.exe` from [Releases](link).

### Step 2: Bypass Windows SmartScreen

1. Run the installer
2. Windows will show: "Windows protected your PC"
3. Click **"More info"**
4. Click **"Run anyway"**

**Why?** We don't have a code signing certificate ($300/year). The software is safe but unsigned.

### Step 3: Install

1. Choose installation location (default: `C:\Program Files\SpectralPredict`)
2. Click "Install"
3. Launch from Start Menu

### Step 4: (Optional) Install Julia

If you want to use Julia-accelerated features:

1. Download Julia 1.11+ from https://julialang.org/downloads/
2. Install (add to PATH when prompted)
3. Restart Spectral Predict

---

## macOS

### Step 1: Download DMG

Download the appropriate DMG:
- **Intel Macs:** `SpectralPredict_v1.0.0_macOS_Intel.dmg`
- **Apple Silicon (M1/M2):** `SpectralPredict_v1.0.0_macOS_ARM64.dmg`

### Step 2: Mount DMG

Double-click the DMG file.

### Step 3: Bypass Gatekeeper

1. **Do NOT double-click the app yet**
2. **Right-click** (or Control-click) on `Spectral Predict.app`
3. Click **"Open"**
4. Click **"Open"** again in the confirmation dialog

**Why?** We don't have an Apple Developer certificate ($99/year) and notarization. The software is safe but unsigned.

### Step 4: Drag to Applications

Drag `Spectral Predict.app` to the Applications folder shortcut.

### Step 5: (Optional) Install Julia

```bash
# Using Homebrew
brew install julia

# Or download from https://julialang.org/downloads/
```

---

## Linux

### Step 1: Download AppImage

Download `SpectralPredict-x86_64.AppImage` from [Releases](link).

### Step 2: Make Executable

```bash
chmod +x SpectralPredict-x86_64.AppImage
```

### Step 3: Run

```bash
./SpectralPredict-x86_64.AppImage
```

**Optional:** Integrate with desktop environment:

```bash
# Install AppImageLauncher (Ubuntu/Debian)
sudo apt install appimagelauncher

# Or manually create .desktop file
```

### Step 4: (Optional) Install Julia

```bash
# Ubuntu/Debian
sudo apt install julia

# Fedora
sudo dnf install julia

# Or download from https://julialang.org/downloads/
```

---

## Troubleshooting

### Windows: "Cannot find vcruntime140.dll"

**Solution:** Install Visual C++ Redistributable
https://aka.ms/vs/17/release/vc_redist.x64.exe

### macOS: "Damaged and can't be opened"

**Solution:** Remove quarantine attribute:

```bash
xattr -cr "/Applications/Spectral Predict.app"
```

### Linux: "Permission denied"

**Solution:** Make AppImage executable:

```bash
chmod +x SpectralPredict-x86_64.AppImage
```

### Julia Not Found

**Verify Julia installation:**

```bash
julia --version
```

**If not found:** Install Julia from https://julialang.org/downloads/

---

## Getting Help

- **Documentation:** [User Guide](link)
- **Issues:** [GitHub Issues](link)
- **Community:** [Discussions](link)
```

---

## Troubleshooting

### Common Build Issues

#### 1. PyInstaller: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'sklearn.utils._cython_blas'`

**Solution:** Add to `hiddenimports` in spec file:

```python
hiddenimports=[
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.tree._utils',
    # Add any other missing modules
]
```

#### 2. py2app: ImportError

**Error:** `ImportError: cannot import name 'X' from 'Y'`

**Solution:** Add to `packages` or `includes` in OPTIONS:

```python
'packages': ['numpy', 'pandas', 'sklearn', 'scipy'],
'includes': ['specific.module.that.failed'],
```

#### 3. Large File Size

**Solutions:**

1. **Exclude unnecessary packages:**
   ```python
   'excludes': ['pytest', 'IPython', 'jupyter', 'matplotlib.tests']
   ```

2. **Don't bundle Julia** (biggest size reduction)

3. **Use UPX compression** (already enabled)

#### 4. Windows: "VCRUNTIME140.dll missing"

**User needs:** Visual C++ Redistributable

**Add to installer:** Include VC++ redist in Inno Setup:

```innosetup
[Files]
Source: "vcredist_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\vcredist_x64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing Visual C++ Runtime..."
```

#### 5. macOS: App Crashes on Launch

**Check Console.app for errors:**

1. Open Console.app
2. Filter for "Spectral Predict"
3. Look for error messages

**Common fixes:**
- Missing dependencies
- Incorrect plist settings
- Python version mismatch

#### 6. Linux: "cannot execute binary file"

**Cause:** Built on wrong architecture

**Solution:** Build on same architecture as target (x86_64)

---

### Testing Checklist

**For each platform, verify:**

- [ ] Installer runs without errors
- [ ] App launches successfully
- [ ] GUI displays correctly (no missing fonts/icons)
- [ ] Can load sample data (CSV)
- [ ] Can run preprocessing
- [ ] Can train models
- [ ] Can save/load models
- [ ] Can export results
- [ ] Julia backend works (if bundled)
- [ ] Help documentation accessible
- [ ] App uninstalls cleanly

---

## Summary

### What You Built

**3 installers without code signing:**

1. **Windows:** `.exe` installer (150-300 MB)
   - Users see "Unknown publisher" → "More info" → "Run anyway"

2. **macOS:** `.dmg` installer (150-300 MB)
   - Users right-click → Open to bypass Gatekeeper

3. **Linux:** `.AppImage` portable app (150-300 MB)
   - No installation required, just `chmod +x` and run

### Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| PyInstaller | Free | Open source |
| Inno Setup | Free | Open source |
| py2app | Free | Open source |
| AppImage tools | Free | Open source |
| GitHub Releases | Free | Unlimited for public repos |
| **Total** | **$0** | No ongoing costs |

### Trade-offs Without Code Signing

**Pros:**
- $0 cost (vs $300-600/year with signing)
- Simpler build process
- No renewal/maintenance

**Cons:**
- Users see security warnings
- More support questions about warnings
- Less professional appearance
- Some corporate IT departments block unsigned software

### When to Upgrade to Code Signing

**Consider code signing if:**
- Distributing to 100+ users
- Targeting corporate/enterprise users
- Users are non-technical
- Want automatic updates
- Need professional credibility

**Cost:** ~$300-600 first year, ~$300-600/year ongoing

---

## Next Steps

1. **Test installers** on clean VMs/machines
2. **Create release** on GitHub
3. **Write user documentation** (INSTALL.md, README.md)
4. **Announce release** (website, mailing list, social media)
5. **Gather feedback** from early users
6. **Iterate** based on user reports

---

**End of Installer Creation Guide**

**Total Time to Create Installers:** 1-2 days (first time), 2-4 hours (subsequent releases)
**Total Cost:** $0
**Output:** 3 platform installers ready for distribution

Good luck with your installer creation! 🚀
