# Vendor-Specific Spectroscopy File Format Support

This document describes the implementation of vendor-specific file format readers for Bruker, PerkinElmer, and Agilent instruments.

## Overview

Three new reader modules have been added to support additional spectroscopy file formats:

1. **Bruker OPUS** (.0, .1, .2, ..., .999) - FTIR binary format
2. **PerkinElmer** (.sp) - IR binary format
3. **Agilent** (.seq, .dmt, .asp, .bsw) - IR and hyperspectral imaging formats

## Files Created

### Reader Modules (src/spectral_predict/readers/)

1. **opus_reader.py** - Bruker OPUS format reader
   - `read_opus_file()` - Read single OPUS file
   - `read_opus_dir()` - Read all OPUS files from directory
   - `convert_wavenumber_to_wavelength()` - Unit conversion helper
   - `convert_wavelength_to_wavenumber()` - Unit conversion helper

2. **perkinelmer_reader.py** - PerkinElmer format reader
   - `read_sp_file()` - Read single .sp file
   - `read_sp_dir()` - Read all .sp files from directory

3. **agilent_reader.py** - Agilent format reader
   - `read_agilent_file()` - Read single Agilent file (any format)
   - `read_agilent_dir()` - Read all Agilent files from directory
   - `read_seq_file()` - Convenience wrapper for .seq files
   - `read_dmt_file()` - Convenience wrapper for .dmt files
   - `read_asp_file()` - Convenience wrapper for .asp files

### Updated Files

1. **src/spectral_predict/io.py**
   - Updated `detect_format()` to recognize new file extensions
   - Added wrapper functions: `read_opus_file()`, `read_perkinelmer_file()`, `read_agilent_file()`
   - Added directory wrappers: `read_opus_dir()`, `read_sp_dir()`, `read_agilent_dir()`
   - Integrated with existing `read_spectra()` unified reader

2. **src/spectral_predict/readers/__init__.py**
   - Added exports for all new reader functions
   - Added comprehensive module documentation

## Installation

### Basic Installation

The base package includes core functionality:

```bash
pip install spectral-predict
```

### Optional Vendor Format Support

Install support for specific vendors:

```bash
# Bruker OPUS support
pip install spectral-predict[opus]

# PerkinElmer support
pip install spectral-predict[perkinelmer]

# Agilent support
pip install spectral-predict[agilent]

# All vendor formats
pip install spectral-predict[all-formats]
```

### Individual Library Installation

Alternatively, install the underlying libraries directly:

```bash
pip install brukeropus        # Bruker OPUS
pip install specio            # PerkinElmer
pip install agilent-ir-formats  # Agilent
```

## Usage Examples

### Bruker OPUS Files

```python
from spectral_predict.io import read_opus_file, read_opus_dir

# Read single OPUS file
spectrum, metadata = read_opus_file('sample.0')
print(f"Wavenumber range: {metadata['wavenumber_range']}")
print(f"Data type: {metadata['data_type']}")  # e.g., 'absorbance'

# Read all OPUS files from directory
df, metadata = read_opus_dir('data/bruker_samples/')
print(f"Loaded {len(df)} OPUS spectra")
print(f"Wavenumber range: {metadata['wavenumber_range'][0]:.1f} - {metadata['wavenumber_range'][1]:.1f} cm⁻¹")
```

### PerkinElmer .sp Files

```python
from spectral_predict.io import read_sp_dir

# Read all .sp files from directory
df, metadata = read_sp_dir('data/perkinelmer_samples/')
print(f"Loaded {len(df)} PerkinElmer spectra")
print(f"X-axis unit: {metadata['x_unit']}")  # 'wavenumber_cm-1' or 'wavelength_nm'
print(f"Range: {metadata['x_range']}")
```

### Agilent Files

```python
from spectral_predict.io import read_agilent_dir

# Read all Agilent files (any format)
df, metadata = read_agilent_dir('data/agilent_samples/')
print(f"Loaded {len(df)} Agilent spectra")
print(f"File formats: {metadata['file_formats']}")

# Read only .seq files with mean extraction from hyperspectral images
df, metadata = read_agilent_dir(
    'data/agilent_imaging/',
    extensions=['seq'],
    extract_mode='mean'  # Options: 'total', 'first', 'mean'
)
print(f"Hyperspectral images: {metadata['n_hyperspectral']}")
```

### Using the Unified Reader

All formats are automatically detected:

```python
from spectral_predict.io import read_spectra

# Auto-detect format from file extension
df, metadata = read_spectra('sample.0')  # Bruker OPUS
df, metadata = read_spectra('sample.sp')  # PerkinElmer
df, metadata = read_spectra('sample.seq')  # Agilent

# Or specify format explicitly
df, metadata = read_spectra('sample.0', format='opus')
```

## File Format Details

### Bruker OPUS (.0, .1, .2, etc.)

- **Extensions**: Numbered extensions (.0, .1, .2, ..., .999)
- **Data**: Infrared spectral data with extensive metadata
- **Library**: [brukeropus](https://github.com/joshduran/brukeropus)
- **X-axis**: Wavenumbers (cm⁻¹) - kept in native format
- **Data types**: Absorbance, transmittance, sample, reference
- **Priority**: Absorbance > Transmittance > Sample > Reference

**Notes**:
- OPUS files typically have multiple data types stored; the reader prioritizes absorbance
- Wavenumbers are kept in cm⁻¹ (not converted to nm) as this is the native format
- Use conversion helpers: `convert_wavenumber_to_wavelength()` and `convert_wavelength_to_wavenumber()`

### PerkinElmer (.sp)

- **Extensions**: .sp (case-insensitive: .SP also supported)
- **Data**: IR spectroscopic data in binary format
- **Library**: [specio](https://specio.readthedocs.io/)
- **X-axis**: Auto-detected (wavenumbers in cm⁻¹ or wavelengths in nm)
- **Detection**: Based on value ranges (400-4000 → wavenumbers, >1000 → wavelengths)

**Notes**:
- X-axis units are auto-detected and stored in metadata['x_unit']
- Multi-spectrum files are supported (first spectrum used by default)

### Agilent (.seq, .dmt, .asp, .bsw)

- **Extensions**:
  - .seq - Single-tile hyperspectral images
  - .dmt - Multi-tile mosaic hyperspectral images
  - .asp - Agilent IR spectrum files
  - .bsw - Agilent batch files
- **Data**: IR spectroscopy and hyperspectral imaging
- **Library**: [agilent-ir-formats](https://github.com/AlexHenderson/agilent-ir-formats)
- **X-axis**: Wavenumbers (cm⁻¹)

**Hyperspectral Imaging Support**:

For .seq and .dmt files containing hyperspectral images (3D: height × width × spectral_points),
the reader provides three extraction modes:

- `'total'` (default): Sum all pixel spectra
- `'mean'`: Average all pixel spectra
- `'first'`: Extract first pixel only (top-left corner)

```python
# Example: Extract mean spectrum from hyperspectral image
spectrum, metadata = read_agilent_file('image.seq', extract_mode='mean')
print(f"Image dimensions: {metadata['image_shape']}")  # e.g., (128, 256)
```

## Return Format

All readers return data in a consistent format:

```python
(df, metadata)
```

### DataFrame (df)
- **Format**: Wide matrix
- **Rows**: Sample IDs (filenames without extension)
- **Columns**: Wavelengths/wavenumbers (float values)
- **Values**: Spectral intensities
- **Sorted**: Columns are sorted in ascending order

### Metadata (dict)
Common fields:
- `'n_spectra'`: Number of spectra loaded
- `'wavelength_range'` or `'wavenumber_range'`: (min, max) tuple
- `'file_format'`: Format identifier ('opus', 'sp', 'agilent', etc.)
- `'data_type'`: Auto-detected type ('reflectance' or 'absorbance')
- `'type_confidence'`: Confidence score (0-100%)
- `'detection_method'`: How data type was detected

Format-specific fields:
- **OPUS**: `'available_data_types'`, `'dominant_data_type'`, `'sample_name'`, `'sample_form'`
- **PerkinElmer**: `'x_unit'` (wavenumber vs wavelength), `'x_unit_counts'`
- **Agilent**: `'file_formats'`, `'extract_mode'`, `'n_hyperspectral'`, `'image_shape'`

## Error Handling

All readers gracefully handle missing optional dependencies:

```python
from spectral_predict.io import read_opus_dir

try:
    df, metadata = read_opus_dir('data/')
except ImportError as e:
    print(e)
    # Output: "Bruker OPUS file support requires the 'brukeropus' library.
    #          Install with: pip install brukeropus
    #          Or install all vendor formats: pip install spectral-predict[all-formats]"
```

### Common Issues

1. **Missing Libraries**: Install the required optional dependency
2. **No Files Found**: Check directory path and file extensions
3. **Binary vs ASCII**: Some formats (e.g., ASD) may require additional libraries for binary files
4. **File Corruption**: Files that fail to read are skipped with warnings

## Implementation Notes

### Design Principles

1. **Modular Architecture**: Each vendor format has its own reader module
2. **Consistent API**: All readers follow the same `(df, metadata)` return pattern
3. **Optional Dependencies**: Vendor libraries are imported only when needed
4. **Error Messages**: Clear, actionable error messages with installation instructions
5. **Graceful Degradation**: Failed files are skipped with warnings rather than crashing
6. **Type Hints**: Full type annotations for better IDE support

### Testing Recommendations

To test the implementations without vendor-specific data files:

```python
# Test import and error handling
from spectral_predict.readers import opus_reader

try:
    df, meta = opus_reader.read_opus_file('nonexistent.0')
except ImportError as e:
    print("Import error handled correctly")
except ValueError as e:
    print("File error handled correctly")
```

### Extension Points

To add support for additional formats:

1. Create new reader in `src/spectral_predict/readers/[vendor]_reader.py`
2. Implement `read_[format]_file()` and `read_[format]_dir()` functions
3. Add wrapper in `src/spectral_predict/io.py`
4. Update `detect_format()` with new extensions
5. Export from `src/spectral_predict/readers/__init__.py`
6. Add optional dependency to `pyproject.toml`

## Library Compatibility

### Tested Versions
- **brukeropus**: Designed for latest version (check PyPI)
- **specio**: Compatible with 0.1.0+
- **agilent-ir-formats**: Compatible with 0.4.0+ (requires Python ≥3.10)

### Python Version Requirements
- **Minimum**: Python 3.10 (due to agilent-ir-formats requirement)
- **Recommended**: Python 3.11+

## References

### Documentation Links

- [brukeropus API docs](https://joshduran.github.io/brukeropus/)
- [specio documentation](https://specio.readthedocs.io/)
- [agilent-ir-formats GitHub](https://github.com/AlexHenderson/agilent-ir-formats)

### Related Formats

Already supported in spectral-predict:
- ASD (.asd, .sig) - via `read_asd_dir()`
- SPC (.spc) - via `read_spc_dir()`
- JCAMP-DX (.jdx, .dx) - via `read_jcamp_file()`
- CSV, Excel, ASCII text formats

## Future Enhancements

Potential improvements:
1. Support for writing OPUS/PerkinElmer/Agilent formats (currently read-only)
2. Advanced metadata extraction (instrument parameters, acquisition settings)
3. Multi-spectrum handling for PerkinElmer files
4. Batch processing utilities
5. Validation and quality control checks
6. Unit conversion utilities for different spectral ranges

## License

Implementation follows the MIT license of the spectral-predict package.
Individual vendor libraries have their own licenses:
- brukeropus: MIT
- specio: BSD-3-Clause
- agilent-ir-formats: MIT

---

**Implementation Date**: 2025-11-11
**Author**: Claude (Anthropic)
**Version**: 1.0.0
