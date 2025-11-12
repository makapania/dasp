# Unified I/O Architecture for Spectral Predict

## Overview

This document describes the unified I/O architecture implemented for Spectral Predict, providing comprehensive support for 10+ spectral file formats with automatic detection and a consistent API.

## Architecture Summary

### Core Components

1. **Unified Dispatcher Functions**
   - `read_spectra(path, format='auto', **kwargs)` - Universal reader with auto-detection
   - `write_spectra(data, path, format, metadata=None, **kwargs)` - Universal writer
   - `detect_format(path)` - Automatic format detection from extensions and magic bytes

2. **Format-Specific Readers**
   - `read_csv_spectra()` - CSV files (wide or long format)
   - `read_excel_spectra()` - Excel files (.xlsx, .xls)
   - `read_asd_dir()` - ASD files (ASCII or binary)
   - `read_spc_dir()` / `read_spc_file()` - SPC files
   - `read_jcamp_file()` - JCAMP-DX files
   - `read_ascii_spectra()` - Generic ASCII text files
   - `read_opus_file()` - Bruker OPUS files
   - `read_perkinelmer_file()` - PerkinElmer files
   - `read_agilent_file()` - Agilent files (in development)

3. **Format-Specific Writers**
   - `write_csv_spectra()` - CSV export
   - `write_excel_spectra()` - Excel export with formatting
   - `write_spc_file()` - SPC export (single spectrum)
   - `write_jcamp_file()` - JCAMP-DX export
   - `write_ascii_spectra()` - Generic ASCII export

## Supported Formats

| Format | Extensions | Read | Write | Auto-Detect | Dependencies |
|--------|-----------|------|-------|-------------|--------------|
| CSV | .csv | ✅ | ✅ | ✅ | Built-in |
| Excel | .xlsx, .xls | ✅ | ✅ | ✅ | openpyxl, xlsxwriter |
| ASD (ASCII) | .asd, .sig | ✅ | ❌ | ✅ | Built-in |
| ASD (Binary) | .asd | ✅ | ❌ | ✅ | specdal (optional) |
| SPC | .spc | ✅ | ✅ | ✅ | spc-io |
| JCAMP-DX | .jdx, .dx, .jcm | ✅ | ✅ | ✅ | jcamp |
| ASCII Text | .txt, .dat | ✅ | ✅ | ✅ | Built-in |
| Bruker OPUS | .0, .1, .2, etc. | ✅ | ❌ | ✅ | brukeropus (optional) |
| PerkinElmer | .sp | ✅ | ❌ | ✅ | specio (optional) |
| Agilent | .seq | 🚧 | ❌ | ✅ | agilent-ir-formats (optional) |

## Format Detection

The `detect_format()` function uses multiple strategies:

1. **Extension-based detection**: Maps common extensions to formats
2. **Bruker OPUS special handling**: Recognizes numbered extensions (.0, .1, etc.)
3. **Magic byte detection**: Reads file headers for ambiguous cases
   - SPC: Checks for 'MK' magic bytes
   - JCAMP: Looks for '##TITLE' or '##JCAMP' headers
   - OPUS: Detects 'OPUS' signature
4. **Directory detection**: Analyzes directory contents to infer format

## Metadata Structure

All readers return a consistent metadata dictionary with:

### Required Fields
- `file_format`: str - Format identifier ('csv', 'excel', 'asd', etc.)
- `n_spectra`: int - Number of spectra loaded
- `wavelength_range`: tuple - (min_wavelength, max_wavelength) in nm
- `data_type`: str - 'reflectance' or 'absorbance'
- `type_confidence`: float - Confidence score (0-100)
- `detection_method`: str - Method used for data type detection

### Format-Specific Fields
- Excel: `sheet_name`
- JCAMP: `jcamp_header` (dict of file metadata)
- SPC: Additional SPC metadata fields

## Usage Examples

### Basic Reading (Auto-Detection)

```python
from spectral_predict.io import read_spectra

# Auto-detect and read any format
df, metadata = read_spectra('data/spectra.csv')
df, metadata = read_spectra('data/spectra.xlsx')
df, metadata = read_spectra('data/asd_files/')
df, metadata = read_spectra('data/spectrum.jdx')

print(f"Loaded {metadata['n_spectra']} spectra")
print(f"Format: {metadata['file_format']}")
print(f"Data type: {metadata['data_type']}")
```

### Explicit Format Specification

```python
# Specify format explicitly
df, metadata = read_spectra('data/file.dat', format='ascii')
df, metadata = read_spectra('data/spectra', format='csv')
```

### Writing Data

```python
from spectral_predict.io import write_spectra
import pandas as pd
import numpy as np

# Create sample data
wavelengths = np.linspace(400, 2400, 2001)
data = pd.DataFrame(
    np.random.rand(3, 2001),
    index=['Sample_1', 'Sample_2', 'Sample_3'],
    columns=wavelengths
)

# Write to various formats
write_spectra(data, 'output.csv', format='csv')
write_spectra(data, 'output.xlsx', format='excel', sheet_name='Spectra')
write_spectra(data.iloc[[0]], 'output.jdx', format='jcamp', title='Sample 1')
```

### Format-Specific Options

```python
# Excel with custom formatting
write_spectra(data, 'output.xlsx', format='excel',
              sheet_name='VIS-NIR',
              freeze_panes=(1, 1),
              float_format='0.0000')

# CSV with custom float format
write_spectra(data, 'output.csv', format='csv',
              float_format='%.4f')

# JCAMP with metadata
write_spectra(data.iloc[[0]], 'output.jdx', format='jcamp',
              title='Bone Sample A',
              data_type='INFRARED SPECTRUM',
              xunits='NANOMETERS',
              yunits='REFLECTANCE')

# ASCII with delimiter
write_spectra(data.iloc[[0]], 'output.txt', format='ascii',
              delimiter='\t',
              include_header=True)
```

## Testing Suite

Comprehensive test coverage across 4 test files:

### test_io_excel.py (15 tests)
- Wide and long format reading
- Sheet selection
- Roundtrip testing
- Custom formatting
- Metadata extraction

### test_io_jcamp.py (11 tests)
- Basic JCAMP reading/writing
- Metadata handling
- Custom units
- Roundtrip verification
- Import error handling

### test_io_spc.py (15 tests)
- SPC file and directory reading
- Single file handling
- Write functionality
- Error handling
- Import validation

### test_io_vendor_formats.py (32 tests)
- ASCII text format (multiple delimiters)
- Bruker OPUS detection
- PerkinElmer detection
- Agilent detection
- Format detection (extensions and magic bytes)
- Roundtrip testing

### test_io_unified.py (32 tests)
- Auto-detection across formats
- Directory format detection
- Unified write/read APIs
- Format-specific parameter passing
- Metadata consistency
- Roundtrip testing
- Error handling

**Total: 105 tests**

## Design Principles

1. **Consistent Interface**: All readers return (DataFrame, metadata) tuples
2. **Auto-Detection**: Format detection works seamlessly across all supported formats
3. **Graceful Degradation**: Optional dependencies raise helpful ImportError messages
4. **Flexible Inputs**: Support both single files and directories where appropriate
5. **Rich Metadata**: Comprehensive metadata returned from all readers
6. **Validation**: All readers validate wavelength counts and ordering
7. **Data Type Detection**: Automatic detection of reflectance vs. absorbance
8. **Format Preservation**: Writers maintain data precision and structure

## Data Validation

All readers perform these validations:

1. **Wavelength Count**: Minimum 100 wavelengths required
2. **Wavelength Ordering**: Must be strictly increasing
3. **Data Type Detection**: Automatic classification as reflectance/absorbance
4. **Duplicate Handling**: Warnings for duplicate sample IDs or wavelengths
5. **Missing Data**: NaN detection and reporting

## Error Handling

The architecture provides clear error messages:

- **Missing Dependencies**: Helpful installation instructions
- **Invalid Formats**: Clear format specification errors
- **File Not Found**: Standard file system errors
- **Validation Failures**: Descriptive validation error messages
- **Parse Errors**: Detailed parsing error information

## Extension Points

To add a new format:

1. Implement reader function: `read_<format>_file(path, **kwargs) -> (DataFrame, dict)`
2. Implement writer function: `write_<format>_file(data, path, metadata, **kwargs)`
3. Add extension mapping to `detect_format()`
4. Add dispatcher case to `read_spectra()` and `write_spectra()`
5. Create test file: `test_io_<format>.py`
6. Update format support table in README

## Performance Considerations

- **Lazy Loading**: Directories are scanned only when needed
- **Efficient Parsing**: Uses pandas and numpy for fast I/O
- **Memory Efficient**: Streaming where possible for large files
- **Caching**: Format detection caches results when appropriate

## Future Enhancements

1. **Multi-file Writers**: Support writing multiple spectra to formats that allow it
2. **Batch Processing**: Parallel reading of multiple files
3. **Metadata Preservation**: Store and retrieve custom metadata in all formats
4. **Format Conversion**: Direct format-to-format conversion utilities
5. **Agilent Support**: Complete implementation of Agilent format reader
6. **Additional Formats**: Support for additional vendor-specific formats

## Dependencies

### Core (Included in Default Install)
- pandas >= 1.3.0
- numpy >= 1.21.0
- openpyxl >= 3.1.0
- xlsxwriter >= 3.2.0
- jcamp >= 1.3.0
- spc-io >= 0.2.0

### Optional (Install Separately)
- specdal - Binary ASD support
- brukeropus - Bruker OPUS support
- specio - PerkinElmer support
- agilent-ir-formats - Agilent support

## Installation Commands

```bash
# Basic installation
pip install -e .

# With specific format support
pip install -e ".[asd]"           # Binary ASD
pip install -e ".[opus]"          # Bruker OPUS
pip install -e ".[perkinelmer]"   # PerkinElmer
pip install -e ".[agilent]"       # Agilent

# All formats
pip install -e ".[all-formats]"
```

## Backward Compatibility

The unified architecture maintains full backward compatibility:

- All existing functions (`read_csv_spectra`, `read_asd_dir`, etc.) remain available
- Existing code continues to work without modification
- New unified API (`read_spectra`, `write_spectra`) recommended for new code
- Consistent return format: (DataFrame, metadata) tuple

## File Organization

```
src/spectral_predict/io.py
├── Existing functions (unchanged)
│   ├── read_csv_spectra()
│   ├── read_asd_dir()
│   ├── read_spc_dir()
│   └── ...
├── Unified I/O Architecture (new)
│   ├── detect_format()
│   ├── read_spectra()
│   ├── write_spectra()
│   └── _detect_directory_format()
└── Format-Specific Readers/Writers (new)
    ├── read_excel_spectra()
    ├── read_jcamp_file()
    ├── read_ascii_spectra()
    ├── read_opus_file()
    ├── write_excel_spectra()
    ├── write_jcamp_file()
    └── ...

tests/
├── test_io_csv.py (existing)
├── test_asd_ascii.py (existing)
├── test_io_excel.py (new)
├── test_io_jcamp.py (new)
├── test_io_spc.py (new)
├── test_io_vendor_formats.py (new)
└── test_io_unified.py (new)
```

## Summary

The unified I/O architecture provides:

- ✅ **10+ Format Support**: Comprehensive coverage of spectroscopy file formats
- ✅ **Auto-Detection**: Seamless format identification
- ✅ **Unified API**: Consistent interface across all formats
- ✅ **105 Tests**: Comprehensive test coverage
- ✅ **Rich Metadata**: Detailed information returned from all readers
- ✅ **Flexible Dependencies**: Optional packages for vendor formats
- ✅ **Backward Compatible**: No breaking changes to existing code
- ✅ **Well Documented**: Complete README with examples and format table
- ✅ **Extensible**: Easy to add new formats

This architecture positions Spectral Predict as a comprehensive solution for spectroscopic data analysis with best-in-class file format support.
