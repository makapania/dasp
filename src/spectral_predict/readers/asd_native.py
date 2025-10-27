"""Native Python reader for binary ASD files.

This module is a stub for future implementation of a pure-Python binary ASD reader.
For now, users should either:
1. Export ASD files to ASCII format (.sig or ASCII .asd), or
2. Install SpecDAL: pip install specdal
"""


def read_binary_asd(asd_file):
    """
    Read a binary ASD file using native Python.

    This is a placeholder for future implementation.

    Parameters
    ----------
    asd_file : Path
        Path to binary ASD file

    Returns
    -------
    pd.Series
        Spectrum with wavelengths as index

    Raises
    ------
    NotImplementedError
        Always raised - this function is not yet implemented

    Notes
    -----
    Binary ASD files contain:
    - Header with metadata (instrument info, GPS, date/time, etc.)
    - Spectral data (typically 2151 channels for ASD FieldSpec)
    - Wavelength calibration coefficients

    Implementation would require:
    - Parsing binary header structure
    - Extracting wavelength calibration
    - Reading spectral values
    - Applying calibration to generate wavelength array

    References
    ----------
    For ASD binary format details, see:
    - SpecDAL library: https://github.com/aviraldg/SpecDAL
    - asdreader R package: https://github.com/pierreroudier/asdreader
    """
    raise NotImplementedError(
        "Native Python binary ASD reader is not yet implemented.\n"
        "\n"
        "Options:\n"
        "  1. Export ASD files to ASCII format (.sig or ASCII .asd)\n"
        "  2. Install SpecDAL: pip install specdal\n"
        "  3. Use R bridge with asdreader package (see asd_r_bridge.py)\n"
        "\n"
        "To contribute a native reader, see:\n"
        "  - src/spectral_predict/readers/asd_native.py\n"
        "  - https://github.com/aviraldg/SpecDAL for format reference"
    )
