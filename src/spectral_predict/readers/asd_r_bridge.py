"""R bridge for reading binary ASD files using R packages.

This module provides optional integration with R packages for reading binary ASD files:
- asdreader: https://github.com/pierreroudier/asdreader
- prospectr: https://github.com/l-ramirez-lopez/prospectr

Requires:
- R installed and available in PATH
- rpy2 Python package: pip install rpy2
- R packages: install.packages(c("asdreader", "prospectr"))
"""

import subprocess
from pathlib import Path


def check_r_available():
    """
    Check if R is available in the system PATH.

    Returns
    -------
    bool
        True if R (Rscript) is available, False otherwise
    """
    try:
        result = subprocess.run(["Rscript", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def read_asd_with_r(asd_file, r_package="asdreader"):
    """
    Read binary ASD file using R bridge.

    This is a placeholder for future implementation.

    Parameters
    ----------
    asd_file : Path
        Path to binary ASD file
    r_package : str
        R package to use ('asdreader' or 'prospectr')

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
    Implementation would use rpy2 to call R functions:

    For asdreader:
        ```r
        library(asdreader)
        spec <- get_spectra(read_asd("file.asd"))
        ```

    For prospectr:
        ```r
        library(prospectr)
        spec <- readASD("file.asd")
        ```

    References
    ----------
    - asdreader: https://github.com/pierreroudier/asdreader
    - prospectr: https://github.com/l-ramirez-lopez/prospectr
    - rpy2: https://rpy2.github.io/
    """
    if not check_r_available():
        raise RuntimeError(
            "R (Rscript) not found in PATH.\n"
            "To use R bridge:\n"
            "  1. Install R: https://www.r-project.org/\n"
            "  2. Ensure Rscript is in your PATH"
        )

    raise NotImplementedError(
        "R bridge for binary ASD files is not yet implemented.\n"
        "\n"
        "Options:\n"
        "  1. Export ASD files to ASCII format (.sig or ASCII .asd)\n"
        "  2. Install SpecDAL: pip install specdal\n"
        "  3. Wait for R bridge implementation\n"
        "\n"
        "To contribute R bridge implementation:\n"
        "  - Install rpy2: pip install rpy2\n"
        "  - Install R package: install.packages('asdreader')\n"
        "  - See src/spectral_predict/readers/asd_r_bridge.py"
    )


def read_asd_with_prospectr(asd_file):
    """
    Read binary ASD file using R prospectr package.

    Convenience wrapper for read_asd_with_r(..., r_package='prospectr').

    Parameters
    ----------
    asd_file : Path
        Path to binary ASD file

    Returns
    -------
    pd.Series
        Spectrum with wavelengths as index
    """
    return read_asd_with_r(asd_file, r_package="prospectr")


def read_asd_with_asdreader(asd_file):
    """
    Read binary ASD file using R asdreader package.

    Convenience wrapper for read_asd_with_r(..., r_package='asdreader').

    Parameters
    ----------
    asd_file : Path
        Path to binary ASD file

    Returns
    -------
    pd.Series
        Spectrum with wavelengths as index
    """
    return read_asd_with_r(asd_file, r_package="asdreader")
