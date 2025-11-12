"""Readers for various spectral file formats.

This package provides readers for vendor-specific spectroscopy file formats:

- Bruker OPUS (.0, .1, .2, etc.) - requires brukeropus
- PerkinElmer (.sp) - requires specio
- Agilent (.seq, .dmt, .asp, .bsw) - requires agilent-ir-formats
- ASD (.asd, .sig) - native Python or requires specdal for binary
- SPC (.spc) - requires spc-io
- JCAMP-DX (.jdx, .dx) - requires jcamp

Install optional dependencies:
    pip install spectral-predict[opus]           # Bruker OPUS
    pip install spectral-predict[perkinelmer]    # PerkinElmer
    pip install spectral-predict[agilent]        # Agilent
    pip install spectral-predict[all-formats]    # All vendor formats
"""

# Bruker OPUS readers
from spectral_predict.readers.opus_reader import (
    read_opus_file,
    read_opus_dir,
    convert_wavenumber_to_wavelength,
    convert_wavelength_to_wavenumber,
)

# PerkinElmer readers
from spectral_predict.readers.perkinelmer_reader import (
    read_sp_file,
    read_sp_dir,
)

# Agilent readers
from spectral_predict.readers.agilent_reader import (
    read_agilent_file,
    read_agilent_dir,
    read_seq_file,
    read_dmt_file,
    read_asp_file,
)

__all__ = [
    # Bruker OPUS
    'read_opus_file',
    'read_opus_dir',
    'convert_wavenumber_to_wavelength',
    'convert_wavelength_to_wavenumber',
    # PerkinElmer
    'read_sp_file',
    'read_sp_dir',
    # Agilent
    'read_agilent_file',
    'read_agilent_dir',
    'read_seq_file',
    'read_dmt_file',
    'read_asp_file',
]
