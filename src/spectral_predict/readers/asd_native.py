"""Native Python reader for legacy binary ASD files.

This module decodes the *oldest* ASD binary format — version string ``b"ASD\\x00"`` —
which stores the spectrum as little-endian **float32** at byte offset 484. SpecDAL and
asdreader assume the modern layout (float64 at the same offset), so they misread these
legacy files and yield all-NaN spectra. Files of this vintage commonly carry instrument
extensions like ``.sco`` or numbered ``.000``/``.001`` rather than ``.asd``.

Modern ASD files (version strings ``as5``..``as8``) store float64 and are left to
SpecDAL: :func:`read_legacy_asd` returns ``None`` for anything that does not match the
legacy float32 layout, signalling the caller to fall back.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Legacy ASD-v1 binary header layout (offsets in bytes from start of file).
_MAGIC = b"ASD\x00"          # version string for the oldest format
_HEADER_BYTES = 484          # spectrum data begins here
_OFF_DATA_TYPE = 186         # uint8: 0=RAW 1=REF 2=RAD ... (1 == reflectance)
_OFF_CH1_WAVELENGTH = 191    # float32: wavelength of first channel (nm)
_OFF_WAVELENGTH_STEP = 195   # float32: nm per channel
_OFF_CHANNELS = 204          # uint16: number of channels
_BYTES_PER_CHANNEL = 4       # float32


def read_legacy_asd(asd_file) -> pd.Series | None:
    """Decode a legacy float32 ASD-v1 binary file.

    Args:
        asd_file: Path to a candidate binary ASD file.

    Returns:
        A spectrum as a ``pd.Series`` indexed by wavelength (nm), or ``None`` if the
        file is not the legacy float32 layout this reader handles (the caller should
        then fall back to SpecDAL).

    Raises:
        ValueError: If the file carries the legacy magic bytes but the header is
            internally inconsistent (corrupt/truncated), so the caller does not
            silently treat real corruption as "modern format, try SpecDAL".
    """
    asd_file = Path(asd_file)
    raw = asd_file.read_bytes()

    # Not a legacy-v1 file -> let the caller fall back to SpecDAL.
    if len(raw) < _HEADER_BYTES or raw[:4] != _MAGIC:
        return None

    ch1, step = struct.unpack_from("<f", raw, _OFF_CH1_WAVELENGTH)[0], struct.unpack_from(
        "<f", raw, _OFF_WAVELENGTH_STEP
    )[0]
    channels = struct.unpack_from("<H", raw, _OFF_CHANNELS)[0]
    data_type = raw[_OFF_DATA_TYPE]

    # Header bounds / sanity. These carry the legacy magic, so inconsistency means a
    # corrupt legacy file, not a modern one -> raise rather than return None.
    if not (0 < channels <= 65535):
        raise ValueError(
            f"{asd_file.name}: implausible channel count {channels} in ASD-v1 header"
        )
    if step == 0 or not np.isfinite(ch1) or not np.isfinite(step):
        raise ValueError(
            f"{asd_file.name}: invalid wavelength axis (ch1={ch1}, step={step})"
        )

    expected_f32 = _HEADER_BYTES + channels * 4
    expected_f64 = _HEADER_BYTES + channels * 8
    if len(raw) == expected_f64:
        # Modern float64 file that merely shares a leading "ASD" — hand it back to
        # SpecDAL rather than misreading it as float32.
        return None
    if len(raw) != expected_f32:
        # Carries the legacy magic but matches neither layout -> corrupt/truncated.
        raise ValueError(
            f"{asd_file.name}: ASD-v1 file size {len(raw)} bytes does not match "
            f"header (expected {expected_f32} for {channels} float32 channels)"
        )

    if data_type != 1:
        # 1 == reflectance. We still decode, but flag so callers/users can investigate.
        logger.warning(
            "%s: ASD dataType=%d (expected 1=reflectance); decoding values as-is.",
            asd_file.name,
            data_type,
        )

    values = np.frombuffer(raw, dtype="<f4", count=channels, offset=_HEADER_BYTES)

    # A spectrum with no finite value is a dead/corrupt payload (the same all-NaN
    # symptom SpecDAL produces on these files). Raise so the caller skips + reports
    # it rather than letting an all-NaN row silently enter the data matrix. Partial
    # NaNs are tolerated (real spectra can have isolated bad channels).
    if not np.any(np.isfinite(values)):
        raise ValueError(
            f"{asd_file.name}: decoded spectrum is entirely non-finite (corrupt payload)"
        )

    wavelengths = ch1 + step * np.arange(channels, dtype="float64")

    series = pd.Series(np.asarray(values, dtype="float64"), index=np.round(wavelengths, 2))
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series


def read_binary_asd(asd_file):
    """Read a binary ASD file using native Python.

    Currently supports only the legacy float32 ASD-v1 layout (see
    :func:`read_legacy_asd`). Modern float64 ASD files still require SpecDAL.

    Args:
        asd_file: Path to a binary ASD file.

    Returns:
        Spectrum as a ``pd.Series`` indexed by wavelength.

    Raises:
        NotImplementedError: If the file is not the supported legacy layout.
    """
    series = read_legacy_asd(asd_file)
    if series is not None:
        return series

    raise NotImplementedError(
        "Native Python binary ASD reader: modern (as5-as8) float64 files are not yet "
        "implemented. Only the legacy float32 ASD-v1 format (e.g. .sco / numbered .000 "
        "files) is supported.\n"
        "\n"
        "For modern (as5-as8) binary ASD files, options are:\n"
        "  1. Export ASD files to ASCII format (.sig or ASCII .asd)\n"
        "  2. Install SpecDAL: pip install specdal\n"
    )


# Modern raw (DN) files: as7 / as8 carry the sample DN block, a reference header, then the
# white-reference DN block, all float64, after the same 484-byte header.
_RAW_MAGICS = (b"as7", b"as8")
_OFF_DATA_FORMAT = 199  # uint8: 2 = float64
_REFERENCE_HEADER_BYTES = 20  # without the description; the description follows it


def read_asd_white_reference(asd_file) -> tuple[np.ndarray, np.ndarray] | None:
    """Wavelengths and white-reference DN stored in a raw modern ASD file.

    Only files that hold raw digital numbers (data type 0, float64) with a stored white
    reference are decoded: the as7 and as8 layouts. Everything else (legacy reflectance
    files, ASCII exports, files without a stored reference, other versions, anything
    truncated or inconsistent) returns ``None``; this reader never raises on content.

    Args:
        asd_file: Path to a candidate ASD file.

    Returns:
        ``(wavelengths_nm, white_reference_dn)``, each (channels,), or ``None``.
    """
    try:
        raw = Path(asd_file).read_bytes()
    except OSError:
        return None
    # No NUL is required after the magic: as7/as8 headers continue with the comment text.
    if len(raw) < _HEADER_BYTES or raw[:3] not in _RAW_MAGICS:
        return None
    if raw[_OFF_DATA_TYPE] != 0 or raw[_OFF_DATA_FORMAT] != 2:
        return None
    ch1 = struct.unpack_from("<f", raw, _OFF_CH1_WAVELENGTH)[0]
    step = struct.unpack_from("<f", raw, _OFF_WAVELENGTH_STEP)[0]
    n = struct.unpack_from("<H", raw, _OFF_CHANNELS)[0]
    if not (np.isfinite(ch1) and np.isfinite(step)) or step <= 0 or n < 2:
        return None
    ref_hdr = _HEADER_BYTES + 8 * n
    if len(raw) < ref_hdr + _REFERENCE_HEADER_BYTES:
        return None
    if raw[:3] == b"as7":
        if raw[ref_hdr] != 1:
            return None
        desc_len = struct.unpack_from("<H", raw, ref_hdr + 17)[0]
    else:
        if struct.unpack_from("<H", raw, ref_hdr)[0] != 0xFFFF:
            return None
        desc_len = struct.unpack_from("<H", raw, ref_hdr + 18)[0]
    white_off = ref_hdr + _REFERENCE_HEADER_BYTES + desc_len
    if len(raw) < white_off + 8 * n:
        return None
    white = np.frombuffer(raw, dtype="<f8", count=n, offset=white_off).astype(np.float64)
    wl = ch1 + step * np.arange(n, dtype=np.float64)
    return wl, white
