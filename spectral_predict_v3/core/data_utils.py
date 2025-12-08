"""
Data alignment utilities for Spectral Predict V3.

This module provides robust sample ID matching between spectral data and reference Y values.
Implements progressive matching strategies to handle real-world filename variations.

Author: Spectral Predict Development Team
"""

import re
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


def _is_nan_safe(val) -> bool:
    """Check if value is NaN, handling any type safely."""
    if val is None:
        return True
    if isinstance(val, float):
        return np.isnan(val)
    if isinstance(val, str):
        return val.strip() == '' or val.lower() == 'nan'
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def _count_nan_safe(values) -> int:
    """Count NaN values in array, handling any dtype."""
    return sum(1 for v in values if _is_nan_safe(v))


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for flexible matching.

    Removes common spectral file extensions, spaces, underscores, and converts to lowercase.
    This enables matching between files with variations in naming conventions.

    Args:
        filename: Original filename or sample ID

    Returns:
        Normalized filename (lowercase, no spaces/underscores/extensions)

    Examples:
        >>> normalize_filename("Sample 001.asd")
        'sample001'
        >>> normalize_filename("Spectrum_042.csv")
        'spectrum042'
        >>> normalize_filename("TEST.SPC")
        'test'
    """
    if not isinstance(filename, str):
        filename = str(filename)

    # Remove common spectral file extensions (case-insensitive)
    extensions = [".asd", ".sig", ".csv", ".txt", ".spc", ".xlsx", ".xls"]
    filename_lower = filename.lower()

    for ext in extensions:
        if filename_lower.endswith(ext):
            filename = filename[:-len(ext)]
            break

    # Remove spaces and underscores, convert to lowercase
    filename = filename.replace(" ", "").replace("_", "").lower()

    return filename


def extract_numeric_id(filename: str) -> Optional[str]:
    """
    Extract trailing numeric ID from filename, stripping leading zeros.

    This handles cases where filenames have numeric suffixes like:
    - "Spectrum00001" -> "1"
    - "Sample_042" -> "42"
    - "test001.asd" -> "1"

    Args:
        filename: Filename to extract numeric ID from

    Returns:
        Numeric ID with leading zeros stripped, or None if no trailing numbers found

    Examples:
        >>> extract_numeric_id("Spectrum00001")
        '1'
        >>> extract_numeric_id("Sample_042.asd")
        '42'
        >>> extract_numeric_id("NoNumbers")
        None
    """
    if not isinstance(filename, str):
        filename = str(filename)

    # First normalize to remove extensions
    normalized = normalize_filename(filename)

    # Find trailing digits
    match = re.search(r'(\d+)$', normalized)

    if match:
        # Strip leading zeros but keep at least one digit
        numeric_str = match.group(1).lstrip('0') or '0'
        return numeric_str

    return None


def align_xy(
    sample_ids: List[str],
    y_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    return_alignment_info: bool = False
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Align spectral sample IDs with reference Y values using progressive matching strategies.

    This function implements a robust matching algorithm that tries multiple strategies:
    1. Exact match (case-sensitive)
    2. Without extension match ("sample.asd" matches "sample")
    3. Case-insensitive match
    4. Normalized match (removes spaces/underscores)
    5. Numeric ID extraction (strips leading zeros)

    Args:
        sample_ids: List of sample IDs from spectral data (e.g., filenames)
        y_df: DataFrame containing reference Y values
        id_column: Column name in y_df containing sample IDs
        target_column: Column name in y_df containing target Y values
        return_alignment_info: If True, return detailed alignment diagnostics

    Returns:
        If return_alignment_info=False:
            y_values: numpy array of Y values aligned to sample_ids (NaN for unmatched)

        If return_alignment_info=True:
            y_values: numpy array of Y values aligned to sample_ids (NaN for unmatched)
            alignment_info: dict with keys:
                - matched_ids: List of successfully matched sample IDs
                - unmatched_spectra: List of spectral IDs with no Y value
                - unmatched_reference: List of Y file IDs with no spectrum
                - n_nan_dropped: Number of matched samples with NaN target values
                - n_matched: Total number of successful matches
                - used_fuzzy_matching: Whether non-exact matching was used
                - match_strategy_used: Primary matching strategy ('exact', 'normalized', 'numeric')

    Raises:
        ValueError: If id_column or target_column not found in y_df
        TypeError: If inputs are wrong type

    Examples:
        >>> sample_ids = ["Spectrum001.asd", "Spectrum002.asd"]
        >>> y_df = pd.DataFrame({
        ...     'SampleID': ['Spectrum 001', 'Spectrum 002'],
        ...     'Concentration': [1.5, 2.3]
        ... })
        >>> y_values, info = align_xy(sample_ids, y_df, 'SampleID', 'Concentration',
        ...                            return_alignment_info=True)
        >>> y_values
        array([1.5, 2.3])
        >>> info['n_matched']
        2
    """
    # ===== INPUT VALIDATION =====
    print("\n=== Starting align_xy ===")
    print(f"Sample IDs count: {len(sample_ids)}")
    print(f"Y DataFrame shape: {y_df.shape}")

    if not isinstance(sample_ids, (list, np.ndarray, pd.Series)):
        raise TypeError(f"sample_ids must be list, array, or Series, got {type(sample_ids)}")

    if not isinstance(y_df, pd.DataFrame):
        raise TypeError(f"y_df must be a pandas DataFrame, got {type(y_df)}")

    if id_column not in y_df.columns:
        raise ValueError(f"id_column '{id_column}' not found in y_df. Available: {list(y_df.columns)}")

    if target_column not in y_df.columns:
        raise ValueError(f"target_column '{target_column}' not found in y_df. Available: {list(y_df.columns)}")

    # Convert sample_ids to list of strings
    sample_ids = [str(sid) for sid in sample_ids]

    # ===== HANDLE EMPTY Y FILE =====
    if y_df.empty or len(y_df) == 0:
        print("WARNING: Y DataFrame is empty. Returning all NaN values.")
        y_values = np.full(len(sample_ids), np.nan)

        if return_alignment_info:
            alignment_info = {
                'matched_ids': [],
                'unmatched_spectra': sample_ids.copy(),
                'unmatched_reference': [],
                'n_nan_dropped': 0,
                'n_matched': 0,
                'used_fuzzy_matching': False,
                'match_strategy_used': 'none_empty_y'
            }
            return y_values, alignment_info
        return y_values, None

    # ===== CHECK FOR DUPLICATES IN Y FILE =====
    y_ids = y_df[id_column].astype(str).tolist()
    duplicate_y_ids = [id for id in set(y_ids) if y_ids.count(id) > 1]

    if duplicate_y_ids:
        print(f"WARNING: Found {len(duplicate_y_ids)} duplicate IDs in Y file: {duplicate_y_ids[:5]}")
        print("Using first occurrence for each duplicate.")
        # Keep first occurrence of each ID
        y_df = y_df.drop_duplicates(subset=[id_column], keep='first')
        y_ids = y_df[id_column].astype(str).tolist()

    # ===== STRATEGY 1: EXACT MATCH =====
    print("\nTrying Strategy 1: Exact match...")
    y_id_to_value = dict(zip(y_ids, y_df[target_column].values))

    exact_matches = {}
    for sample_id in sample_ids:
        if sample_id in y_id_to_value:
            exact_matches[sample_id] = y_id_to_value[sample_id]

    print(f"Exact matches: {len(exact_matches)}/{len(sample_ids)}")

    if len(exact_matches) == len(sample_ids):
        # All matched exactly, we're done
        y_values = np.array([exact_matches[sid] for sid in sample_ids], dtype=object)

        # Count NaN values (safe for any dtype)
        n_nan = _count_nan_safe(y_values)
        if n_nan > 0:
            print(f"WARNING: {n_nan} matched samples have NaN target values")

        if return_alignment_info:
            matched_with_values = [sid for sid in sample_ids if not _is_nan_safe(exact_matches[sid])]
            alignment_info = {
                'matched_ids': list(exact_matches.keys()),
                'unmatched_spectra': [],
                'unmatched_reference': [yid for yid in y_ids if yid not in sample_ids],
                'n_nan_dropped': n_nan,
                'n_matched': len(exact_matches),
                'used_fuzzy_matching': False,
                'match_strategy_used': 'exact'
            }
            print(f"Alignment complete: {alignment_info['n_matched']} matched (exact)")
            return y_values, alignment_info

        return y_values, None

    # ===== STRATEGY 2-5: PROGRESSIVE FUZZY MATCHING =====
    print("\nExact match incomplete. Trying fuzzy matching strategies...")

    # Build lookup maps for Y file IDs
    y_normalized_map = {}  # normalized -> (original_id, value)
    y_numeric_map = {}     # numeric_id -> (original_id, value)

    for y_id, y_value in y_id_to_value.items():
        # Normalized lookup
        normalized = normalize_filename(y_id)
        if normalized not in y_normalized_map:  # Keep first if duplicates after normalization
            y_normalized_map[normalized] = (y_id, y_value)

        # Numeric lookup - ONLY if Y ID is purely numeric (after normalization)
        # This prevents "other1" from matching "spec1" via numeric ID "1"
        # But allows "1" or "001" to match "Spectrum00001.asd"
        normalized_y = normalize_filename(y_id)
        if normalized_y.isdigit():  # Y ID is purely numeric
            numeric_id = normalized_y.lstrip('0') or '0'
            if numeric_id not in y_numeric_map:
                y_numeric_map[numeric_id] = (y_id, y_value)

    print(f"Built normalized lookup: {len(y_normalized_map)} entries")
    print(f"Built numeric lookup: {len(y_numeric_map)} entries")

    # Try to match each sample ID
    matches = {}  # sample_id -> (y_value, strategy_used)
    match_strategies_count = {'exact': 0, 'normalized': 0, 'numeric': 0}

    for sample_id in sample_ids:
        # Check if already matched exactly
        if sample_id in exact_matches:
            matches[sample_id] = (exact_matches[sample_id], 'exact')
            match_strategies_count['exact'] += 1
            continue

        # Try normalized matching
        normalized = normalize_filename(sample_id)
        if normalized in y_normalized_map:
            original_y_id, y_value = y_normalized_map[normalized]
            matches[sample_id] = (y_value, 'normalized')
            match_strategies_count['normalized'] += 1
            print(f"  Normalized match: '{sample_id}' -> '{original_y_id}'")
            continue

        # Try numeric matching
        numeric_id = extract_numeric_id(sample_id)
        if numeric_id and numeric_id in y_numeric_map:
            original_y_id, y_value = y_numeric_map[numeric_id]
            matches[sample_id] = (y_value, 'numeric')
            match_strategies_count['numeric'] += 1
            print(f"  Numeric match: '{sample_id}' (ID={numeric_id}) -> '{original_y_id}'")
            continue

        # No match found
        matches[sample_id] = (np.nan, 'none')

    print(f"\nMatching complete:")
    print(f"  Exact: {match_strategies_count['exact']}")
    print(f"  Normalized: {match_strategies_count['normalized']}")
    print(f"  Numeric: {match_strategies_count['numeric']}")
    print(f"  Unmatched: {len(sample_ids) - sum(match_strategies_count.values())}")

    # Build aligned Y values array (dtype=object to handle strings)
    y_values = np.array([matches[sid][0] for sid in sample_ids], dtype=object)

    # Determine primary strategy used
    if match_strategies_count['normalized'] > 0 or match_strategies_count['numeric'] > 0:
        used_fuzzy = True
        if match_strategies_count['normalized'] > match_strategies_count['numeric']:
            primary_strategy = 'normalized'
        elif match_strategies_count['numeric'] > 0:
            primary_strategy = 'numeric'
        else:
            primary_strategy = 'exact'
    else:
        used_fuzzy = False
        primary_strategy = 'exact'

    # ===== BUILD ALIGNMENT INFO =====
    matched_ids = [sid for sid, (val, strategy) in matches.items() if strategy != 'none']
    unmatched_spectra = [sid for sid, (val, strategy) in matches.items() if strategy == 'none']

    # Count NaN values in matched samples (safe for any dtype)
    n_nan = sum(1 for sid in matched_ids if _is_nan_safe(matches[sid][0]))
    if n_nan > 0:
        print(f"WARNING: {n_nan} matched samples have NaN target values")

    # Find Y IDs that weren't matched to any spectrum
    matched_y_ids = set()
    for sample_id in matched_ids:
        _, strategy = matches[sample_id]
        if strategy == 'exact':
            matched_y_ids.add(sample_id)
        elif strategy == 'normalized':
            normalized = normalize_filename(sample_id)
            if normalized in y_normalized_map:
                matched_y_ids.add(y_normalized_map[normalized][0])
        elif strategy == 'numeric':
            numeric_id = extract_numeric_id(sample_id)
            if numeric_id and numeric_id in y_numeric_map:
                matched_y_ids.add(y_numeric_map[numeric_id][0])

    unmatched_reference = [yid for yid in y_ids if yid not in matched_y_ids]

    # ===== FINAL REPORTING =====
    print(f"\n=== Alignment Summary ===")
    print(f"Total samples: {len(sample_ids)}")
    print(f"Matched: {len(matched_ids)} ({len(matched_ids)/len(sample_ids)*100:.1f}%)")
    print(f"Unmatched spectra: {len(unmatched_spectra)}")
    print(f"Unmatched reference: {len(unmatched_reference)}")
    print(f"NaN target values: {n_nan}")
    print(f"Primary matching strategy: {primary_strategy}")

    if unmatched_spectra:
        print(f"\nFirst 5 unmatched spectra: {unmatched_spectra[:5]}")
    if unmatched_reference:
        print(f"First 5 unmatched reference IDs: {unmatched_reference[:5]}")

    if return_alignment_info:
        alignment_info = {
            'matched_ids': matched_ids,
            'unmatched_spectra': unmatched_spectra,
            'unmatched_reference': unmatched_reference,
            'n_nan_dropped': n_nan,
            'n_matched': len(matched_ids),
            'used_fuzzy_matching': used_fuzzy,
            'match_strategy_used': primary_strategy
        }
        return y_values, alignment_info

    return y_values, None


# ===== CONVENIENCE FUNCTIONS =====

def print_alignment_report(alignment_info: Dict[str, Any], verbose: bool = True) -> None:
    """
    Pretty-print alignment results for debugging.

    Args:
        alignment_info: Dict returned by align_xy with return_alignment_info=True
        verbose: If True, print sample lists; if False, only print summary
    """
    print("\n" + "="*60)
    print("ALIGNMENT REPORT")
    print("="*60)

    print(f"\nMatching Strategy: {alignment_info['match_strategy_used']}")
    print(f"Fuzzy Matching Used: {alignment_info['used_fuzzy_matching']}")

    print(f"\nTotal Matched: {alignment_info['n_matched']}")
    print(f"Unmatched Spectra: {len(alignment_info['unmatched_spectra'])}")
    print(f"Unmatched Reference: {len(alignment_info['unmatched_reference'])}")
    print(f"NaN Target Values: {alignment_info['n_nan_dropped']}")

    if verbose:
        if alignment_info['unmatched_spectra']:
            print(f"\nUnmatched Spectra (showing first 10):")
            for sid in alignment_info['unmatched_spectra'][:10]:
                print(f"  - {sid}")

        if alignment_info['unmatched_reference']:
            print(f"\nUnmatched Reference IDs (showing first 10):")
            for yid in alignment_info['unmatched_reference'][:10]:
                print(f"  - {yid}")

    print("="*60 + "\n")


def validate_alignment(y_values: np.ndarray, min_samples: int = 10) -> Tuple[bool, str]:
    """
    Validate alignment results meet minimum requirements.

    Args:
        y_values: Aligned Y values array
        min_samples: Minimum number of non-NaN samples required

    Returns:
        is_valid: True if alignment is valid for modeling
        message: Description of validation result
    """
    n_total = len(y_values)
    n_valid = n_total - _count_nan_safe(y_values)

    if n_valid == 0:
        return False, "No valid Y values after alignment"

    if n_valid < min_samples:
        return False, f"Only {n_valid} valid samples, need at least {min_samples}"

    pct_valid = n_valid / n_total * 100

    if pct_valid < 50:
        return False, f"Only {pct_valid:.1f}% of samples matched (need >50%)"

    return True, f"Alignment valid: {n_valid}/{n_total} samples ({pct_valid:.1f}%)"
