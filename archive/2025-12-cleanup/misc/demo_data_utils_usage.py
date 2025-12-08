"""
Demonstration of data_utils.py for real-world spectral data alignment.
This shows how V3 will use this module for multi-import scenarios.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r'C:\Users\sponheim\git\dasp')

from spectral_predict_v3.core.data_utils import align_xy, print_alignment_report, validate_alignment

def demo_real_world_scenario():
    """Simulate a real-world spectral data import scenario."""
    print("="*70)
    print("DEMO: Real-World Spectral Data Alignment")
    print("="*70)

    # Scenario: User imports spectral files with .asd extensions
    # Reference Y file has IDs without extensions and inconsistent spacing
    print("\nScenario: Import .asd spectral files, align with Y reference file")
    print("-"*70)

    # Spectral file IDs (as they appear in ASD file system)
    spectral_ids = [
        "Corn Sample 001.asd",
        "Corn Sample 002.asd",
        "Corn Sample 003.asd",
        "Wheat Spectrum 0010.asd",
        "Wheat Spectrum 0011.asd",
        "Wheat Spectrum 0012.asd",
        "Unknown_Sample.asd",  # Will not match
    ]

    # Reference Y file (Excel/CSV with inconsistent formatting)
    y_df = pd.DataFrame({
        'SampleID': [
            'CornSample001',      # No spaces, no extension
            'Corn Sample 002',    # Has spaces, no extension
            'CORN SAMPLE 003',    # Different case
            '10',                 # Just the number (will match via numeric)
            '11',                 # Just the number
            '12',                 # Just the number
            'ExtraSample999',     # Not in spectral data
        ],
        'Protein': [12.5, 13.2, 11.8, 8.5, 9.1, 8.8, 15.0]
    })

    print(f"\nSpectral files imported: {len(spectral_ids)}")
    print("Sample spectral IDs:")
    for i, sid in enumerate(spectral_ids[:3], 1):
        print(f"  {i}. {sid}")

    print(f"\nReference Y file shape: {y_df.shape}")
    print("Y file preview:")
    print(y_df.head())

    # Perform alignment
    print("\n" + "="*70)
    print("PERFORMING ALIGNMENT")
    print("="*70)

    y_values, alignment_info = align_xy(
        sample_ids=spectral_ids,
        y_df=y_df,
        id_column='SampleID',
        target_column='Protein',
        return_alignment_info=True
    )

    # Print detailed results
    print_alignment_report(alignment_info, verbose=True)

    # Show aligned values
    print("\nAligned Y values (Protein %):")
    for sid, y_val in zip(spectral_ids, y_values):
        if np.isnan(y_val):
            print(f"  {sid:40s} -> No match (NaN)")
        else:
            print(f"  {sid:40s} -> {y_val:.1f}%")

    # Validate alignment
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    is_valid, msg = validate_alignment(y_values, min_samples=5)
    print(f"Alignment valid: {is_valid}")
    print(f"Message: {msg}")

    if is_valid:
        print("\nAlignment is ready for modeling!")
        print(f"  - {alignment_info['n_matched']} samples matched")
        print(f"  - {len(alignment_info['unmatched_spectra'])} spectra without Y values (will be excluded)")
        print(f"  - {len(alignment_info['unmatched_reference'])} Y values without spectra")
    else:
        print("\nWARNING: Alignment may not be suitable for modeling")

    # Show which matching strategies were used
    print("\n" + "="*70)
    print("MATCHING ANALYSIS")
    print("="*70)
    print(f"Fuzzy matching used: {alignment_info['used_fuzzy_matching']}")
    print(f"Primary strategy: {alignment_info['match_strategy_used']}")

    if alignment_info['used_fuzzy_matching']:
        print("\nNote: Fuzzy matching successfully handled filename variations:")
        print("  - Extension differences (.asd vs no extension)")
        print("  - Space/underscore differences")
        print("  - Case differences (CORN vs Corn)")
        print("  - Numeric ID extraction (Wheat Spectrum 0010 -> 10)")

    return y_values, alignment_info


def demo_edge_cases():
    """Demonstrate handling of problematic scenarios."""
    print("\n\n" + "="*70)
    print("DEMO: Edge Case Handling")
    print("="*70)

    # Edge case 1: Mostly unmatched data
    print("\n1. Low match rate (should warn):")
    print("-"*70)

    spectral_ids = ["spec1", "spec2", "spec3", "spec4", "spec5"]
    y_df = pd.DataFrame({
        'ID': ['other1', 'spec1'],  # Only 1 match
        'Value': [10.0, 20.0]
    })

    y_values, info = align_xy(spectral_ids, y_df, 'ID', 'Value', return_alignment_info=True)
    is_valid, msg = validate_alignment(y_values, min_samples=3)

    print(f"Match rate: {info['n_matched']}/{len(spectral_ids)} ({info['n_matched']/len(spectral_ids)*100:.0f}%)")
    print(f"Valid for modeling: {is_valid}")
    print(f"Validation message: {msg}")

    # Edge case 2: Empty Y file
    print("\n2. Empty reference file (should handle gracefully):")
    print("-"*70)

    spectral_ids = ["spec1", "spec2"]
    y_df = pd.DataFrame(columns=['ID', 'Value'])

    y_values, info = align_xy(spectral_ids, y_df, 'ID', 'Value', return_alignment_info=True)
    is_valid, msg = validate_alignment(y_values, min_samples=1)

    print(f"Matches: {info['n_matched']}")
    print(f"Valid for modeling: {is_valid}")
    print(f"Validation message: {msg}")

    # Edge case 3: Duplicate IDs in Y file
    print("\n3. Duplicate IDs in Y file (should use first):")
    print("-"*70)

    spectral_ids = ["sample1"]
    y_df = pd.DataFrame({
        'ID': ['sample1', 'sample1', 'sample1'],
        'Value': [10.0, 20.0, 30.0]
    })

    y_values, info = align_xy(spectral_ids, y_df, 'ID', 'Value', return_alignment_info=True)
    print(f"Y value used: {y_values[0]} (should be 10.0, the first occurrence)")


if __name__ == "__main__":
    print("\n")
    print("#"*70)
    print("# DATA_UTILS.PY DEMONSTRATION")
    print("# Real-world usage scenarios for V3 multi-import")
    print("#"*70)

    try:
        # Main demonstration
        demo_real_world_scenario()

        # Edge cases
        demo_edge_cases()

        print("\n\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\ndata_utils.py is ready for V3 integration!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
