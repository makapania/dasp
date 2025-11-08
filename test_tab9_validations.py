#!/usr/bin/env python3
"""
Test script to verify Tab 9 validation checks are present in spectral_predict_gui_optimized.py

This script reads the GUI file and checks for the presence of all expected validation checks.
"""

import re
import sys

def test_validations():
    """Check for presence of all validation checks."""

    filepath = r"C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return False

    print("=" * 80)
    print("Tab 9 Validation Check Verification")
    print("=" * 80)
    print()

    # Define expected validation checks
    checks = {
        "Section B: Load Paired Spectra": [
            ("Same Instrument (pre-load)", r"Same Instrument Selected.*Master and slave instruments must be different"),
            ("Sample Count Mismatch", r"Sample Count Mismatch.*Master has.*samples.*Slave has.*samples"),
            ("Few Samples Warning", r"Few Samples.*Only.*paired samples loaded"),
            ("No Wavelength Overlap", r"No Wavelength Overlap.*Instruments must have overlapping wavelength"),
            ("Limited Wavelength Overlap", r"Limited Wavelength Overlap.*overlap is.*% of instrument range"),
            ("Overlap in Info Display", r"Wavelength overlap:.*min_overlap_pct"),
        ],
        "Section C: Build Transfer Model": [
            ("Data Loaded Check (hasattr)", r"No Paired Spectra Loaded.*Please load paired standardization spectra"),
            ("Same Instrument Check", r"Same Instrument Selected.*Master and slave instruments must be different"),
            ("DS Lambda Range Check", r"DS Ridge Lambda must be between 0 and 100"),
            ("DS Lambda Type Check", r"DS Ridge Lambda must be a number"),
            ("PDS Window Range Check", r"PDS Window must be between 5 and 101"),
            ("PDS Window Odd Check", r"PDS Window must be an odd number"),
            ("PDS Window Type Check", r"PDS Window must be an integer"),
        ],
        "Section E: Predict with Transfer Model": [
            ("Master Model Check", r"Master Model Not Loaded.*Please load the master model in Section A"),
            ("Transfer Model Check", r"Transfer Model Not Loaded.*Please load or build a transfer model"),
            ("Wavelength Compatibility", r"Wavelength Range Mismatch.*Transfer model expects wavelengths"),
            ("Extrapolation Warning", r"Extrapolation Warning.*exceed master model training range"),
        ],
    }

    all_passed = True
    total_checks = 0
    passed_checks = 0

    for section, section_checks in checks.items():
        print(f"\n{section}")
        print("-" * 80)

        for check_name, pattern in section_checks:
            total_checks += 1
            # Use DOTALL flag to match across newlines
            if re.search(pattern, content, re.DOTALL):
                print(f"  [PASS] {check_name}")
                passed_checks += 1
            else:
                print(f"  [FAIL] {check_name}")
                all_passed = False

    print()
    print("=" * 80)
    print(f"Summary: {passed_checks}/{total_checks} checks passed")
    print("=" * 80)

    if all_passed:
        print("\n[SUCCESS] All Tab 9 validation checks are present!")
        return True
    else:
        print(f"\n[WARNING] {total_checks - passed_checks} validation checks are missing or incorrect.")
        return False

if __name__ == "__main__":
    success = test_validations()
    sys.exit(0 if success else 1)
