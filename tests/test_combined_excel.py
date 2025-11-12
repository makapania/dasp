"""
Test combined Excel format reading (spectra + targets in one file).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectral_predict.io import read_combined_excel, detect_combined_excel_format


def create_test_combined_excel():
    """Create a test Excel file with spectra + targets."""
    # Generate synthetic spectral data
    n_samples = 50
    wavelengths = np.arange(400, 2501, 1)  # 2101 wavelengths

    # Create DataFrame with specimen IDs, wavelengths, and target
    data = {}

    # Add specimen IDs
    data['specimen_id'] = [f'Sample_{i+1}' for i in range(n_samples)]

    # Add wavelength columns
    for wl in wavelengths:
        # Simulate reflectance spectra (0-1 range)
        data[str(wl)] = np.random.uniform(0.1, 0.9, n_samples)

    # Add target variable
    data['collagen'] = np.random.uniform(5.0, 15.0, n_samples)

    df = pd.DataFrame(data)

    # Save to Excel
    output_path = Path('test_combined_spectra.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Created test file: {output_path}")
    return output_path


def create_test_combined_excel_no_id():
    """Create a test Excel file WITHOUT specimen ID column."""
    # Generate synthetic spectral data
    n_samples = 30
    wavelengths = np.arange(400, 2501, 1)  # 2101 wavelengths

    # Create DataFrame with wavelengths and target (NO ID column)
    data = {}

    # Add wavelength columns
    for wl in wavelengths:
        data[str(wl)] = np.random.uniform(0.1, 0.9, n_samples)

    # Add target variable with REPEATED values (realistic for measurements)
    # Real target variables have low uniqueness (<80%)
    base_values = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]  # 10 unique values
    data['nitrogen'] = np.random.choice(base_values, size=n_samples)  # Repeat values

    df = pd.DataFrame(data)

    # Save to Excel
    output_path = Path('test_combined_spectra_no_id.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Created test file (no ID): {output_path}")
    return output_path


def create_test_combined_excel_mixed_order():
    """Create Excel file with target column BEFORE wavelengths."""
    # Generate synthetic spectral data
    n_samples = 40
    wavelengths = np.arange(400, 2501, 1)  # 2101 wavelengths

    # Create DataFrame with target FIRST, then ID, then wavelengths
    data = {}

    # Target column first
    data['protein'] = np.random.uniform(10.0, 30.0, n_samples)

    # Then specimen ID
    data['sample_name'] = [f'Spec_{i+1}' for i in range(n_samples)]

    # Then wavelength columns
    for wl in wavelengths:
        data[str(wl)] = np.random.uniform(0.1, 0.9, n_samples)

    df = pd.DataFrame(data)

    # Save to Excel
    output_path = Path('test_combined_spectra_mixed.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Created test file (mixed order): {output_path}")
    return output_path


def test_combined_excel_with_id():
    """Test reading combined Excel with specimen ID column."""
    print("\n" + "="*60)
    print("TEST 1: Combined Excel WITH Specimen ID")
    print("="*60)

    file_path = create_test_combined_excel()

    try:
        X, y, metadata = read_combined_excel(file_path)

        print(f"\n[OK] Successfully read combined Excel file")
        print(f"  Spectra shape: {X.shape}")
        print(f"  Targets shape: {y.shape}")
        print(f"  Specimen ID column: {metadata['specimen_id_col']}")
        print(f"  Target column: {metadata['y_col']}")
        print(f"  Wavelength range: {metadata['wavelength_range']}")
        print(f"  Data type: {metadata['data_type']} ({metadata['type_confidence']:.1f}%)")
        print(f"  Generated IDs: {metadata['generated_ids']}")

        # Validate
        assert X.shape[0] == y.shape[0], "Mismatch in number of samples"
        assert X.shape[1] >= 100, "Too few wavelengths"
        assert metadata['specimen_id_col'] == 'specimen_id'
        assert metadata['y_col'] == 'collagen'
        assert not metadata['generated_ids']

        print("\n[OK] All validations passed!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink()


def test_combined_excel_no_id():
    """Test reading combined Excel WITHOUT specimen ID (auto-generate)."""
    print("\n" + "="*60)
    print("TEST 2: Combined Excel WITHOUT Specimen ID (auto-generate)")
    print("="*60)

    file_path = create_test_combined_excel_no_id()

    try:
        # First, read the file and check columns
        df_check = pd.read_excel(file_path)
        print(f"\nColumns in Excel file: {list(df_check.columns[:5])} ... {list(df_check.columns[-3:])}")
        print(f"Total columns: {len(df_check.columns)}")

        X, y, metadata = read_combined_excel(file_path)

        print(f"\n[OK] Successfully read combined Excel file (no ID)")
        print(f"  Spectra shape: {X.shape}")
        print(f"  Targets shape: {y.shape}")
        print(f"  Specimen ID column: {metadata['specimen_id_col']}")
        print(f"  Target column: {metadata['y_col']}")
        print(f"  Generated IDs: {metadata['generated_ids']}")
        print(f"  First 5 IDs: {list(X.index[:5])}")

        # Validate
        assert X.shape[0] == y.shape[0]
        assert metadata['specimen_id_col'] == '__GENERATED__'
        assert metadata['y_col'] == 'nitrogen'
        assert metadata['generated_ids'] == True
        assert X.index[0] == 'Sample_1'
        assert X.index[1] == 'Sample_2'

        print("\n[OK] All validations passed!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink()


def test_combined_excel_mixed_order():
    """Test reading combined Excel with columns in mixed order."""
    print("\n" + "="*60)
    print("TEST 3: Combined Excel with MIXED Column Order")
    print("="*60)

    file_path = create_test_combined_excel_mixed_order()

    try:
        X, y, metadata = read_combined_excel(file_path)

        print(f"\n[OK] Successfully read combined Excel file (mixed order)")
        print(f"  Spectra shape: {X.shape}")
        print(f"  Targets shape: {y.shape}")
        print(f"  Specimen ID column: {metadata['specimen_id_col']}")
        print(f"  Target column: {metadata['y_col']}")
        print(f"  Column order was: protein, sample_name, wavelengths...")

        # Validate
        assert X.shape[0] == y.shape[0]
        assert metadata['specimen_id_col'] == 'sample_name'
        assert metadata['y_col'] == 'protein'
        assert not metadata['generated_ids']

        print("\n[OK] All validations passed!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink()


def test_detect_combined_excel():
    """Test detection of combined Excel format in a directory."""
    print("\n" + "="*60)
    print("TEST 4: Detect Combined Excel Format")
    print("="*60)

    # Create a test Excel file
    file_path = create_test_combined_excel()
    directory = file_path.parent

    try:
        is_combined, detected_path, sheet_name = detect_combined_excel_format(directory)

        print(f"\n[OK] Detection successful")
        print(f"  Is combined format: {is_combined}")
        print(f"  Detected file: {detected_path}")
        print(f"  Sheet name: {sheet_name}")

        # Validate
        assert is_combined == True, "Should detect single Excel file as combined"
        assert detected_path == str(file_path), "Should detect correct file"
        assert sheet_name == 0, "Should default to first sheet"

        print("\n[OK] Detection validated!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    finally:
        # Cleanup
        if file_path.exists():
            file_path.unlink()


if __name__ == '__main__':
    print("="*60)
    print("COMBINED EXCEL FORMAT TESTS")
    print("="*60)

    try:
        test_combined_excel_with_id()
        test_combined_excel_no_id()
        test_combined_excel_mixed_order()
        test_detect_combined_excel()

        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"[FAIL] TESTS FAILED: {e}")
        print("="*60)
        sys.exit(1)
