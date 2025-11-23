"""Test CSV import to diagnose permission error."""

import sys
from pathlib import Path

# Test the exact import flow
filepath = r"C:\Users\sponheim\Desktop\RawData_cleaned.csv"

print(f"Testing file import: {filepath}")
print(f"File exists: {Path(filepath).exists()}")
print(f"Is file: {Path(filepath).is_file()}")
print(f"Parent dir: {Path(filepath).parent}")
print()

# Test with spectral_predict.io
try:
    from spectral_predict.io import read_reference_csv

    print("Attempting read_reference_csv...")
    df = read_reference_csv(filepath, 'Sample_ID')
    print(f"SUCCESS! Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns[:5])}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()

# Test direct pandas read
try:
    import pandas as pd
    print("Attempting pd.read_csv...")
    df = pd.read_csv(filepath, nrows=5)
    print(f"SUCCESS! Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns[:5])}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
