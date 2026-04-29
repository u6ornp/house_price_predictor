"""
Check what columns are in your CSV file
"""

import pandas as pd

file_path = r"C:\Users\surfe\Downloads\realtor-data\realtor-data.csv"

print("=" * 70)
print("CHECKING YOUR CSV COLUMNS")
print("=" * 70)

try:
    df = pd.read_csv(file_path)

    print(f"\n✓ File loaded successfully!")
    print(f"✓ Total rows: {len(df)}")
    print(f"✓ Total columns: {len(df.columns)}")

    print(f"\n\nCOLUMN NAMES IN YOUR FILE:")
    print("-" * 70)
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    print(f"\n\nFIRST 5 ROWS:")
    print("-" * 70)
    print(df.head())

    print(f"\n\nDATA TYPES:")
    print("-" * 70)
    print(df.dtypes)

    print(f"\n\nMISSING VALUES:")
    print("-" * 70)
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} missing values")

    print(f"\n\nPRICE STATISTICS:")
    print("-" * 70)
    if 'price' in df.columns:
        print(df['price'].describe())
    else:
        print("⚠ No 'price' column found!")
        print("\nLooking for columns containing 'price':")
        for col in df.columns:
            if 'price' in col.lower():
                print(f"  Found: {col}")
                print(df[col].describe())

except FileNotFoundError:
    print(f"✗ File not found: {file_path}")
    print("\nPlease check:")
    print("  1. Is the file path correct?")
    print("  2. Does the file exist?")
except Exception as e:
    print(f"✗ Error: {e}")
