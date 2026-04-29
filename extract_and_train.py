"""
Extract ZIP file and train model
"""

import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

print("=" * 70)
print("EXTRACT & TRAIN MODEL")
print("=" * 70)

# ============================================================================
# EXTRACT ZIP FILE
# ============================================================================
print("\nSTEP 1: Extracting ZIP file...")

zip_path = r"C:\Users\surfe\Downloads\realtor-data.zip"
extract_path = r"C:\Users\surfe\Downloads\realtor_data"

try:
    # Create extraction directory
    os.makedirs(extract_path, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"✓ Extracted to: {extract_path}")

    # List extracted files
    extracted_files = os.listdir(extract_path)
    print(f"✓ Found {len(extracted_files)} files/folders:")
    for file in extracted_files[:10]:
        print(f"    - {file}")

except Exception as e:
    print(f"✗ Error extracting: {e}")
    exit()

# ============================================================================
# FIND CSV FILE
# ============================================================================
print("\nSTEP 2: Finding CSV file...")

csv_file = None

# Look for CSV files
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.csv'):
            csv_file = os.path.join(root, file)
            print(f"✓ Found: {file}")
            break
    if csv_file:
        break

if not csv_file:
    print("✗ No CSV file found in ZIP")
    print("\nPlease check:")
    print(f"  1. Open {extract_path} manually")
    print("  2. Find the CSV file")
    print("  3. Update csv_file variable below")
    exit()

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\nSTEP 3: Loading CSV file...")

try:
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    exit()

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 70)
print("DATA OVERVIEW")
print("=" * 70)

print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset shape: {len(df)} rows, {len(df.columns)} columns")

print(f"\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Check if columns exist
print(f"\nChecking required columns...")
required = ['price', 'bed', 'bath', 'house_size']
available = []

# Try different column name variations
column_mapping = {}
for col in df.columns:
    col_lower = col.lower().strip()
    if 'price' in col_lower and 'price' not in column_mapping:
        column_mapping['price'] = col
    elif 'bed' in col_lower and 'bed' not in column_mapping:
        column_mapping['bed'] = col
    elif 'bath' in col_lower and 'bath' not in column_mapping:
        column_mapping['bath'] = col
    elif 'size' in col_lower and 'house_size' not in column_mapping:
        column_mapping['house_size'] = col
    elif 'sqft' in col_lower and 'house_size' not in column_mapping:
        column_mapping['house_size'] = col

print(f"\nFound columns:")
for key, val in column_mapping.items():
    print(f"  {key}: {val}")

if len(column_mapping) < 4:
    print("\n⚠ Missing some required columns!")
    print("\nYour columns are:")
    print(df.columns.tolist())
    print("\nPlease tell me which columns correspond to:")
    print("  - Price")
    print("  - Bedrooms")
    print("  - Bathrooms")
    print("  - House size")
    exit()

# ============================================================================
# CLEAN DATA
# ============================================================================
print("\n" + "=" * 70)
print("CLEANING DATA")
print("=" * 70)

# Rename columns to standard names
df_clean = df.copy()
for key, col in column_mapping.items():
    df_clean[key] = df[col]

# Also add acre_lot if available
if 'acre_lot' not in column_mapping:
    for col in df.columns:
        if 'acre' in col.lower() or 'lot' in col.lower():
            df_clean['acre_lot'] = df[col]
            break
    if 'acre_lot' not in df_clean.columns:
        df_clean['acre_lot'] = 0.1  # Default value

# Keep only our columns
df_clean = df_clean[['price', 'bed', 'bath', 'house_size', 'acre_lot']]

initial_count = len(df_clean)

# Remove rows with missing values
df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'house_size'])

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Remove outliers (optional - remove extremely high/low prices)
Q1 = df_clean['price'].quantile(0.01)
Q3 = df_clean['price'].quantile(0.99)
df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]

removed = initial_count - len(df_clean)
print(f"Removed {removed} rows with missing/invalid data")
print(f"✓ Clean dataset: {len(df_clean)} houses")

print(f"\nPrice statistics:")
print(df_clean['price'].describe())

if len(df_clean) < 50:
    print("\n⚠ WARNING: You only have", len(df_clean), "houses")
    print("  For best accuracy, ideally need 100+ houses")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

X = df_clean[['bed', 'bath', 'house_size', 'acre_lot']].copy()
X['acre_lot'] = X['acre_lot'].fillna(X['acre_lot'].mean())
y = df_clean['price'].copy()

print(f"Features: bed, bath, house_size, acre_lot")
print(f"Target: price")
print(f"Total examples: {len(X)}")

# ============================================================================
# SPLIT DATA
# ============================================================================
print("\n" + "=" * 70)
print("SPLITTING DATA")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} houses (80%)")
print(f"Testing set: {len(X_test)} houses (20%)")

# ============================================================================
# SCALE DATA
# ============================================================================
print("\n" + "=" * 70)
print("SCALING DATA")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data scaled")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

print("Training Random Forest with 100 trees...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("✓ Model trained!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTraining Set:")
print(f"  R² Score: {r2_train:.4f}")
print(f"  MAE: ${mae_train:,.0f}")

print(f"\nTesting Set (More important):")
print(f"  R² Score: {r2_test:.4f}")
print(f"  MAE: ${mae_test:,.0f}")
print(f"  RMSE: ${rmse_test:,.0f}")

print(f"\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if r2_test >= 0.8:
    print(f"✓ EXCELLENT! Model is {r2_test*100:.1f}% accurate")
    print("  Use this model with confidence")
elif r2_test >= 0.7:
    print(f"✓ VERY GOOD! Model is {r2_test*100:.1f}% accurate")
    print("  Model is reliable")
elif r2_test >= 0.6:
    print(f"✓ GOOD! Model is {r2_test*100:.1f}% accurate")
    print("  Reasonably reliable")
elif r2_test >= 0.5:
    print(f"~ OKAY! Model is {r2_test*100:.1f}% accurate")
    print("  Use as rough guide")
else:
    print(f"✗ POOR! Model is {r2_test*100:.1f}% accurate")
    print("  Need more data or different features")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (What Matters Most)")
print("=" * 70)

features = ['Bedrooms', 'Bathrooms', 'House Size', 'Acre Lot']
importances = model.feature_importances_

for feat, imp in zip(features, importances):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15} {bar} {imp*100:5.1f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'house_scaler.pkl')

print("✓ Saved: house_model.pkl")
print("✓ Saved: house_scaler.pkl")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, color='blue')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
ax.set_xlabel('Actual Price ($)', fontsize=11)
ax.set_ylabel('Predicted Price ($)', fontsize=11)
ax.set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Prediction Errors
ax = axes[0, 1]
errors = y_test - y_pred_test
ax.scatter(y_pred_test, errors, alpha=0.6, s=50, color='green')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Price ($)', fontsize=11)
ax.set_ylabel('Error ($)', fontsize=11)
ax.set_title('Prediction Errors', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Error Distribution
ax = axes[1, 0]
ax.hist(errors, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
ax.set_xlabel('Error ($)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature Importance
ax = axes[1, 1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax.barh(features, importances, color=colors)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
for i, (feat, imp) in enumerate(zip(features, importances)):
    ax.text(imp, i, f' {imp*100:.1f}%', va='center', fontweight='bold')
ax.set_xlim(0, max(importances) * 1.2)

plt.tight_layout()
plt.savefig('model_results.png', dpi=100, bbox_inches='tight')
print("✓ Saved: model_results.png")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 70)
print("✓ COMPLETE!")
print("=" * 70)
print("\nYour model is trained and ready!")
print("\nNext: Run python use_model.py to make predictions")
