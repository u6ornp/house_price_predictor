"""
Diagnose Problems & Find Solutions
Analyze your data to see why R² is low
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("DIAGNOSING LOW R² PROBLEM")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
file_path = r"C:\Users\surfe\Downloads\realtor-data\realtor-data.csv"

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# Auto-detect columns
col_price = None
col_bed = None
col_bath = None
col_size = None
col_acre = None

for col in df.columns:
    col_lower = col.lower().strip()
    if 'price' in col_lower and not col_price:
        col_price = col
    elif 'bed' in col_lower and not col_bed:
        col_bed = col
    elif 'bath' in col_lower and not col_bath:
        col_bath = col
    elif ('size' in col_lower or 'sqft' in col_lower) and not col_size:
        col_size = col
    elif ('acre' in col_lower or 'lot' in col_lower) and not col_acre:
        col_acre = col

# Prepare data
df_clean = df[[col_price, col_bed, col_bath, col_size]].copy()
if col_acre:
    df_clean[col_acre] = df[col_acre]
else:
    df_clean['acre_lot'] = 0.1

df_clean.columns = ['price', 'bed', 'bath', 'house_size', 'acre_lot']
df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'house_size'])
df_clean = df_clean.drop_duplicates()

print(f"\n✓ Loaded {len(df_clean)} houses")

# ============================================================================
# DIAGNOSIS 1: DATA QUALITY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS 1: DATA QUALITY")
print("=" * 70)

print(f"\nDataset size: {len(df_clean)} houses")
if len(df_clean) < 50:
    print("⚠️  TOO SMALL! You need at least 100+ houses")
    print("   With only", len(df_clean), "houses, ML can't learn patterns")
elif len(df_clean) < 100:
    print("⚠️  Small dataset. Would be better with 100+")
else:
    print("✓ Good dataset size")

# ============================================================================
# DIAGNOSIS 2: PRICE VARIATION
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS 2: PRICE VARIATION")
print("=" * 70)

print(f"\nPrice statistics:")
print(f"  Min: ${df_clean['price'].min():,.0f}")
print(f"  Max: ${df_clean['price'].max():,.0f}")
print(f"  Mean: ${df_clean['price'].mean():,.0f}")
print(f"  Std Dev: ${df_clean['price'].std():,.0f}")

price_range = df_clean['price'].max() - df_clean['price'].min()
price_cv = df_clean['price'].std() / df_clean['price'].mean()

print(f"\nPrice range: ${price_range:,.0f}")
print(f"Price coefficient of variation: {price_cv:.2f}")

if price_cv > 1.0:
    print("⚠️  HUGE PRICE VARIATION!")
    print("   Similar houses have very different prices")
    print("   This confuses the model")
else:
    print("✓ Price variation is reasonable")

# ============================================================================
# DIAGNOSIS 3: FEATURE RELATIONSHIPS
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS 3: DO FEATURES PREDICT PRICE?")
print("=" * 70)

# Calculate correlation
from scipy.stats import pearsonr

correlations = {}
for col in ['bed', 'bath', 'house_size', 'acre_lot']:
    if col in df_clean.columns:
        corr, p_value = pearsonr(df_clean[col], df_clean['price'])
        correlations[col] = corr
        strength = "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"  {col:12} correlation: {corr:7.3f} ({strength})")

if all(abs(c) < 0.3 for c in correlations.values()):
    print("\n⚠️  PROBLEM FOUND!")
    print("   Features have WEAK correlation with price")
    print("   Your features don't explain price variation!")
else:
    print("\n✓ Features have reasonable correlation with price")

# ============================================================================
# DIAGNOSIS 4: OUTLIERS
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS 4: OUTLIERS & ANOMALIES")
print("=" * 70)

# Find price outliers
Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['price'] < lower_bound) | (df_clean['price'] > upper_bound)]

print(f"\nPrice outliers (unusual prices):")
print(f"  Found: {len(outliers)} houses ({len(outliers)/len(df_clean)*100:.1f}%)")

if len(outliers) / len(df_clean) > 0.2:
    print("⚠️  Many outliers! This confuses the model")
    print("   Consider removing extreme prices")
else:
    print("✓ Reasonable number of outliers")

# ============================================================================
# DIAGNOSIS 5: MISSING FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS 5: MISSING IMPORTANT FEATURES")
print("=" * 70)

print("\nYour current features:")
print("  ✓ Bedrooms")
print("  ✓ Bathrooms")
print("  ✓ House Size")
print("  ✓ Acre Lot")

print("\nMissing features that would help:")
print("  ✗ Location/City (VERY important!)")
print("  ✗ House condition")
print("  ✗ Year built")
print("  ✗ Number of garages")
print("  ✗ Pool/No pool")
print("  ✗ Renovated/Not renovated")
print("  ✗ Property type")

print("\n⚠️  MAJOR ISSUE: You're missing LOCATION!")
print("   Location is often THE most important factor")
print("   Two identical houses in different cities cost differently")

# ============================================================================
# SOLUTIONS
# ============================================================================
print("\n" + "=" * 70)
print("SOLUTIONS TO IMPROVE R²")
print("=" * 70)

print("\n1️⃣  COLLECT MORE DATA")
print("   Current: " + len(df_clean), "houses")
print("   Target: 100-500+ houses")
print("   Impact: R² could improve to 0.5-0.7")

print("\n2️⃣  ADD LOCATION/CITY (CRITICAL!)")
print("   Check your CSV for columns like:")
print("      - city")
print("      - zip_code")
print("      - neighborhood")
print("      - state")
print("   Impact: R² could jump to 0.4-0.6+")

print("\n3️⃣  ADD HOUSE CONDITION")
print("   If available, add:")
print("      - condition (excellent/good/fair/poor)")
print("      - year_built")
print("   Impact: R² could improve to 0.5-0.7")

print("\n4️⃣  REMOVE PRICE OUTLIERS")
print("   Remove extreme prices (very cheap/very expensive)")
print("   This helps the model focus on typical houses")
print("   Impact: R² could improve by 0.1-0.2")

print("\n5️⃣  CHECK DATA QUALITY")
print("   Make sure:")
print("      - No incorrect data")
print("      - Realistic values")
print("      - No typos in prices")
print("   Impact: R² could improve by 0.05-0.15")

# ============================================================================
# RECOMMENDED NEXT STEPS
# ============================================================================
print("\n" + "=" * 70)
print("RECOMMENDED NEXT STEPS")
print("=" * 70)

print("\n📋 YOUR IMMEDIATE ACTIONS:")
print("\n1. Check your CSV for these columns:")

print("\n   Columns in your file:")
for i, col in enumerate(df.columns[:15]):  # Show first 15 columns
    print(f"      {i+1}. {col}")

print("\n2. Tell me which column is:")
print("      - Location/City (if available)")
print("      - House condition (if available)")
print("      - Year built (if available)")

print("\n3. Or give me these stats:")
print("      - How many unique cities/locations?")
print("      - Do you have condition data?")
print("      - Do you have year built?")

# ============================================================================
# CREATE DIAGNOSTIC CHARTS
# ============================================================================
print("\nCreating diagnostic charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price distribution
ax = axes[0, 0]
ax.hist(df_clean['price'], bins=20, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(df_clean['price'].mean(), color='r', linestyle='--', lw=2, label=f'Mean: ${df_clean["price"].mean():,.0f}')
ax.axvline(df_clean['price'].median(), color='g', linestyle='--', lw=2, label=f'Median: ${df_clean["price"].median():,.0f}')
ax.set_xlabel('Price ($)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Size vs Price
ax = axes[0, 1]
ax.scatter(df_clean['house_size'], df_clean['price'], alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
ax.set_xlabel('House Size (sqft)', fontsize=11, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax.set_title('House Size vs Price', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Beds vs Price
ax = axes[1, 0]
for bed in sorted(df_clean['bed'].unique()):
    prices = df_clean[df_clean['bed'] == bed]['price']
    ax.scatter([bed] * len(prices), prices, alpha=0.6, s=50, label=f'{int(bed)} bed')
ax.set_xlabel('Bedrooms', fontsize=11, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
ax.set_title('Bedrooms vs Price', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Correlation heatmap
ax = axes[1, 1]
corr_data = []
corr_labels = []
for col, corr in correlations.items():
    corr_data.append(abs(corr))
    corr_labels.append(col)

colors = ['red' if c < 0.3 else 'orange' if c < 0.6 else 'green' for c in corr_data]
bars = ax.barh(corr_labels, corr_data, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Correlation Strength', fontsize=11, fontweight='bold')
ax.set_title('Feature-Price Correlation', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
for i, (label, val) in enumerate(zip(corr_labels, corr_data)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('diagnostic_report.png', dpi=100, bbox_inches='tight')
print("✓ Saved: diagnostic_report.png")

print("\n" + "=" * 70)
print("✓ Diagnostic complete!")
print("=" * 70)
print("\nCheck diagnostic_report.png to see the analysis visually")
