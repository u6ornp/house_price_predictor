"""
Flexible Accuracy Check - Auto-detects column names
Calculates R² value and model accuracy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

print("=" * 70)
print("ACCURACY CHECK - AUTO-DETECT COLUMNS")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

file_path = r"C:\Users\surfe\Downloads\realtor-data\realtor-data.csv"

try:
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit()

# ============================================================================
# AUTO-DETECT COLUMN NAMES
# ============================================================================
print("\n" + "=" * 70)
print("AUTO-DETECTING COLUMN NAMES")
print("=" * 70)

print(f"\nYour columns: {list(df.columns)}")

# Find column names (case-insensitive)
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

print("\nDetected columns:")
print(f"  Price: {col_price}")
print(f"  Bedrooms: {col_bed}")
print(f"  Bathrooms: {col_bath}")
print(f"  House Size: {col_size}")
print(f"  Acre/Lot: {col_acre}")

# Check if we found all required columns
if not all([col_price, col_bed, col_bath, col_size]):
    print("\n✗ Could not auto-detect all required columns!")
    print("\nManual mapping needed:")
    print("Please edit the script and change these lines:")
    print(f"  col_price = '{col_price or 'CHANGE_ME'}'")
    print(f"  col_bed = '{col_bed or 'CHANGE_ME'}'")
    print(f"  col_bath = '{col_bath or 'CHANGE_ME'}'")
    print(f"  col_size = '{col_size or 'CHANGE_ME'}'")
    print(f"  col_acre = '{col_acre or 'DEFAULT_0.1'}'")
    exit()

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

df_clean = df[[col_price, col_bed, col_bath, col_size]].copy()
if col_acre:
    df_clean[col_acre] = df[col_acre]
else:
    df_clean['acre_lot'] = 0.1

# Rename to standard names
df_clean.columns = ['price', 'bed', 'bath', 'house_size', 'acre_lot']

print(f"Before cleaning: {len(df_clean)} rows")

# Remove missing values
df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'house_size'])

# Remove duplicates
df_clean = df_clean.drop_duplicates()

print(f"After cleaning: {len(df_clean)} rows")

if len(df_clean) < 20:
    print(f"\n⚠ WARNING: Only {len(df_clean)} houses!")
    print("  Need at least 20-30 for meaningful accuracy")

print(f"\nPrice range: ${df_clean['price'].min():,.0f} - ${df_clean['price'].max():,.0f}")
print(f"Average price: ${df_clean['price'].mean():,.0f}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE PREPARATION")
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

if len(X) < 10:
    test_size = 0.2
    print(f"⚠ Small dataset: using 80/20 split")
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42
)

print(f"Training: {len(X_train)} houses")
print(f"Testing: {len(X_test)} houses")

# ============================================================================
# SCALE DATA
# ============================================================================
print("\nScaling data...")

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

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("✓ Model trained!")

# ============================================================================
# CALCULATE ACCURACY (R² VALUE)
# ============================================================================
print("\n" + "=" * 70)
print("ACCURACY RESULTS")
print("=" * 70)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n╔════════════════════════════════════════════════════════════════════╗")
print("║                         R² VALUE (ACCURACY)                        ║")
print("╚════════════════════════════════════════════════════════════════════╝")

print(f"\nTraining Set R²: {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"Testing Set R²:  {r2_test:.4f} ({r2_test*100:.2f}%)")

print("\n" + "─" * 70)
print("WHAT R² MEANS:")
print("─" * 70)
print("  R² = 1.0   → Perfect predictions (100% accurate)")
print("  R² = 0.9   → Excellent (90% accurate)")
print("  R² = 0.8   → Very Good (80% accurate)")
print("  R² = 0.7   → Good (70% accurate)")
print("  R² = 0.6   → Decent (60% accurate)")
print("  R² = 0.5   → Okay (50% accurate)")
print("  R² = 0.0   → Useless (0% accurate)")
print("  R² < 0.0   → Worse than random guess")

print("\n" + "─" * 70)
print("YOUR MODEL:")
print("─" * 70)

if r2_test >= 0.9:
    verdict = "✓✓✓ EXCELLENT"
    explanation = "Extremely accurate. Use with full confidence."
elif r2_test >= 0.8:
    verdict = "✓✓ VERY GOOD"
    explanation = "Highly reliable. Use with confidence."
elif r2_test >= 0.7:
    verdict = "✓ GOOD"
    explanation = "Reasonably accurate. Can rely on predictions."
elif r2_test >= 0.6:
    verdict = "~ DECENT"
    explanation = "Moderate accuracy. Use as a guide."
elif r2_test >= 0.5:
    verdict = "~ OKAY"
    explanation = "Weak but better than nothing."
elif r2_test >= 0.4:
    verdict = "✗ POOR"
    explanation = "Not very reliable. Consider collecting more data."
else:
    verdict = "✗✗ VERY POOR"
    explanation = "Not reliable. Need more data or different approach."

print(f"\n{verdict}")
print(f"{explanation}")

print("\n" + "─" * 70)
print("PREDICTION ERROR:")
print("─" * 70)
print(f"Average Error (MAE): ${mae_test:,.0f}")
print(f"Error as percentage: {(mae_test/y_test.mean())*100:.1f}%")
print(f"Root Mean Squared Error: ${rmse_test:,.0f}")

print("\nWhat this means:")
print(f"  If house costs $100,000")
print(f"  Prediction will be ~${100000-mae_test:,.0f} to ${100000+mae_test:,.0f}")

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
    pct = imp * 100
    print(f"  {feat:15} {bar} {pct:5.1f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'house_scaler.pkl')

print("✓ Model saved: house_model.pkl")
print("✓ Scaler saved: house_scaler.pkl")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=80, color='blue', edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax.set_title(f'Actual vs Predicted (R² = {r2_test:.4f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Errors
ax = axes[0, 1]
errors = y_test - y_pred_test
ax.scatter(y_pred_test, errors, alpha=0.6, s=80, color='green', edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero error')
ax.set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
ax.set_ylabel('Error ($)', fontsize=11, fontweight='bold')
ax.set_title(f'Prediction Errors (MAE = ${mae_test:,.0f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[1, 0]
ax.hist(errors, bins=10, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
ax.axvline(x=errors.mean(), color='orange', linestyle='--', lw=2, label=f'Mean error: ${errors.mean():,.0f}')
ax.set_xlabel('Error ($)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Feature importance
ax = axes[1, 1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax.barh(features, importances, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
for i, (feat, imp) in enumerate(zip(features, importances)):
    ax.text(imp + 0.01, i, f'{imp*100:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_results.png', dpi=100, bbox_inches='tight')
print("✓ Saved: accuracy_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n📊 Dataset Size: {len(df_clean)} houses")
print(f"📈 R² Score: {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"💰 Average Error: ${mae_test:,.0f}")
print(f"🎯 Verdict: {verdict}")

if r2_test < 0.6:
    print(f"\n💡 Tips to improve accuracy:")
    print(f"   1. Collect more houses (currently {len(df_clean)}, need 100+)")
    print(f"   2. Add more features (location, condition, year built, etc.)")
    print(f"   3. Remove price outliers (very cheap or very expensive houses)")

print("\n✓ Complete! Check accuracy_results.png for visualizations")
