"""
Check Model Accuracy
See how well the model predicts prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=" * 70)
print("MODEL ACCURACY CHECK")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nLoading data...")

data = """brokered_by,status,price,bed,bath,acre_lot,street,city,state,zip_code,house_size,prev_sold_date
103378.0,for_sale,105000.0,3,2,0.12,1962661.0,Adjuntas,Puerto Rico,00601,920.0,
52707.0,for_sale,80000.0,4,2,0.08,1902874.0,Adjuntas,Puerto Rico,00601,1527.0,
103379.0,for_sale,67000.0,2,1,0.15,1404990.0,Juana Diaz,Puerto Rico,00795,748.0,
31239.0,for_sale,145000.0,4,2,0.1,1947675.0,Ponce,Puerto Rico,00731,1800.0,
34632.0,for_sale,65000.0,6,2,0.05,331151.0,Mayaguez,Puerto Rico,00680,,
103378.0,for_sale,179000.0,4,3,0.46,1850806.0,San Sebastian,Puerto Rico,00612,2520.0,
1205.0,for_sale,50000.0,3,1,0.2,1298094.0,Ciales,Puerto Rico,00639,2040.0,
50739.0,for_sale,71600.0,3,2,0.08,1048466.0,Ponce,Puerto Rico,00731,1050.0,
81909.0,for_sale,100000.0,2,1,0.09,734904.0,Ponce,Puerto Rico,00730,1092.0,
65672.0,for_sale,300000.0,5,3,7.46,1946226.0,Las Marias,Puerto Rico,00670,5403.0,
52707.0,for_sale,89000.0,3,2,13.39,1902814.0,Isabela,Puerto Rico,00662,1106.0,
52707.0,for_sale,150000.0,3,2,0.08,1773902.0,Juana Diaz,Puerto Rico,00795,1045.0,
46019.0,for_sale,155000.0,3,2,0.1,1946165.0,Lares,Puerto Rico,00669,4161.0,
52707.0,for_sale,79000.0,5,2,0.12,1761024.0,Utuado,Puerto Rico,00641,1620.0,
88441.0,for_sale,649000.0,5,5,0.74,1879215.0,Ponce,Puerto Rico,00731,2677.0,
50739.0,for_sale,120000.0,3,2,0.08,17854.0,Yauco,Puerto Rico,00698,1100.0,
51202.0,for_sale,235000.0,4,4,0.22,13687.0,Mayaguez,Puerto Rico,00680,3450.0,
12876.0,for_sale,105000.0,3,2,0.08,1868721.0,Ponce,Puerto Rico,00728,1500.0,
109906.0,for_sale,575000.0,3,2,3.88,1312671.0,San Sebastian,Puerto Rico,00685,4000.0,
46019.0,for_sale,140000.0,6,3,0.25,6710.0,Anasco,Puerto Rico,00610,1230.0,
52707.0,for_sale,50000.0,2,1,0.23,1902835.0,Yauco,Puerto Rico,00698,621.0,
52707.0,for_sale,165000.0,6,3,0.1,117231.0,Moca,Puerto Rico,00676,3000.0,
81909.0,for_sale,189000.0,3,1,2.0,1307740.0,Coamo,Puerto Rico,00769,1213.0,
52707.0,for_sale,115000.0,3,2,,481889.0,Ponce,Puerto Rico,00716,1148.0,
12434.0,for_sale,122500.0,3,2,0.05,437877.0,Yauco,Puerto Rico,00698,1118.0,
52464.0,for_sale,255000.0,3,2,0.28,1948621.0,San Sebastian,Puerto Rico,00685,1500.0,
81495.0,for_sale,425000.0,4,3,0.3,955013.0,Ponce,Puerto Rico,00730,3000.0,
87549.0,for_sale,93000.0,4,2,0.11,1854860.0,Guayanilla,Puerto Rico,00656,1300.0,"""

with open('houses.csv', 'w') as f:
    f.write(data)

df = pd.read_csv('houses.csv')
df = df.dropna(subset=['price', 'bed', 'bath', 'house_size'])

print(f"✓ Loaded {len(df)} houses")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\nPreparing data...")

X = df[['bed', 'bath', 'house_size', 'acre_lot']].copy()
X['acre_lot'] = X['acre_lot'].fillna(X['acre_lot'].mean())
y = df['price'].copy()

# ============================================================================
# STEP 3: SPLIT DATA (80% train, 20% test)
# ============================================================================
print("\nSplitting data...")
print("  80% for training (model learns from these)")
print("  20% for testing (check if it works on NEW data)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training: {len(X_train)} houses")
print(f"  Testing: {len(X_test)} houses")

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
print("\nTraining model...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

print("✓ Model trained!")

# ============================================================================
# STEP 5: CHECK ACCURACY
# ============================================================================
print("\n" + "=" * 70)
print("ACCURACY METRICS")
print("=" * 70)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n1. R² SCORE (Coefficient of Determination)")
print("   " + "-" * 60)
print(f"   Training R²: {train_r2:.4f}")
print(f"   Testing R²:  {test_r2:.4f}")
print("\n   What this means:")
print("   - Range: 0 to 1 (higher is better)")
print("   - 1.0 = Perfect predictions")
print("   - 0.8+ = EXCELLENT")
print("   - 0.6+ = GOOD")
print("   - 0.4+ = OKAY")
print("   - <0.4 = POOR")

if test_r2 >= 0.8:
    print(f"\n   ✓ Your model is EXCELLENT ({test_r2:.1%} accurate)")
elif test_r2 >= 0.6:
    print(f"\n   ✓ Your model is GOOD ({test_r2:.1%} accurate)")
elif test_r2 >= 0.4:
    print(f"\n   ~ Your model is OKAY ({test_r2:.1%} accurate)")
else:
    print(f"\n   ✗ Your model needs improvement ({test_r2:.1%} accurate)")

print("\n2. MAE (Mean Absolute Error)")
print("   " + "-" * 60)
print(f"   Training MAE: ${train_mae:,.0f}")
print(f"   Testing MAE:  ${test_mae:,.0f}")
print("\n   What this means:")
print(f"   - On average, predictions are off by ${test_mae:,.0f}")
print(f"   - If house costs $100,000, prediction might be")
print(f"     between ${100000-test_mae:,.0f} and ${100000+test_mae:,.0f}")

error_percentage = (test_mae / y_test.mean()) * 100
print(f"\n   - As percentage: {error_percentage:.1f}% error")

print("\n3. RMSE (Root Mean Squared Error)")
print("   " + "-" * 60)
print(f"   Testing RMSE: ${test_rmse:,.0f}")
print("\n   What this means:")
print("   - Similar to MAE but penalizes bigger errors more")
print("   - RMSE > MAE = some predictions are very wrong")
print("   - RMSE ≈ MAE = errors are consistent")

# ============================================================================
# STEP 6: DETAILED ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED ERROR ANALYSIS (What the Model Gets Wrong)")
print("=" * 70)

errors = y_test - y_pred_test
abs_errors = np.abs(errors)

print(f"\nBiggest Errors (Testing Set):")
print("-" * 60)

# Find biggest errors
error_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_test,
    'error': errors.values,
    'error_pct': (errors.values / y_test.values) * 100
})
error_df = error_df.sort_values('error', key=abs)

for idx, row in error_df.tail(3).iterrows():
    print(f"\n  Actual: ${row['actual']:,.0f}")
    print(f"  Predicted: ${row['predicted']:,.0f}")
    print(f"  Error: ${row['error']:+,.0f} ({row['error_pct']:+.1f}%)")

print("\n\nSmallest Errors (Testing Set):")
print("-" * 60)

for idx, row in error_df.head(3).iterrows():
    print(f"\n  Actual: ${row['actual']:,.0f}")
    print(f"  Predicted: ${row['predicted']:,.0f}")
    print(f"  Error: ${row['error']:+,.0f} ({row['error_pct']:+.1f}%)")

# ============================================================================
# STEP 7: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=80, color='blue')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)', fontsize=11)
ax.set_ylabel('Predicted Price ($)', fontsize=11)
ax.set_title('Actual vs Predicted Prices (Test Set)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Prediction Errors
ax = axes[0, 1]
ax.scatter(y_pred_test, errors.values, alpha=0.6, s=80, color='green')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Price ($)', fontsize=11)
ax.set_ylabel('Error ($)', fontsize=11)
ax.set_title('Prediction Errors', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Error Distribution
ax = axes[1, 0]
ax.hist(errors.values, bins=8, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
ax.set_xlabel('Error ($)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature Importance
ax = axes[1, 1]
feature_names = ['Bedrooms', 'Bathrooms', 'House Size', 'Acre Lot']
importances = model.feature_importances_
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax.barh(feature_names, importances, color=colors)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Feature Importance (What Matters Most)', fontsize=12, fontweight='bold')
for i, (name, imp) in enumerate(zip(feature_names, importances)):
    ax.text(imp, i, f' {imp*100:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_accuracy.png', dpi=100, bbox_inches='tight')
print("✓ Saved: model_accuracy.png")
print("\nOpen this file to see visualizations!")

# ============================================================================
# STEP 8: FINAL VERDICT
# ============================================================================
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

print(f"\nModel Performance Summary:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  Average Error: ${test_mae:,.0f}")
print(f"  Error %: {error_percentage:.1f}%")

print(f"\n{'VERDICT:':^70}")
if test_r2 >= 0.8:
    print(f"{'✓ MODEL IS RELIABLE FOR DECISIONS':^70}")
    print(f"{'Use this model with confidence':^70}")
elif test_r2 >= 0.6:
    print(f"{'✓ MODEL IS REASONABLY GOOD':^70}")
    print(f"{'Use this model but verify important decisions':^70}")
elif test_r2 >= 0.4:
    print(f"{'~ MODEL IS OKAY':^70}")
    print(f"{'Use this model as a rough guide only':^70}")
else:
    print(f"{'✗ MODEL NEEDS MORE DATA':^70}")
    print(f"{'Collect more houses before relying on predictions':^70}")

print(f"\nRecommendation:")
if len(df) < 20:
    print(f"  ⚠ You only have {len(df)} houses. Collect 30-50 for better accuracy!")
else:
    print(f"  ✓ You have {len(df)} houses. Good dataset size!")

print("\n" + "=" * 70)
