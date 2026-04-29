"""
House Price Prediction - Supervised ML Tutorial
Learn to build a model that predicts house prices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# ============================================================================
# STEP 1: PREPARE YOUR DATA
# ============================================================================

# Sample house data
house_data = {
    'size_sqft': [1500, 2000, 1200, 3000, 1800, 2500, 1100, 2200, 1900, 3500,
                  1600, 2100, 1400, 2800, 2000, 1700, 2300, 1300, 2600, 1900,
                  1550, 2050, 1250, 2950, 1850, 2450, 1150, 2250, 1950, 3450],
    'bedrooms': [3, 4, 2, 5, 3, 4, 2, 4, 3, 5,
                 3, 4, 2, 4, 3, 3, 4, 2, 4, 3,
                 3, 4, 2, 5, 3, 4, 2, 4, 3, 5],
    'bathrooms': [2, 2.5, 1, 3, 2, 2.5, 1, 2.5, 2, 3,
                  2, 2.5, 1, 3, 2, 2, 2.5, 1, 3, 2,
                  2, 2.5, 1, 3, 2, 2.5, 1, 2.5, 2, 3],
    'age_years': [10, 5, 20, 3, 8, 6, 25, 4, 7, 2,
                  9, 5, 22, 4, 8, 10, 6, 23, 5, 9,
                  11, 4, 21, 2, 9, 7, 24, 3, 8, 1],
    'purchase_price': [250000, 380000, 180000, 550000, 300000, 420000, 150000, 350000, 310000, 650000,
                       280000, 390000, 170000, 480000, 320000, 290000, 400000, 160000, 460000, 330000,
                       260000, 375000, 190000, 560000, 305000, 410000, 155000, 360000, 315000, 640000]
}

# Create DataFrame
df = pd.DataFrame(house_data)
print("=" * 70)
print("STEP 1: YOUR DATA")
print("=" * 70)
print(df.head(10))
print(f"\nTotal houses: {len(df)}")
print(f"\nData summary:")
print(df.describe())

# ============================================================================
# STEP 2: LOAD YOUR OWN DATA (Optional)
# ============================================================================
# Uncomment this if you have a CSV file:
# df = pd.read_csv('your_houses.csv')

# ============================================================================
# STEP 3: DATA CLEANING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")
df = df.dropna()

# Remove duplicates
print(f"Duplicate rows before removal: {df.duplicated().sum()}")
df = df.drop_duplicates()

print(f"✓ Clean dataset shape: {df.shape}")

# ============================================================================
# STEP 4: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: SEPARATE FEATURES AND TARGET")
print("=" * 70)

# Features (X): what we use to predict
# Target (y): what we want to predict (price)
X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
y = df['purchase_price']

print("Features (X):")
print(X.head())
print("\nTarget (y) - Prices:")
print(y.head())
print(f"\nPrice range: ${y.min():,.0f} - ${y.max():,.0f}")

# ============================================================================
# STEP 5: SPLIT INTO TRAINING AND TESTING DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: SPLIT DATA (Training vs Testing)")
print("=" * 70)

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} houses (80%)")
print(f"Testing set: {len(X_test)} houses (20%)")
print(f"\nWhy split? To test if model works on new data it hasn't seen")

# ============================================================================
# STEP 6: SCALE THE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: NORMALIZE DATA")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data normalized (0 mean, 1 standard deviation)")
print(f"  Before: size ranges from {X['size_sqft'].min()} to {X['size_sqft'].max()}")
print(f"  After scaling: mean ≈ 0, std ≈ 1")

# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAIN PREDICTION MODEL")
print("=" * 70)

# Random Forest is great for this - handles non-linear relationships
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

print("✓ Model trained!")
print(f"\nModel type: Random Forest")
print("Why? Works well with real estate prices, handles complex patterns")

# Show feature importance
feature_names = ['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)']
importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance*100:.1f}% importance")

# ============================================================================
# STEP 8: TEST THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: EVALUATE MODEL PERFORMANCE")
print("=" * 70)

# Make predictions on test set
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate accuracy metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTraining Set Performance:")
print(f"  R² Score: {train_r2:.4f} (1.0 = perfect)")
print(f"  MAE: ${train_mae:,.0f} (avg error)")

print(f"\nTesting Set Performance:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE: ${test_mae:,.0f}")
print(f"  RMSE: ${test_rmse:,.0f}")

if test_r2 > 0.85:
    print("\n✓ EXCELLENT! Model predicts prices very well")
elif test_r2 > 0.70:
    print("\n✓ GOOD! Model is reasonably accurate")
else:
    print("\n⚠ Model needs more data or tuning")

# ============================================================================
# STEP 9: PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: PREDICT PRICE FOR NEW HOUSE")
print("=" * 70)

def predict_house_price(size_sqft, bedrooms, bathrooms, age_years):
    """
    Predict the price of a house based on features

    Args:
        size_sqft: square footage
        bedrooms: number of bedrooms
        bathrooms: number of bathrooms
        age_years: age in years

    Returns:
        dict with prediction and confidence info
    """

    # Create feature array
    new_house = np.array([[size_sqft, bedrooms, bathrooms, age_years]])

    # Scale using the fitted scaler
    new_house_scaled = scaler.transform(new_house)

    # Make prediction
    predicted_price = model.predict(new_house_scaled)[0]

    # Calculate prediction interval (rough estimate)
    std_error = np.std(y_test - model.predict(X_test_scaled))
    confidence_interval = 1.96 * std_error  # 95% confidence

    return {
        'predicted_price': round(predicted_price, 2),
        'lower_estimate': round(predicted_price - confidence_interval, 2),
        'upper_estimate': round(predicted_price + confidence_interval, 2),
        'price_per_sqft': round(predicted_price / size_sqft, 2),
        'confidence_interval': round(confidence_interval, 2)
    }

# ============================================================================
# STEP 10: TEST WITH EXAMPLE HOUSES
# ============================================================================
print("\n--- TEST PREDICTIONS ---")

test_cases = [
    {'size': 2000, 'bed': 4, 'bath': 2.5, 'age': 5, 'name': 'Modern 4BR house'},
    {'size': 1200, 'bed': 2, 'bath': 1, 'age': 20, 'name': 'Old small house'},
    {'size': 3200, 'bed': 5, 'bath': 3, 'age': 2, 'name': 'Brand new luxury'},
]

for test in test_cases:
    result = predict_house_price(test['size'], test['bed'], test['bath'], test['age'])
    print(f"\n{test['name']}:")
    print(f"  Features: {test['size']} sqft | {test['bed']} bed | {test['bath']} bath | {test['age']} yrs")
    print(f"  → Predicted price: ${result['predicted_price']:,.0f}")
    print(f"  → Estimated range: ${result['lower_estimate']:,.0f} - ${result['upper_estimate']:,.0f}")
    print(f"  → Price per sqft: ${result['price_per_sqft']:.2f}")

# ============================================================================
# STEP 11: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: VISUALIZE MODEL")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted (Test Set)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=80)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)', fontsize=11)
ax.set_ylabel('Predicted Price ($)', fontsize=11)
ax.set_title('Actual vs Predicted Prices', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals (Prediction Errors)
ax = axes[0, 1]
residuals = y_test - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.6, s=80)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Price ($)', fontsize=11)
ax.set_ylabel('Residual (Error) ($)', fontsize=11)
ax.set_title('Prediction Errors', fontsize=12)
ax.grid(True, alpha=0.3)

# Feature Importance
ax = axes[1, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.barh(feature_names, model.feature_importances_, color=colors)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Feature Importance in Price Prediction', fontsize=12)
ax.set_xlim(0, max(model.feature_importances_) * 1.2)
for i, (name, imp) in enumerate(zip(feature_names, model.feature_importances_)):
    ax.text(imp, i, f' {imp*100:.1f}%', va='center')

# Size vs Price (with predictions)
ax = axes[1, 1]
ax.scatter(X_test['size_sqft'], y_test, label='Actual prices', alpha=0.6, s=80)
ax.scatter(X_test['size_sqft'], y_pred_test, label='Predictions', alpha=0.6, s=80, marker='x')
ax.set_xlabel('Size (sqft)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Size vs Price (Actual vs Predicted)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=100, bbox_inches='tight')
print("✓ Saved: model_evaluation.png")
plt.close()

# ============================================================================
# STEP 12: SAVE THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: SAVE YOUR TRAINED MODEL")
print("=" * 70)

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✓ Saved: house_price_model.pkl")
print("✓ Saved: scaler.pkl")
print("\nNow you can load these and predict prices on new houses!")

# ============================================================================
# BONUS: HOW GOOD IS MY MODEL?
# ============================================================================
print("\n" + "=" * 70)
print("UNDERSTANDING MODEL QUALITY")
print("=" * 70)
print(f"\nR² Score = {test_r2:.4f}")
print("What does this mean?")
print(f"  - The model explains {test_r2*100:.1f}% of the price variation")
print(f"  - 0.0 = useless, 1.0 = perfect, >0.7 = good")

print(f"\nAverage Error = ${test_mae:,.0f}")
print("What does this mean?")
print(f"  - On average, predictions are off by ${test_mae:,.0f}")
print(f"  - For houses costing ~${y_test.mean():,.0f}, that's {test_mae/y_test.mean()*100:.1f}% error")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Replace sample data with YOUR real house data")
print("2. Run this script to train a model on your data")
print("3. Use predict_house_price() to evaluate new houses")
print("4. Check model_evaluation.png to see how well it works")
