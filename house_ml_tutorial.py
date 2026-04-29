"""
House Value ML Tutorial - Step by Step
Learn unsupervised learning to evaluate house prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: PREPARE YOUR DATA
# ============================================================================

# Sample house data - replace this with your actual data
house_data = {
    'size_sqft': [1500, 2000, 1200, 3000, 1800, 2500, 1100, 2200, 1900, 3500,
                  1600, 2100, 1400, 2800, 2000, 1700, 2300, 1300, 2600, 1900],
    'bedrooms': [3, 4, 2, 5, 3, 4, 2, 4, 3, 5,
                 3, 4, 2, 4, 3, 3, 4, 2, 4, 3],
    'bathrooms': [2, 2.5, 1, 3, 2, 2.5, 1, 2.5, 2, 3,
                  2, 2.5, 1, 3, 2, 2, 2.5, 1, 3, 2],
    'purchase_price': [250000, 380000, 180000, 550000, 300000, 420000, 150000, 350000, 310000, 650000,
                       280000, 390000, 170000, 480000, 320000, 290000, 400000, 160000, 460000, 330000],
    'age_years': [10, 5, 20, 3, 8, 6, 25, 4, 7, 2,
                  9, 5, 22, 4, 8, 10, 6, 23, 5, 9]
}

# Create DataFrame
df = pd.DataFrame(house_data)
print("=" * 70)
print("STEP 1: YOUR DATA")
print("=" * 70)
print(df.head(10))
print(f"\nDataset shape: {df.shape} (rows, columns)")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# STEP 2: LOAD YOUR OWN DATA (Optional)
# ============================================================================
# If you have a CSV file, uncomment this:
# df = pd.read_csv('your_houses.csv')

# ============================================================================
# STEP 3: DATA CLEANING & PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Remove rows with missing data (if any)
df = df.dropna()

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicate rows
df = df.drop_duplicates()

print(f"Clean dataset shape: {df.shape}")

# ============================================================================
# STEP 4: FEATURE SCALING (IMPORTANT!)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE SCALING")
print("=" * 70)
print("\nWhy scale? Different features have different units:")
print(f"  - Size: 1000-3500 sqft")
print(f"  - Bedrooms: 2-5")
print(f"  - Price: 150,000-650,000")
print("\nScaling makes them comparable for ML")

# Select features for ML (exclude price - we want to predict it)
features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years']
X = df[features].copy()

# Standardize features (0 mean, 1 std deviation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print(f"\nBefore scaling (first house):\n{X.iloc[0]}")
print(f"\nAfter scaling (first house):\n{X_scaled_df.iloc[0]}")

# ============================================================================
# STEP 5: FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: FIND OPTIMAL CLUSTERS (Elbow Method)")
print("=" * 70)
print("\nWhat are clusters?")
print("Groups of similar houses. We want to find natural groups.")

# Try different number of clusters
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    print(f"K={k}: Inertia = {kmeans_temp.inertia_:.2f}")

# Plot elbow curve
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method - Find Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=100)
print("\n✓ Saved: elbow_curve.png")
plt.close()

# ============================================================================
# STEP 6: BUILD CLUSTERING MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: BUILD CLUSTERING MODEL")
print("=" * 70)

# Use K=3 clusters (good balance)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nClusters created: {optimal_k}")
print(f"\nHouses per cluster:")
print(df['cluster'].value_counts().sort_index())

# Analyze clusters
print("\nCluster characteristics:")
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n--- CLUSTER {i} ({len(cluster_data)} houses) ---")
    print(cluster_data[['size_sqft', 'bedrooms', 'purchase_price', 'age_years']].describe().loc[['mean', 'min', 'max']])

# ============================================================================
# STEP 7: DETECT OUTLIERS (Good deals and bad deals)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: DETECT OUTLIERS (Unusual Prices)")
print("=" * 70)

# Use Isolation Forest to find unusual houses
isolator = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = isolator.fit_predict(X_scaled)
df['is_outlier'] = df['anomaly'] == -1

outliers = df[df['is_outlier']]
print(f"\nFound {len(outliers)} unusual houses:")
if len(outliers) > 0:
    print(outliers[['size_sqft', 'bedrooms', 'purchase_price', 'age_years', 'cluster']])

# ============================================================================
# STEP 8: CREATE PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: PREDICT PRICE FOR NEW HOUSE")
print("=" * 70)

def predict_house_value(size, bedrooms, bathrooms, age):
    """
    Predict if a house is a good deal by comparing to similar houses

    Args:
        size: square footage
        bedrooms: number of bedrooms
        bathrooms: number of bathrooms
        age: age in years

    Returns:
        dict with price prediction and analysis
    """

    # Scale the new data using the fitted scaler
    new_house = np.array([[size, bedrooms, bathrooms, age]])
    new_house_scaled = scaler.transform(new_house)

    # Find which cluster it belongs to
    cluster = kmeans.predict(new_house_scaled)[0]

    # Get all houses in this cluster
    similar_houses = df[df['cluster'] == cluster]

    if len(similar_houses) == 0:
        return {"error": "No similar houses found"}

    # Calculate statistics from similar houses
    avg_price = similar_houses['purchase_price'].mean()
    min_price = similar_houses['purchase_price'].min()
    max_price = similar_houses['purchase_price'].max()
    std_price = similar_houses['purchase_price'].std()

    # Analyze price per square foot
    similar_houses['price_per_sqft'] = similar_houses['purchase_price'] / similar_houses['size_sqft']
    avg_price_per_sqft = similar_houses['price_per_sqft'].mean()
    expected_price = size * avg_price_per_sqft

    return {
        'cluster': cluster,
        'similar_houses': len(similar_houses),
        'expected_price': round(expected_price, 2),
        'price_range': (round(min_price, 2), round(max_price, 2)),
        'average_price': round(avg_price, 2),
        'std_deviation': round(std_price, 2),
        'price_per_sqft': round(avg_price_per_sqft, 2),
        'similar_houses_data': similar_houses[['size_sqft', 'bedrooms', 'purchase_price']].to_dict()
    }

# TEST: Predict value for 3 example houses
print("\n--- TEST HOUSES ---")

test_houses = [
    {'size': 2000, 'bed': 4, 'bath': 2.5, 'age': 5, 'name': 'Modern house'},
    {'size': 1200, 'bed': 2, 'bath': 1, 'age': 20, 'name': 'Small old house'},
    {'size': 3200, 'bed': 5, 'bath': 3, 'age': 2, 'name': 'Large new house'},
]

for test in test_houses:
    result = predict_house_value(test['size'], test['bed'], test['bath'], test['age'])
    print(f"\n{test['name']}:")
    print(f"  Size: {test['size']} sqft | Bedrooms: {test['bed']} | Age: {test['age']} years")
    print(f"  → Cluster: {result['cluster']}")
    print(f"  → Expected price: ${result['expected_price']:,.0f}")
    print(f"  → Similar houses range: ${result['price_range'][0]:,.0f} - ${result['price_range'][1]:,.0f}")
    print(f"  → Average similar price: ${result['average_price']:,.0f}")
    print(f"  → Price/sqft: ${result['price_per_sqft']:.2f}")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: VISUALIZATIONS")
print("=" * 70)

# Plot 1: Size vs Price by Cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Size vs Price
ax = axes[0, 0]
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['size_sqft'], cluster_data['purchase_price'],
               label=f'Cluster {cluster}', s=100, alpha=0.7)
ax.set_xlabel('Size (sqft)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('House Size vs Purchase Price', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Bedrooms vs Price
ax = axes[0, 1]
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['bedrooms'], cluster_data['purchase_price'],
               label=f'Cluster {cluster}', s=100, alpha=0.7)
ax.set_xlabel('Bedrooms', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Bedrooms vs Purchase Price', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Price Distribution by Cluster
ax = axes[1, 0]
cluster_prices = [df[df['cluster'] == i]['purchase_price'].values for i in range(optimal_k)]
ax.boxplot(cluster_prices, labels=[f'Cluster {i}' for i in range(optimal_k)])
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Price Distribution by Cluster', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Age vs Price
ax = axes[1, 1]
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['age_years'], cluster_data['purchase_price'],
               label=f'Cluster {cluster}', s=100, alpha=0.7)
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('House Age vs Purchase Price', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_analysis.png', dpi=100)
print("✓ Saved: house_analysis.png")
plt.close()

# ============================================================================
# STEP 10: SAVE THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: SAVE YOUR MODEL")
print("=" * 70)

import joblib

# Save the models and scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler_model.pkl')
joblib.dump(isolator, 'isolator_model.pkl')
joblib.dump(df, 'training_data.pkl')

print("✓ Saved: kmeans_model.pkl (clustering model)")
print("✓ Saved: scaler_model.pkl (data scaler)")
print("✓ Saved: isolator_model.pkl (outlier detector)")
print("✓ Saved: training_data.pkl (original data)")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
print("\nNext: Use load_and_predict.py to load your model and predict new houses")
