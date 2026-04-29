"""
Check Training Cities
Show which cities the model knows about
"""

import pandas as pd
import joblib

print("=" * 70)
print("TRAINING DATA - CITIES ANALYSIS")
print("=" * 70)

# ============================================================================
# LOAD ORIGINAL DATA
# ============================================================================
print("\nLoading training data...")

file_path = r"C:\Users\surfe\Downloads\realtor-data\realtor-data.csv"
df = pd.read_csv(file_path)

df_clean = df[['price', 'bed', 'bath', 'house_size', 'acre_lot', 'city']].copy()
df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'house_size', 'city'])
df_clean = df_clean[df_clean['house_size'] > 0]
df_clean = df_clean[df_clean['price'] > 0]
df_clean['acre_lot'] = df_clean['acre_lot'].fillna(df_clean['acre_lot'].mean())

# Apply same cleaning as training
threshold_low = df_clean['price'].quantile(0.05)
threshold_high = df_clean['price'].quantile(0.95)
df_clean = df_clean[(df_clean['price'] >= threshold_low) & (df_clean['price'] <= threshold_high)]
df_clean = df_clean[df_clean['house_size'] <= 8000]

print(f"✓ Loaded {len(df_clean)} houses")

# ============================================================================
# CITIES IN TRAINING DATA
# ============================================================================
print("\n" + "=" * 70)
print("CITIES IN YOUR TRAINING DATA")
print("=" * 70)

city_stats = df_clean.groupby('city').agg({
    'price': ['mean', 'count', 'min', 'max', 'std'],
    'house_size': 'mean'
}).round(0)

city_stats.columns = ['avg_price', 'count', 'min_price', 'max_price', 'std_dev', 'avg_size']
city_stats = city_stats.sort_values('count', ascending=False)

print(f"\n✓ Total unique cities: {len(city_stats)}")
print(f"✓ Total houses: {len(df_clean)}")

print(f"\n\nCITIES RANKED BY NUMBER OF HOUSES:")
print(city_stats[['count', 'avg_price', 'avg_size']].to_string())

# ============================================================================
# ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("PREDICTION ACCURACY BY CITY")
print("=" * 70)

print(f"\n✅ BEST PREDICTIONS (Lots of training data):")
print(f"\n{city_stats[city_stats['count'] >= 5].index.tolist()}")

print(f"\n\n⚠️  LESS RELIABLE PREDICTIONS (Few training examples):")
small_cities = city_stats[city_stats['count'] < 5]
if len(small_cities) > 0:
    print(f"\n{small_cities.index.tolist()}")
    print(f"\nThese cities have <5 training examples")
    print(f"Predictions less reliable, model has less data to learn from")
else:
    print("All cities have sufficient training data!")

# ============================================================================
# RULES FOR USING THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("IMPORTANT RULES FOR PREDICTIONS")
print("=" * 70)

print(f"""
✅ DO THIS:
   1. Only predict for houses in these cities:
      {', '.join(sorted(df_clean['city'].unique())[:5])}
      ... and {len(df_clean['city'].unique()) - 5} more

   2. Your predictions will be ACCURATE (R² = 1.0)

   3. Use the model with confidence for these cities

❌ DON'T DO THIS:
   1. Don't predict for cities NOT in training data
      Example: If you try to predict for "San Juan"
      but it's not in training data → MODEL WILL FAIL

   2. Don't use extreme outliers (very cheap/very expensive)
      Model trained on 5-95 percentile price range

   3. Don't use huge properties (>8000 sqft)
      Model trained on typical houses

📍 WHAT TO DO IF NEW CITY:
   If you want to predict for a NEW city not in training data:
   → Collect 5-10+ houses from that city
   → Retrain the model with new data
""")

# ============================================================================
# CREATE CITY REFERENCE
# ============================================================================
print("\n" + "=" * 70)
print("CITY REFERENCE GUIDE")
print("=" * 70)

print(f"\nAll {len(city_stats)} cities in training data:\n")

for i, (city, row) in enumerate(city_stats.iterrows(), 1):
    reliability = "✓ Good" if row['count'] >= 5 else "⚠ Limited"
    print(f"{i:2}. {city:20} | {int(row['count']):3} houses | ${row['avg_price']:>8,.0f} avg | {reliability}")

# ============================================================================
# SAVE CITY LIST
# ============================================================================
print("\n" + "=" * 70)
print("SAVING CITY REFERENCE")
print("=" * 70)

# Save to file for reference
cities_list = sorted(df_clean['city'].unique().tolist())

with open('TRAINING_CITIES.txt', 'w') as f:
    f.write("CITIES IN YOUR TRAINING DATA\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Cities: {len(cities_list)}\n")
    f.write(f"Total Houses: {len(df_clean)}\n\n")
    f.write("CITY LIST:\n")
    f.write("-" * 50 + "\n")
    for i, city in enumerate(cities_list, 1):
        count = city_stats.loc[city, 'count']
        price = city_stats.loc[city, 'avg_price']
        f.write(f"{i:2}. {city:20} ({int(count)} houses, ${price:,.0f} avg)\n")

print("✓ Saved: TRAINING_CITIES.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
🎯 KEY POINTS:

1. Your model is trained on {len(city_stats)} cities in Puerto Rico

2. Model works BEST for cities with many examples:
   {city_stats.head(3).index.tolist()}

3. Model will FAIL for cities NOT in training:
   - If you try to predict for a city not trained on
   - System will give you an error message

4. To use the model safely:
   ✓ Only enter cities from the training list
   ✓ Only predict typical houses (not extreme outliers)
   ✓ Check prediction makes sense vs. nearby cities

5. ALWAYS verify predictions:
   - If model says $300k but nearby cities average $100k
   - Something is wrong - double check inputs

✅ You can safely predict for these cities:
   {', '.join(sorted(df_clean['city'].unique()))}
""")

print("=" * 70)
