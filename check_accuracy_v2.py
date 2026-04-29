"""
Better Accuracy Check - Using Price Per Square Foot
Sometimes simpler is better!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 70)
print("ALTERNATIVE APPROACH: PREDICT PRICE PER SQUARE FOOT")
print("=" * 70)

# Load data
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

print("\n" + "=" * 70)
print("APPROACH 1: PREDICT ABSOLUTE PRICE (Original)")
print("=" * 70)

X = df[['bed', 'bath', 'house_size', 'acre_lot']].copy()
X['acre_lot'] = X['acre_lot'].fillna(X['acre_lot'].mean())
y = df['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model1 = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)
r2_1 = r2_score(y_test, y_pred1)
mae_1 = mean_absolute_error(y_test, y_pred1)

print(f"\nPredicting: Price directly")
print(f"R² Score: {r2_1:.4f} (you got this)")
print(f"MAE Error: ${mae_1:,.0f}")
print(f"\n⚠ This approach doesn't work well with your data")

print("\n" + "=" * 70)
print("APPROACH 2: PREDICT PRICE PER SQUARE FOOT")
print("=" * 70)

# Calculate price per sqft
df['price_per_sqft'] = df['price'] / df['house_size']

print(f"\nPrice per sqft range in your data:")
print(f"  Min: ${df['price_per_sqft'].min():.2f}/sqft")
print(f"  Max: ${df['price_per_sqft'].max():.2f}/sqft")
print(f"  Avg: ${df['price_per_sqft'].mean():.2f}/sqft")

X2 = df[['bed', 'bath', 'acre_lot']].copy()
X2['acre_lot'] = X2['acre_lot'].fillna(X2['acre_lot'].mean())
y2 = df['price_per_sqft'].copy()

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

model2 = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model2.fit(X2_train_scaled, y2_train)

y2_pred = model2.predict(X2_test_scaled)
r2_2 = r2_score(y2_test, y2_pred)
mae_2 = mean_absolute_error(y2_test, y2_pred)

print(f"\nPredicting: Price per square foot")
print(f"R² Score: {r2_2:.4f}")
print(f"MAE Error: ${mae_2:.2f}/sqft")

if r2_2 > r2_1:
    print(f"\n✓ This works BETTER! (R² improved from {r2_1:.3f} to {r2_2:.3f})")
else:
    print(f"\n✗ This doesn't help much either")

print("\n" + "=" * 70)
print("APPROACH 3: AVERAGE PRICE BY FEATURES")
print("=" * 70)

print("\nInstead of ML, just use simple averages:")
print("\nAverage price by bedroom count:")
for beds in sorted(df['bed'].unique()):
    avg = df[df['bed'] == beds]['price'].mean()
    count = len(df[df['bed'] == beds])
    print(f"  {int(beds)} bed: ${avg:,.0f} (from {count} houses)")

print("\nAverage price per sqft by bedroom count:")
for beds in sorted(df['bed'].unique()):
    avg_psf = df[df['bed'] == beds]['price_per_sqft'].mean()
    count = len(df[df['bed'] == beds])
    print(f"  {int(beds)} bed: ${avg_psf:.2f}/sqft (from {count} houses)")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("""
Your data has a problem:
  - Only 28 houses
  - HUGE price variation
  - Features don't strongly predict price

BEST OPTIONS:

1. COLLECT MORE DATA (50-100+ houses)
   → Model will learn better patterns

2. USE SIMPLER METHOD (Recommended for now)
   → Use average price per sqft by bedroom count
   → Compare asking price to this average
   → Much more reliable with small data

3. ADD MORE FEATURES
   → House condition (excellent/good/fair/poor)
   → Neighborhood/location details
   → Year built or renovated
   → Special features

EXAMPLE WITH SIMPLER METHOD:
   3-bedroom average: $120,000
   House size: 1200 sqft
   Average $/sqft for 3-bed: $100/sqft
   Fair price = 1200 * $100 = $120,000

   If asking $110,000 → GOOD DEAL
   If asking $140,000 → OVERPRICED
""")
