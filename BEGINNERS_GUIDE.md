# House Price ML - Complete Beginner's Guide

## What You're Learning
You'll build an AI model that learns house prices and predicts if a new house is a good deal.

---

## PART 1: INSTALL PYTHON & TOOLS

### Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.12 (or latest)
3. Click installer and check "Add Python to PATH" ✓
4. Click Install

### Step 2: Verify Installation
1. Open Command Prompt (search "cmd" on Windows)
2. Type: `python --version`
3. You should see: `Python 3.12.x` (some version number)

### Step 3: Install Required Libraries
Libraries are like tools. We need:
- `pandas` = organize data
- `scikit-learn` = build ML models
- `matplotlib` = show graphs
- `joblib` = save/load models

Open Command Prompt and run:
```
pip install pandas scikit-learn matplotlib joblib numpy
```

Wait for it to finish. You'll see "Successfully installed..."

---

## PART 2: PREPARE YOUR DATA

### What is Data?
Your data is a table (like Excel) with house info:

```
size_sqft | bedrooms | bathrooms | age_years | purchase_price
2000      | 4        | 2.5       | 5         | 380000
1200      | 2        | 1         | 20        | 180000
3000      | 5        | 3         | 3         | 550000
```

- Each **row** = one house
- Each **column** = one characteristic
- We want the model to learn: "If size=2000 and bedrooms=4 → price=380000"

### Where to Get Your Data
**Option A: You Have Data**
- Export from Excel as CSV file
- Save as: `my_houses.csv`

**Option B: Use Sample Data** 
- I'll provide fake data to start

**CSV File Format:**
```
size_sqft,bedrooms,bathrooms,age_years,purchase_price
2000,4,2.5,5,380000
1200,2,1,20,180000
3000,5,3,3,550000
```

---

## PART 3: UNDERSTAND THE CODE

### Key Concepts

**1. Variables** = Container to store information
```python
name = "John"           # Text
age = 25                # Number
price = 350000.50       # Decimal number
```

**2. Lists** = Multiple items in one variable
```python
prices = [100000, 200000, 300000]
print(prices[0])        # Shows: 100000 (first item)
```

**3. Dictionaries** = Named containers
```python
house = {
    'size': 2000,
    'bedrooms': 4,
    'price': 380000
}
print(house['size'])    # Shows: 2000
```

**4. Functions** = Reusable code blocks
```python
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)           # Shows: 8
```

**5. Libraries** = Pre-written code you import
```python
import pandas as pd     # pd is a nickname for pandas

data = pd.read_csv('houses.csv')  # Load data from file
```

---

## PART 4: YOUR FIRST SCRIPT

Create a file called `step1_load_data.py`:

```python
# Step 1: Load Data

# Import libraries (tools we need)
import pandas as pd

# Load your CSV file
df = pd.read_csv('my_houses.csv')

# Show first 5 rows
print("First 5 houses:")
print(df.head())

# Show basic info
print("\nNumber of houses:", len(df))
print("\nColumn names:")
print(df.columns)

# Show statistics
print("\nPrice statistics:")
print(df['purchase_price'].describe())
```

**What this does:**
1. Imports pandas library
2. Reads your CSV file
3. Shows you the data
4. Shows price statistics (min, max, average)

**How to run it:**
1. Save the code as `step1_load_data.py`
2. Open Command Prompt
3. Go to the folder: `cd C:\Users\surfe\Downloads\Claude`
4. Run it: `python step1_load_data.py`

---

## PART 5: STEP-BY-STEP PROCESS

### Step 1: Load and Explore Data
```python
import pandas as pd

# Read your CSV file
df = pd.read_csv('my_houses.csv')

# Look at it
print(df.head(10))          # First 10 rows
print(df.describe())        # Statistics (average, min, max)
print(df.info())            # Data types and missing values
```

### Step 2: Clean Data
```python
# Remove rows with missing data
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print(f"Clean data: {len(df)} houses")
```

### Step 3: Prepare for ML
```python
# Features (X) = what we use to predict
X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]

# Target (y) = what we predict (price)
y = df['purchase_price']

print("Features shape:", X.shape)  # Number of rows and columns
print("Target shape:", y.shape)    # Number of prices
```

### Step 4: Split Data
```python
from sklearn.model_selection import train_test_split

# 80% training (learn), 20% testing (verify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # Same split every time
)

print(f"Training: {len(X_train)} houses")
print(f"Testing: {len(X_test)} houses")
```

### Step 5: Scale Data
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled successfully!")
```

**Why scale?** Make all numbers comparable (1-5000 vs 100000-650000)

### Step 6: Train Model
```python
from sklearn.ensemble import RandomForestRegressor

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train it (learn patterns from data)
model.fit(X_train_scaled, y_train)

print("Model trained!")
```

### Step 7: Test Model
```python
from sklearn.metrics import r2_score, mean_absolute_error

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
r2 = r2_score(y_test, y_pred)
error = mean_absolute_error(y_test, y_pred)

print(f"Model R² Score: {r2:.2f}")  # Higher is better (0-1)
print(f"Average Error: ${error:,.0f}")  # How much it's usually off
```

### Step 8: Predict New House
```python
import numpy as np

# New house: 2000 sqft, 4 bed, 2.5 bath, 5 years old
new_house = np.array([[2000, 4, 2.5, 5]])
new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled)[0]

print(f"Predicted price: ${predicted_price:,.0f}")
```

### Step 9: Save Model
```python
import joblib

# Save for later use
joblib.dump(model, 'my_model.pkl')
joblib.dump(scaler, 'my_scaler.pkl')

print("Model saved!")
```

### Step 10: Load and Use Saved Model
```python
import joblib
import numpy as np

# Load saved model
model = joblib.load('my_model.pkl')
scaler = joblib.load('my_scaler.pkl')

# Use it
new_house = np.array([[2500, 5, 3, 2]])
new_house_scaled = scaler.transform(new_house)
price = model.predict(new_house_scaled)[0]

print(f"Price: ${price:,.0f}")
```

---

## PART 6: COMMON ERRORS & FIXES

### Error: "No module named 'pandas'"
**Solution:** You didn't install libraries
```
pip install pandas scikit-learn matplotlib joblib numpy
```

### Error: "File not found: my_houses.csv"
**Solution:** File is in wrong folder or wrong name
- Check the file exists in same folder as your script
- Check spelling (capitals matter!)

### Error: "ValueError: could not convert"
**Solution:** CSV has non-number values in number columns
- Open CSV in Excel
- Fix any text mixed with numbers
- Save as CSV again

### Error: Model predictions are all the same
**Solution:** Need more data (at least 20-30 houses)
- Collect more house data

---

## PART 7: UNDERSTANDING RESULTS

### R² Score
```
R² = 0.95 → Excellent! 95% accurate
R² = 0.80 → Good
R² = 0.60 → Okay
R² = 0.40 → Poor
```

### Average Error
```
If model says $400,000 ± $10,000
Actual price is probably between $390,000 - $410,000
```

### Feature Importance
```
If Size = 45%, Bedrooms = 30%, Age = 20%, Bathrooms = 5%
→ Size matters most, then bedrooms
```

---

## PART 8: YOUR HOMEWORK

1. **Get your data:** 
   - Find 30+ real houses (Zillow, realtor.com)
   - Create CSV: `my_houses.csv`
   - Columns: size_sqft, bedrooms, bathrooms, age_years, purchase_price

2. **Create `train_model.py`:**
   - Load your data
   - Train the model
   - Test it
   - Save it

3. **Create `predict.py`:**
   - Load saved model
   - Predict prices for new houses

4. **Verify it works:**
   - Predict a house you know
   - Check if prediction is close to actual price

---

## READY TO START?

Tell me:
1. Do you have house data (CSV file)?
2. Or should I create sample data for you to practice with?
3. Which step do you want to start with?

I'll write the exact code you need to run!
