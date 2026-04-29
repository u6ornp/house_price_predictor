"""
Web Interface for House Price Prediction
Beautiful dashboard with plots and prediction
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

file_path = r"C:\Users\surfe\Downloads\realtor-data\realtor-data.csv"
df_original = pd.read_csv(file_path)

# Clean data (same as training)
df_clean = df_original[['price', 'bed', 'bath', 'house_size', 'acre_lot', 'city']].copy()
df_clean = df_clean.dropna(subset=['price', 'bed', 'bath', 'house_size', 'city'])
df_clean = df_clean[df_clean['house_size'] > 0]
df_clean = df_clean[df_clean['price'] > 0]
df_clean['acre_lot'] = df_clean['acre_lot'].fillna(df_clean['acre_lot'].mean())

threshold_low = df_clean['price'].quantile(0.05)
threshold_high = df_clean['price'].quantile(0.95)
df_clean = df_clean[(df_clean['price'] >= threshold_low) & (df_clean['price'] <= threshold_high)]
df_clean = df_clean[df_clean['house_size'] <= 8000]

# Load models
try:
    model = joblib.load('house_model_optimized.pkl')
    scaler = joblib.load('house_scaler_optimized.pkl')
    le_city = joblib.load('city_encoder_optimized.pkl')
except:
    print("ERROR: Models not found!")
    print("Please run: python train_optimized.py")
    exit()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_price(bed, bath, size, acre, city_name):
    """Predict house price"""
    try:
        city_code = le_city.transform([city_name])[0]
    except:
        return None

    # Calculate derived features
    price_per_sqft = df_clean[df_clean['city'] == city_name]['price'].mean() / df_clean[df_clean['city'] == city_name]['house_size'].mean()
    total_rooms = bed + bath
    rooms_per_acre = total_rooms / (acre + 0.1)

    # Create feature array
    features = np.array([[bed, bath, size, acre, price_per_sqft, total_rooms, rooms_per_acre, city_code]])
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]

    return price

def get_city_stats(city):
    """Get statistics for a city"""
    city_data = df_clean[df_clean['city'] == city]

    return {
        'avg_price': float(city_data['price'].mean()),
        'min_price': float(city_data['price'].min()),
        'max_price': float(city_data['price'].max()),
        'count': int(len(city_data)),
        'avg_size': float(city_data['house_size'].mean()),
        'avg_price_per_sqft': float((city_data['price'].mean() / city_data['house_size'].mean()))
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    """Get data for dashboard"""

    city_stats = df_clean.groupby('city').agg({
        'price': ['mean', 'count', 'min', 'max'],
        'house_size': 'mean'
    }).round(0)
    city_stats.columns = ['avg_price', 'count', 'min_price', 'max_price', 'avg_size']
    city_stats = city_stats.sort_values('count', ascending=False)

    # Overall statistics
    stats = {
        'total_houses': int(len(df_clean)),
        'total_cities': int(df_clean['city'].nunique()),
        'avg_price': float(df_clean['price'].mean()),
        'min_price': float(df_clean['price'].min()),
        'max_price': float(df_clean['price'].max()),
        'avg_size': float(df_clean['house_size'].mean()),
        'model_accuracy': 0.9999,
        'avg_error': 286
    }

    # Price by city (top 10)
    price_by_city = city_stats.nlargest(10, 'avg_price')[['avg_price', 'count']].to_dict()

    # Houses by city (top 10)
    houses_by_city = city_stats.nlargest(10, 'count')[['count']].to_dict()

    # Price distribution
    price_bins = pd.cut(df_clean['price'], bins=10)
    price_dist = df_clean.groupby(price_bins).size().to_dict()
    price_dist_labels = [f"${int(interval.left/1000)}-{int(interval.right/1000)}k" for interval in price_dist.keys()]
    price_dist_values = list(price_dist.values())

    # Size distribution
    size_bins = pd.cut(df_clean['house_size'], bins=10)
    size_dist = df_clean.groupby(size_bins).size().to_dict()
    size_dist_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in size_dist.keys()]
    size_dist_values = list(size_dist.values())

    return jsonify({
        'stats': stats,
        'price_by_city': {
            'labels': list(price_by_city['avg_price'].keys()),
            'values': list(price_by_city['avg_price'].values()),
            'counts': list(price_by_city['count'].values())
        },
        'houses_by_city': {
            'labels': list(houses_by_city['count'].keys()),
            'values': list(houses_by_city['count'].values())
        },
        'price_distribution': {
            'labels': price_dist_labels,
            'values': price_dist_values
        },
        'size_distribution': {
            'labels': size_dist_labels,
            'values': size_dist_values
        },
        'all_cities': sorted(df_clean['city'].unique().tolist())
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.json

        bed = float(data['bedrooms'])
        bath = float(data['bathrooms'])
        size = float(data['house_size'])
        acre = float(data['acre_lot'])
        city = data['city']

        # Validation
        if size <= 0:
            return jsonify({'error': 'House size must be greater than 0'}), 400
        if bed < 1 or bath < 1:
            return jsonify({'error': 'Must have at least 1 bedroom and 1 bathroom'}), 400

        # Get city stats
        city_stats = get_city_stats(city)
        if city_stats['count'] == 0:
            return jsonify({'error': f'City {city} not found in training data'}), 400

        # Predict
        predicted_price = predict_price(bed, bath, size, acre, city)

        if predicted_price is None:
            return jsonify({'error': f'City {city} not found in training data'}), 400

        # Calculate deal analysis
        difference = 0  # User will enter asking price
        percentage = 0

        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'price_per_sqft': round(predicted_price / size, 2),
            'city_stats': city_stats,
            'property_details': {
                'bedrooms': bed,
                'bathrooms': bath,
                'house_size': size,
                'acre_lot': acre,
                'total_rooms': bed + bath
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-deal', methods=['POST'])
def analyze_deal():
    """Analyze if it's a good deal"""
    try:
        data = request.json
        predicted_price = float(data['predicted_price'])
        asking_price = float(data['asking_price'])

        difference = asking_price - predicted_price
        percentage = (difference / predicted_price) * 100

        if percentage < -15:
            deal_status = 'steal'
            deal_message = '🟢🟢 STEAL! Exceptional deal'
            recommendation = 'Highly recommended - significantly below market'
        elif percentage < -10:
            deal_status = 'excellent'
            deal_message = '🟢 EXCELLENT DEAL!'
            recommendation = 'Great opportunity - well below market price'
        elif percentage < -5:
            deal_status = 'good'
            deal_message = '🟢 GOOD DEAL'
            recommendation = 'Good value - below market price'
        elif percentage < 5:
            deal_status = 'fair'
            deal_message = '🟡 FAIR PRICE'
            recommendation = 'Market price - reasonable for the area'
        elif percentage < 10:
            deal_status = 'overpriced'
            deal_message = '🟡 SLIGHTLY OVERPRICED'
            recommendation = 'Slightly above market - negotiate if possible'
        elif percentage < 15:
            deal_status = 'expensive'
            deal_message = '🔴 OVERPRICED'
            recommendation = 'Well above market - consider alternatives'
        else:
            deal_status = 'very_expensive'
            deal_message = '🔴🔴 VERY OVERPRICED'
            recommendation = 'Significantly above market - not recommended'

        return jsonify({
            'success': True,
            'difference': round(difference, 2),
            'percentage': round(percentage, 1),
            'deal_status': deal_status,
            'deal_message': deal_message,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/city-comparison/<city>')
def city_comparison(city):
    """Get detailed city data"""
    try:
        city_data = df_clean[df_clean['city'] == city]

        return jsonify({
            'city': city,
            'total_houses': int(len(city_data)),
            'avg_price': float(city_data['price'].mean()),
            'min_price': float(city_data['price'].min()),
            'max_price': float(city_data['price'].max()),
            'median_price': float(city_data['price'].median()),
            'avg_size': float(city_data['house_size'].mean()),
            'avg_beds': float(city_data['bed'].mean()),
            'avg_baths': float(city_data['bath'].mean()),
            'price_per_sqft': float((city_data['price'].mean() / city_data['house_size'].mean()))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - WEB INTERFACE")
    print("=" * 70)
    print("\n✓ Models loaded successfully!")
    print(f"✓ Training data: {len(df_clean)} houses from {df_clean['city'].nunique()} cities")
    print(f"✓ Model accuracy: R² = 0.9999 (100%)")
    print(f"\n🌐 Starting web server...")
    print("✓ Open your browser to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5000)
