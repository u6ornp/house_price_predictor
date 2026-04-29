# House Price Prediction - ML Training Project

A complete machine learning project for predicting house prices in Puerto Rico with 99.99% accuracy.

## 📁 Project Structure

```
ML Training/
├── web_app.py                      ← Main Flask web application
├── train_optimized.py              ← Model training script
├── predict_optimized.py            ← Prediction interface
│
├── templates/
│   └── index.html                  ← Web dashboard & prediction UI
│
├── *.pkl                           ← Trained model files
│   ├── house_model_optimized.pkl   ← Random Forest model
│   ├── house_scaler_optimized.pkl  ← Feature scaler
│   └── city_encoder_optimized.pkl  ← City encoder
│
├── realtor-data.csv                ← Training dataset (600+ houses)
├── houses.csv                      ← Sample data
│
├── requirements.txt                ← Python dependencies
├── Procfile                        ← Heroku deployment config
├── runtime.txt                     ← Python version
│
├── DEPLOYMENT_SUMMARY.md           ← Overview of deployment options
├── QUICK_SETUP.md                  ← 5-minute quick start
├── GITHUB_DEPLOYMENT_GUIDE.md      ← Detailed deployment guide
├── BEGINNERS_GUIDE.md              ← Getting started guide
│
├── Analysis Scripts/
│   ├── improve_accuracy.py         ← Test different ML approaches
│   ├── accuracy_check_flexible.py  ← Flexible accuracy checking
│   ├── check_training_cities.py    ← List trained cities
│   ├── location_analysis.py        ← Location importance analysis
│   ├── diagnose_and_improve.py     ← Diagnostic tools
│   └── [other analysis scripts]
│
└── Analysis Results/
    ├── accuracy_improvements.png   ← Accuracy comparison chart
    ├── accuracy_results.png        ← Results visualization
    ├── model_optimized.png         ← Model performance
    └── [other result images]
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_optimized.py
```

This trains the Random Forest model and saves 3 files:
- `house_model_optimized.pkl`
- `house_scaler_optimized.pkl`
- `city_encoder_optimized.pkl`

### 3. Run Web Interface
```bash
python web_app.py
```

Visit: `http://localhost:5000`

### 4. Make Predictions
- Go to "Make Prediction" tab
- Select city, enter house details
- Get price prediction and deal analysis

## 📊 Model Performance

- **Algorithm**: Random Forest Regressor (200 trees)
- **Accuracy**: R² = 0.9999 (99.99%)
- **Average Error**: ±$286
- **Training Data**: 600+ houses from 30+ Puerto Rico cities
- **Features**: 8 (bedrooms, bathrooms, size, acres, + 4 derived)

## 🎯 Features

✅ **Web Dashboard**
- View training data statistics
- Interactive charts and visualizations
- Market analysis by city

✅ **Price Prediction**
- Predict house prices
- Analyze if property is good deal
- View market comparisons

✅ **Data Visualization**
- Top cities by price
- Price distribution
- House size distribution
- Market trends

## 📈 Feature Engineering

The model uses these features:
- **Raw**: bedrooms, bathrooms, house_size, acre_lot, city
- **Derived**: price_per_sqft, total_rooms, rooms_per_acre

Derived features improved accuracy from R² = 0.57 → R² = 0.9999

## 🌍 Supported Cities

Model trained on 30+ Puerto Rico cities:
San Juan, Caguas, Carolina, Bayamón, Ponce, Mayagüez, and more.

Run `python check_training_cities.py` to see full list.

## 🧪 Testing & Analysis

Scripts available for analysis:
- `improve_accuracy.py` - Compare different ML approaches
- `location_analysis.py` - Analyze location importance
- `check_training_cities.py` - List trained cities and data
- `accuracy_check_flexible.py` - Flexible accuracy metrics

## 🚢 Deployment

### Option 1: Replit (Easiest - 2 minutes)
1. Create GitHub repo
2. Push code: `git push`
3. Go to Replit.com → Import from GitHub
4. Click "Run"

### Option 2: Heroku
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

### Option 3: Railway / Render
See `GITHUB_DEPLOYMENT_GUIDE.md`

## 📋 API Endpoints

```
GET  /                    → Main web interface
GET  /api/dashboard-data  → Dashboard statistics & charts
POST /api/predict         → Predict house price
POST /api/analyze-deal    → Analyze deal quality
GET  /api/city-comparison/<city> → City statistics
```

## 🔧 Technology Stack

- **Backend**: Python, Flask
- **ML**: Scikit-learn (Random Forest)
- **Data**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **Deployment**: Gunicorn, Heroku/Railway/Render

## 📝 Training Data

File: `realtor-data.csv`
- 600+ properties
- Multiple cities
- Features: price, bedrooms, bathrooms, house size, acres, city
- Cleaned and normalized

## 🎓 Learning Resources

- `BEGINNERS_GUIDE.md` - Step-by-step tutorial
- `QUICK_START.txt` - Quick reference
- Analysis scripts - See different ML techniques

## 🐛 Troubleshooting

**Models not loading?**
```bash
python train_optimized.py
```

**Web server not starting?**
- Check port 5000 is available
- Install Flask: `pip install Flask`

**Predictions wrong?**
- Ensure city is in training data
- Check input ranges are reasonable

## 📧 Next Steps

1. Deploy to web (see QUICK_SETUP.md)
2. Share URL with friends/colleagues
3. Collect user feedback
4. Enhance with more features/data

---

**Happy predicting! 🏠📈**
