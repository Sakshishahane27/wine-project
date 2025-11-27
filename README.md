# 🍷 Wine Quality Prediction System

A machine learning-based web application for predicting wine quality using multiple ML models including XGBoost, Random Forest, and Linear Regression.

## 📋 Features

- **Multiple ML Models**: XGBoost, Random Forest, and Linear Regression
- **High Accuracy**: Achieves ~80% accuracy in wine quality prediction
- **Interactive Web Interface**: Built with Streamlit for easy predictions
- **Data Analysis**: Comprehensive data visualization and feature analysis
- **Feature Engineering**: Advanced preprocessing and feature selection
- **Model Comparison**: Side-by-side performance evaluation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Download `winequality-red.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
   - Place it in the project root directory
   - Or the script will attempt to download it automatically

4. **Train the models**:
   ```bash
   python train_model.py
   ```
   This will:
   - Load and preprocess the data
   - Train all three ML models
   - Evaluate and compare model performance
   - Save trained models to `models/` directory
   - Generate visualizations in `plots/` directory

5. **Run the web application**:
   ```bash
   streamlit run app.py
   ```
   The application will open in your default web browser.

## 📁 Project Structure

```
wine project/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── utils.py               # Utility functions for preprocessing
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── models/               # Trained models (created after training)
│   ├── best_model.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── plots/                # Visualization plots (created after training)
│   ├── model_comparison.png
│   ├── correlation_heatmap.png
│   └── quality_distribution.png
│
└── winequality-red.csv   # Dataset (download separately)
```

## 🎯 Model Performance

The system evaluates three machine learning models:

1. **Linear Regression**: Simple baseline model
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting for high performance

**Expected Performance:**
- Accuracy: ~80% (predictions within 1 point of actual quality)
- R² Score: Varies by model (typically 0.3-0.5 for regression)
- Best Model: Usually XGBoost or Random Forest

## 💻 Usage

### Web Application

The Streamlit app provides three main pages:

1. **Prediction**: Enter wine characteristics to predict quality
   - Input 11 wine features
   - Select a model
   - Get instant quality predictions

2. **Data Analysis**: Explore the dataset
   - View dataset statistics
   - Analyze feature correlations
   - Visualize quality distributions

3. **Model Performance**: Compare model metrics
   - View performance visualizations
   - Compare accuracy and R² scores
   - Understand model characteristics

### Programmatic Usage

You can also use the models programmatically:

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare features (example values)
features = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 8.1, 3.09]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
print(f"Predicted Quality: {prediction[0]:.2f}")
```

## 🔬 Features Used for Prediction

1. Fixed Acidity (g/dm³)
2. Volatile Acidity (g/dm³)
3. Citric Acid (g/dm³)
4. Residual Sugar (g/dm³)
5. Chlorides (g/dm³)
6. Free Sulfur Dioxide (mg/dm³)
7. Total Sulfur Dioxide (mg/dm³)
8. Density (g/cm³)
9. pH
10. Sulphates (g/dm³)
11. Alcohol (% vol)
12. Total Acidity (engineered)
13. Sulfur Ratio (engineered)

## 📊 Data Preprocessing

- **Feature Engineering**: Created derived features (total acidity, sulfur ratio)
- **Scaling**: StandardScaler for normalization
- **Data Splitting**: 80/20 train-test split
- **Missing Values**: Handled with mean imputation

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Seaborn & Matplotlib**: Data visualization
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

## 📈 Model Validation

The system includes:
- Train-test split validation
- Multiple evaluation metrics (MSE, MAE, R², Accuracy)
- Feature importance analysis
- Model comparison visualizations

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Built with Streamlit, scikit-learn, and XGBoost

---

**Note**: Make sure to train the models (`python train_model.py`) before running the web application for the first time.

