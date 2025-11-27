"""
Streamlit Web Application for Wine Quality Prediction
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, preprocess_data, prepare_features
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress Streamlit ScriptRunContext warnings
import logging
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.caching').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Suppress warnings at the Python level
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction System",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Wine Theme CSS with Loading Animation
st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading Screen */
    .loading-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        animation: fadeOut 0.5s ease-out 5s forwards;
    }
    
    @keyframes fadeOut {
        to {
            opacity: 0;
            visibility: hidden;
        }
    }
    
    /* Wine Glass Container */
    .wine-container {
        position: relative;
        width: 200px;
        height: 200px;
        animation: zoomIn 0.8s ease-out 4.2s forwards;
        transform: scale(0.5);
    }
    
    @keyframes zoomIn {
        to {
            transform: scale(15);
            opacity: 0;
        }
    }
    
    /* Wine Glass 1 */
    .wine-glass-1 {
        position: absolute;
        width: 80px;
        height: 120px;
        left: 0;
        top: 20px;
        animation: mix1 2s ease-in-out infinite, colorChange1 2s ease-in-out infinite;
    }
    
    /* Wine Glass 2 */
    .wine-glass-2 {
        position: absolute;
        width: 80px;
        height: 120px;
        right: 0;
        top: 20px;
        animation: mix2 2s ease-in-out infinite, colorChange2 2s ease-in-out infinite;
    }
    
    @keyframes mix1 {
        0%, 100% { transform: translateX(0) rotate(0deg); }
        25% { transform: translateX(40px) rotate(-15deg); }
        50% { transform: translateX(60px) rotate(0deg); }
        75% { transform: translateX(40px) rotate(15deg); }
    }
    
    @keyframes mix2 {
        0%, 100% { transform: translateX(0) rotate(0deg); }
        25% { transform: translateX(-40px) rotate(15deg); }
        50% { transform: translateX(-60px) rotate(0deg); }
        75% { transform: translateX(-40px) rotate(-15deg); }
    }
    
    @keyframes colorChange1 {
        0% { filter: hue-rotate(0deg); }
        25% { filter: hue-rotate(90deg); }
        50% { filter: hue-rotate(180deg); }
        75% { filter: hue-rotate(270deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    @keyframes colorChange2 {
        0% { filter: hue-rotate(180deg); }
        25% { filter: hue-rotate(270deg); }
        50% { filter: hue-rotate(360deg); }
        75% { filter: hue-rotate(90deg); }
        100% { filter: hue-rotate(180deg); }
    }
    
    /* Wine Glass SVG */
    .wine-glass-svg {
        width: 100%;
        height: 100%;
    }
    
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #2d1b3d 0%, #3d2a4d 50%, #4d3a5d 100%);
    }
    
    .main-header {
        font-size: 4rem;
        font-weight: bold;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Georgia', serif;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(255,107,107,0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255,107,107,0.6);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #c44569 0%, #ee5a6f 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(196,69,105,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(196,69,105,0.6);
        background: linear-gradient(135deg, #ee5a6f 0%, #ff6b6b 100%);
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input {
        background: rgba(255,255,255,0.1);
        border: 2px solid rgba(255,107,107,0.3);
        border-radius: 10px;
        color: white;
    }
    
    .stSelectbox>div>div>select {
        background: rgba(255,255,255,0.1);
        border: 2px solid rgba(255,107,107,0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #c44569 0%, #ee5a6f 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 4px 20px rgba(196,69,105,0.6);
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s ease;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .fab:hover {
        transform: scale(1.1) rotate(15deg);
        box-shadow: 0 6px 30px rgba(196,69,105,0.8);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Success/Info Boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(46,213,115,0.2) 0%, rgba(0,184,148,0.2) 100%);
        border: 2px solid rgba(46,213,115,0.5);
        border-radius: 15px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(116,185,255,0.2) 0%, rgba(9,132,227,0.2) 100%);
        border: 2px solid rgba(116,185,255,0.5);
        border-radius: 15px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    </style>
    
    <!-- Loading Screen HTML -->
    <div class="loading-screen" id="loadingScreen">
        <div class="wine-container">
            <div class="wine-glass-1">
                <svg class="wine-glass-svg" viewBox="0 0 100 150" xmlns="http://www.w3.org/2000/svg">
                    <!-- Glass stem -->
                    <rect x="45" y="120" width="10" height="30" fill="#ffffff" opacity="0.8"/>
                    <!-- Glass base -->
                    <ellipse cx="50" cy="150" rx="15" ry="5" fill="#ffffff" opacity="0.8"/>
                    <!-- Wine bowl -->
                    <path d="M 30 20 Q 30 10 40 10 L 60 10 Q 70 10 70 20 L 65 100 Q 65 110 55 110 L 45 110 Q 35 110 35 100 Z" 
                          fill="#8B0000" opacity="0.9"/>
                    <!-- Wine liquid -->
                    <ellipse cx="50" cy="85" rx="15" ry="5" fill="#ff6b6b" opacity="0.7"/>
                </svg>
            </div>
            <div class="wine-glass-2">
                <svg class="wine-glass-svg" viewBox="0 0 100 150" xmlns="http://www.w3.org/2000/svg">
                    <!-- Glass stem -->
                    <rect x="45" y="120" width="10" height="30" fill="#ffffff" opacity="0.8"/>
                    <!-- Glass base -->
                    <ellipse cx="50" cy="150" rx="15" ry="5" fill="#ffffff" opacity="0.8"/>
                    <!-- Wine bowl -->
                    <path d="M 30 20 Q 30 10 40 10 L 60 10 Q 70 10 70 20 L 65 100 Q 65 110 55 110 L 45 110 Q 35 110 35 100 Z" 
                          fill="#8B0000" opacity="0.9"/>
                    <!-- Wine liquid -->
                    <ellipse cx="50" cy="85" rx="15" ry="5" fill="#4ecdc4" opacity="0.7"/>
                </svg>
            </div>
        </div>
    </div>
    
    <!-- Floating Action Button -->
    <div class="fab" onclick="window.scrollTo({top: 0, behavior: 'smooth'})" title="Scroll to Top">
        🍷
    </div>
    
    <script>
        // Remove loading screen after 5 seconds
        setTimeout(function() {
            var loadingScreen = document.getElementById('loadingScreen');
            if (loadingScreen) {
                loadingScreen.style.display = 'none';
            }
        }, 5000);
    </script>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    """Load and cache the wine dataset"""
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
    return df


@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'Linear Regression': 'models/linear_regression.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'Best Model': 'models/best_model.pkl'
    }
    
    scaler_path = 'models/scaler.pkl'
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                print(f"Error loading {name} from {path}: {str(e)}")
        else:
            print(f"Model file not found: {path}")
    
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {str(e)}")
    else:
        print(f"Scaler file not found: {scaler_path}")
    
    return models, scaler


def get_feature_names():
    """Get feature names for the model"""
    return ['fixed acidity', 'volatile acidity', 'citric acid',
           'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
           'total_acidity', 'sulfur_ratio']


def predict_quality(model, scaler, features):
    """Predict wine quality using the model"""
    feature_names = get_feature_names()
    
    # Check if all required features are present
    missing_features = [f for f in feature_names if f not in features]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    feature_array = np.array([features[f] for f in feature_names]).reshape(1, -1)
    
    if scaler:
        feature_array = scaler.transform(feature_array)
    
    prediction = model.predict(feature_array)[0]
    return max(0, min(10, round(prediction, 2)))  # Clamp between 0-10


def main():
    """Main application"""
    try:
        # Advanced Wine Theme Header
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-header">🍷 Wine Quality Prediction System 🍷</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; font-style: italic; margin-top: -1rem;">
                Uncork the Secrets of Wine Excellence
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page", ["Prediction", "Data Analysis", "Model Performance"])
        
        # Load data and models
        df = load_dataset()
        models, scaler = load_models()
        
        if df is None:
            st.error("⚠️ Dataset not found. Please ensure 'winequality-red.csv' exists in the project directory.")
            st.info("You can download it from: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/")
            return
        
        if not models:
            st.warning("⚠️ No trained models found. Please run 'train_model.py' first to train the models.")
            return
        
        # Page routing
        if page == "Prediction":
            prediction_page(models, scaler, df)
        elif page == "Data Analysis":
            data_analysis_page(df)
        elif page == "Model Performance":
            model_performance_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)


def prediction_page(models, scaler, df):
    """Wine quality prediction interface"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #ff6b6b; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">
            🍷 Predict Wine Quality 🍷
        </h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem; font-style: italic;">
            Enter the wine characteristics below to uncork the secret of its quality
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Wine Characteristics")
        
        # Get feature ranges from dataset for better UX
        feature_ranges = {
            'fixed acidity': (df['fixed acidity'].min(), df['fixed acidity'].max()),
            'volatile acidity': (df['volatile acidity'].min(), df['volatile acidity'].max()),
            'citric acid': (df['citric acid'].min(), df['citric acid'].max()),
            'residual sugar': (df['residual sugar'].min(), df['residual sugar'].max()),
            'chlorides': (df['chlorides'].min(), df['chlorides'].max()),
            'free sulfur dioxide': (df['free sulfur dioxide'].min(), df['free sulfur dioxide'].max()),
            'total sulfur dioxide': (df['total sulfur dioxide'].min(), df['total sulfur dioxide'].max()),
            'density': (df['density'].min(), df['density'].max()),
            'pH': (df['pH'].min(), df['pH'].max()),
            'sulphates': (df['sulphates'].min(), df['sulphates'].max()),
            'alcohol': (df['alcohol'].min(), df['alcohol'].max()),
        }
        
        # Input fields
        cols = st.columns(3)
        features = {}
        
        with cols[0]:
            features['fixed acidity'] = st.number_input(
                "Fixed Acidity (g/dm³)", 
                min_value=float(feature_ranges['fixed acidity'][0]),
                max_value=float(feature_ranges['fixed acidity'][1]),
                value=float(df['fixed acidity'].mean()),
                step=0.1
            )
            features['volatile acidity'] = st.number_input(
                "Volatile Acidity (g/dm³)",
                min_value=float(feature_ranges['volatile acidity'][0]),
                max_value=float(feature_ranges['volatile acidity'][1]),
                value=float(df['volatile acidity'].mean()),
                step=0.01
            )
            features['citric acid'] = st.number_input(
                "Citric Acid (g/dm³)",
                min_value=float(feature_ranges['citric acid'][0]),
                max_value=float(feature_ranges['citric acid'][1]),
                value=float(df['citric acid'].mean()),
                step=0.01
            )
            features['residual sugar'] = st.number_input(
                "Residual Sugar (g/dm³)",
                min_value=float(feature_ranges['residual sugar'][0]),
                max_value=float(feature_ranges['residual sugar'][1]),
                value=float(df['residual sugar'].mean()),
                step=0.1
            )
        
        with cols[1]:
            features['chlorides'] = st.number_input(
                "Chlorides (g/dm³)",
                min_value=float(feature_ranges['chlorides'][0]),
                max_value=float(feature_ranges['chlorides'][1]),
                value=float(df['chlorides'].mean()),
                step=0.001
            )
            features['free sulfur dioxide'] = st.number_input(
                "Free Sulfur Dioxide (mg/dm³)",
                min_value=float(feature_ranges['free sulfur dioxide'][0]),
                max_value=float(feature_ranges['free sulfur dioxide'][1]),
                value=float(df['free sulfur dioxide'].mean()),
                step=1.0
            )
            features['total sulfur dioxide'] = st.number_input(
                "Total Sulfur Dioxide (mg/dm³)",
                min_value=float(feature_ranges['total sulfur dioxide'][0]),
                max_value=float(feature_ranges['total sulfur dioxide'][1]),
                value=float(df['total sulfur dioxide'].mean()),
                step=1.0
            )
            features['density'] = st.number_input(
                "Density (g/cm³)",
                min_value=float(feature_ranges['density'][0]),
                max_value=float(feature_ranges['density'][1]),
                value=float(df['density'].mean()),
                step=0.0001,
                format="%.4f"
            )
        
        with cols[2]:
            features['pH'] = st.number_input(
                "pH",
                min_value=float(feature_ranges['pH'][0]),
                max_value=float(feature_ranges['pH'][1]),
                value=float(df['pH'].mean()),
                step=0.01
            )
            features['sulphates'] = st.number_input(
                "Sulphates (g/dm³)",
                min_value=float(feature_ranges['sulphates'][0]),
                max_value=float(feature_ranges['sulphates'][1]),
                value=float(df['sulphates'].mean()),
                step=0.01
            )
            features['alcohol'] = st.number_input(
                "Alcohol (% vol)",
                min_value=float(feature_ranges['alcohol'][0]),
                max_value=float(feature_ranges['alcohol'][1]),
                value=float(df['alcohol'].mean()),
                step=0.1
            )
        
        # Calculate derived features
        features['total_acidity'] = features['fixed acidity'] + features['volatile acidity']
        features['sulfur_ratio'] = features['total sulfur dioxide'] / (features['free sulfur dioxide'] + 1)
        
        # Model selection
        st.subheader("Model Selection")
        selected_model_name = st.selectbox(
            "Choose a model for prediction",
            list(models.keys()),
            index=list(models.keys()).index('Best Model') if 'Best Model' in models else 0
        )
        
        if st.button("🍷 Uncork the Quality Prediction 🍷", type="primary", use_container_width=True):
            try:
                if not scaler:
                    st.error("Scaler not found. Please retrain the models.")
                    return
                
                model = models[selected_model_name]
                prediction = predict_quality(model, scaler, features)
                
                # Display prediction
                st.markdown("---")
                st.success(f"### Predicted Wine Quality: **{prediction:.2f}** / 10")
                
                # Quality interpretation
                if prediction >= 7:
                    quality_text = "Excellent Quality 🏆"
                    quality_color = "#2ecc71"
                elif prediction >= 5:
                    quality_text = "Good Quality 👍"
                    quality_color = "#3498db"
                else:
                    quality_text = "Average Quality ⚠️"
                    quality_color = "#f39c12"
                
                st.markdown(f'<div style="text-align: center; padding: 1rem; background-color: {quality_color}; color: white; border-radius: 0.5rem; font-size: 1.5rem; font-weight: bold;">{quality_text}</div>', unsafe_allow_html=True)
                
                # Show predictions from all models
                st.markdown("---")
                st.subheader("Predictions from All Models")
                all_predictions = {}
                for model_name, model in models.items():
                    pred = predict_quality(model, scaler, features)
                    all_predictions[model_name] = pred
                
                pred_df = pd.DataFrame(list(all_predictions.items()), columns=['Model', 'Predicted Quality'])
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
    
    with col2:
        st.subheader("📊 Dataset Statistics")
        st.metric("Total Samples", len(df))
        st.metric("Features", len(get_feature_names()))
        st.metric("Avg Quality", f"{df['quality'].mean():.2f}")
        st.metric("Quality Range", f"{df['quality'].min()}-{df['quality'].max()}")
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.info("""
        **Higher quality wines typically have:**
        - Higher alcohol content
        - Moderate acidity
        - Balanced sulfur dioxide
        - Appropriate pH levels
        """)


def data_analysis_page(df):
    """Data analysis and visualization page"""
    st.header("📊 Data Analysis & Visualization")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Feature Analysis", "Quality Distribution"])
    
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(20), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Quality Range:** {df['quality'].min()} - {df['quality'].max()}")
            st.write(f"**Average Quality:** {df['quality'].mean():.2f}")
    
    with tab2:
        st.subheader("Feature Correlation with Quality")
        feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        
        correlations = df[feature_cols + ['quality']].corr()['quality'].sort_values(ascending=False)
        correlations = correlations.drop('quality')
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Feature Correlation with Wine Quality",
            labels={'x': 'Correlation Coefficient', 'y': 'Features'},
            color=correlations.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Relationships")
        selected_feature = st.selectbox("Select a feature to compare with quality", feature_cols)
        
        fig = px.scatter(
            df, x=selected_feature, y='quality',
            title=f"{selected_feature} vs Quality",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Quality Score Distribution")
        fig = px.histogram(
            df, x='quality', nbins=11,
            title="Distribution of Wine Quality Scores",
            labels={'quality': 'Quality Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            quality_counts = df['quality'].value_counts().sort_index()
            st.subheader("Quality Counts")
            st.bar_chart(quality_counts)
        
        with col2:
            if 'quality_category' in df.columns:
                category_counts = df['quality_category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Quality Category Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)


def model_performance_page():
    """Model performance comparison page"""
    st.header("🎯 Model Performance")
    
    # Check if plots exist
    if os.path.exists('plots/model_comparison.png'):
        st.subheader("Model Comparison")
        st.image('plots/model_comparison.png', use_container_width=True)
    
    if os.path.exists('plots/correlation_heatmap.png'):
        st.subheader("Feature Correlation Matrix")
        st.image('plots/correlation_heatmap.png', use_container_width=True)
    
    if os.path.exists('plots/quality_distribution.png'):
        st.subheader("Quality Distribution")
        st.image('plots/quality_distribution.png', use_container_width=True)
    
    if not any(os.path.exists(f'plots/{f}') for f in ['model_comparison.png', 'correlation_heatmap.png', 'quality_distribution.png']):
        st.info("📝 Run 'train_model.py' to generate model performance visualizations.")
    
    st.markdown("---")
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Linear Regression**
        - Simple and interpretable
        - Fast training
        - Good baseline model
        """)
    
    with col2:
        st.markdown("""
        **Random Forest**
        - Handles non-linear relationships
        - Feature importance analysis
        - Robust to overfitting
        """)
    
    with col3:
        st.markdown("""
        **XGBoost**
        - Gradient boosting algorithm
        - High performance
        - Handles complex patterns
        """)


if __name__ == "__main__":
    main()

