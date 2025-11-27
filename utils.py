"""
Utility functions for data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(file_path='winequality-red.csv'):
    """
    Load wine quality dataset
    If file doesn't exist, download from UCI repository
    """
    if os.path.exists(file_path):
        # Try semicolon first (UCI format), then comma
        try:
            df = pd.read_csv(file_path, sep=';')
            # Check if we got the right number of columns
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=',')
        except:
            # If semicolon fails, try comma
            df = pd.read_csv(file_path, sep=',')
    else:
        # Try to download from UCI ML repository
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        try:
            df = pd.read_csv(url, sep=';')
            df.to_csv(file_path, index=False)
        except:
            # If download fails, create a sample dataset structure
            print("Warning: Could not download dataset. Please ensure winequality-red.csv exists.")
            return None
    return df


def preprocess_data(df):
    """
    Preprocess the wine quality dataset
    """
    # Handle any missing values in numeric columns first
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Create quality categories (0-3: poor, 4-6: average, 7-10: good)
    df['quality_category'] = pd.cut(df['quality'], bins=[0, 4, 6, 10], 
                                     labels=['Poor', 'Average', 'Good'])
    
    # Feature engineering
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    df['sulfur_ratio'] = df['total sulfur dioxide'] / (df['free sulfur dioxide'] + 1)
    
    return df


def prepare_features(df, target='quality'):
    """
    Prepare features and target variable for modeling
    """
    # Select features (excluding target and derived categorical)
    feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                   'residual sugar', 'chlorides', 'free sulfur dioxide',
                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    
    # Add engineered features if they exist
    if 'total_acidity' in df.columns:
        feature_cols.append('total_acidity')
    if 'sulfur_ratio' in df.columns:
        feature_cols.append('sulfur_ratio')
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    return X, y, feature_cols


def scale_features(X_train, X_test, save_scaler=True):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if save_scaler:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

