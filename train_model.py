"""
Train and evaluate multiple ML models for Wine Quality Prediction
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from utils import load_data, preprocess_data, prepare_features, split_data, scale_features

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def train_models(X_train, X_test, y_train, y_test, use_scaling=True):
    """
    Train multiple ML models and return their performance metrics
    """
    models = {}
    results = {}
    
    # Scale features if needed
    if use_scaling:
        X_train_processed, X_test_processed, scaler = scale_features(X_train, X_test, save_scaler=True)
    else:
        X_train_processed, X_test_processed = X_train, X_test
        scaler = None
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_processed, y_train)
    lr_pred = lr_model.predict(X_test_processed)
    
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = {
        'mse': mean_squared_error(y_test, lr_pred),
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred),
        'accuracy': calculate_accuracy(y_test, lr_pred)
    }
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_processed, y_train)
    rf_pred = rf_model.predict(X_test_processed)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'mse': mean_squared_error(y_test, rf_pred),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred),
        'accuracy': calculate_accuracy(y_test, rf_pred)
    }
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
    xgb_model.fit(X_train_processed, y_train)
    xgb_pred = xgb_model.predict(X_test_processed)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'mse': mean_squared_error(y_test, xgb_pred),
        'mae': mean_absolute_error(y_test, xgb_pred),
        'r2': r2_score(y_test, xgb_pred),
        'accuracy': calculate_accuracy(y_test, xgb_pred)
    }
    
    return models, results, scaler


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy as percentage of predictions within 1 point of actual quality
    """
    accuracy = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    return accuracy


def feature_importance_analysis(model, feature_names, model_name):
    """
    Analyze and visualize feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df


def save_models(models, scaler, best_model_name):
    """
    Save trained models to disk
    """
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
    
    if scaler:
        joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save best model separately for easy loading
    joblib.dump(models[best_model_name], 'models/best_model.pkl')
    print(f"\nBest model ({best_model_name}) saved to models/best_model.pkl")


def visualize_results(results, df, feature_cols):
    """
    Create visualization plots for model comparison and data analysis
    """
    os.makedirs('plots', exist_ok=True)
    
    # 1. Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    r2_scores = [results[m]['r2'] for m in model_names]
    
    axes[0].bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    axes[1].bar(model_names, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[1].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(r2_scores):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[feature_cols + ['quality']].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Quality Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='quality', palette='viridis')
    plt.title('Wine Quality Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations saved to 'plots' directory")


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Wine Quality Prediction System - Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading data...")
    df = load_data()
    if df is None:
        print("Error: Could not load dataset. Please ensure winequality-red.csv exists.")
        return
    
    print(f"   Dataset shape: {df.shape}")
    
    print("\n2. Preprocessing data...")
    df = preprocess_data(df)
    
    print("\n3. Preparing features...")
    X, y, feature_cols = prepare_features(df)
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    print("\n5. Training models...")
    models, results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    print("\n6. Model Performance:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    # Select best model based on accuracy
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\n[Best Model] {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.2f}%)")
    
    print("\n7. Saving models...")
    save_models(models, scaler, best_model_name)
    
    print("\n8. Creating visualizations...")
    visualize_results(results, df, feature_cols)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

