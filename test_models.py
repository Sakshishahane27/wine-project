"""
Quick test script to verify models can be loaded correctly
"""
import joblib
import os
import numpy as np

print("Testing model loading...")
print("=" * 50)

# Check if models exist
model_files = {
    'Linear Regression': 'models/linear_regression.pkl',
    'Random Forest': 'models/random_forest.pkl',
    'XGBoost': 'models/xgboost.pkl',
    'Best Model': 'models/best_model.pkl'
}

scaler_path = 'models/scaler.pkl'

print("\n1. Checking model files:")
for name, path in model_files.items():
    exists = os.path.exists(path)
    status = "Found" if exists else "Missing"
    print(f"   {name}: [{status}] - {path}")
    if exists:
        try:
            model = joblib.load(path)
            print(f"      Model loaded successfully. Type: {type(model)}")
        except Exception as e:
            print(f"      ERROR loading model: {str(e)}")

print(f"\n2. Checking scaler:")
exists = os.path.exists(scaler_path)
status = "Found" if exists else "Missing"
print(f"   Scaler: [{status}] - {scaler_path}")
if exists:
    try:
        scaler = joblib.load(scaler_path)
        print(f"      Scaler loaded successfully. Type: {type(scaler)}")
        print(f"      Scaler expects {scaler.n_features_in_} features")
    except Exception as e:
        print(f"      ERROR loading scaler: {str(e)}")

print("\n3. Testing prediction with sample data:")
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Sample feature values (13 features)
    sample_features = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 8.1, 3.09]])
    
    print(f"   Sample features shape: {sample_features.shape}")
    print(f"   Expected features: {scaler.n_features_in_}")
    
    if sample_features.shape[1] == scaler.n_features_in_:
        scaled_features = scaler.transform(sample_features)
        prediction = model.predict(scaled_features)
        print(f"   [OK] Prediction successful: {prediction[0]:.2f}")
    else:
        print(f"   [ERROR] Feature mismatch! Expected {scaler.n_features_in_}, got {sample_features.shape[1]}")
        
except Exception as e:
    print(f"   [ERROR] {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Test complete!")

