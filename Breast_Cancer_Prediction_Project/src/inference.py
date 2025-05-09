import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('models/model.pkl')
le = joblib.load('models/label_encoder.pkl')

# Sample input
sample = pd.DataFrame([{
    'radius_mean': 14.0,
    'texture_mean': 20.0,
    'perimeter_mean': 90.0,
    'area_mean': 600.0,
    'smoothness_mean': 0.1,
    'compactness_mean': 0.2,
    'concavity_mean': 0.1,
    'concave points_mean': 0.05,
    'symmetry_mean': 0.2,
    'fractal_dimension_mean': 0.06,
    'radius_se': 0.5,
    'texture_se': 1.0,
    'perimeter_se': 3.0,
    'area_se': 40.0,
    'smoothness_se': 0.005,
    'compactness_se': 0.02,
    'concavity_se': 0.02,
    'concave points_se': 0.01,
    'symmetry_se': 0.02,
    'fractal_dimension_se': 0.003,
    'radius_worst': 16.0,
    'texture_worst': 25.0,
    'perimeter_worst': 100.0,
    'area_worst': 800.0,
    'smoothness_worst': 0.15,
    'compactness_worst': 0.3,
    'concavity_worst': 0.2,
    'concave points_worst': 0.1,
    'symmetry_worst': 0.3,
    'fractal_dimension_worst': 0.08
}])

# Predict
prediction = model.predict(sample)
prediction_label = le.inverse_transform(prediction)
print(f"Prediction: {prediction_label[0]}")
