import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('models/model.pkl')
le = joblib.load('models/label_encoder.pkl')

st.title("Breast Cancer Prediction")

# Input fields
st.sidebar.header("Input Features")
def user_input_features():
    data = {
        'radius_mean': st.sidebar.slider('Radius Mean', 6.0, 30.0, 14.0),
        'texture_mean': st.sidebar.slider('Texture Mean', 10.0, 40.0, 20.0),
        'perimeter_mean': st.sidebar.slider('Perimeter Mean', 40.0, 200.0, 90.0),
        'area_mean': st.sidebar.slider('Area Mean', 140.0, 2500.0, 600.0),
        'smoothness_mean': st.sidebar.slider('Smoothness Mean', 0.05, 0.2, 0.1),
        'compactness_mean': st.sidebar.slider('Compactness Mean', 0.02, 0.4, 0.2),
        'concavity_mean': st.sidebar.slider('Concavity Mean', 0.0, 0.5, 0.1),
        'concave points_mean': st.sidebar.slider('Concave Points Mean', 0.0, 0.2, 0.05),
        'symmetry_mean': st.sidebar.slider('Symmetry Mean', 0.1, 0.4, 0.2),
        'fractal_dimension_mean': st.sidebar.slider('Fractal Dimension Mean', 0.05, 0.1, 0.06),
        'radius_se': st.sidebar.slider('Radius SE', 0.1, 2.5, 0.5),
        'texture_se': st.sidebar.slider('Texture SE', 0.5, 5.0, 1.0),
        'perimeter_se': st.sidebar.slider('Perimeter SE', 1.0, 10.0, 3.0),
        'area_se': st.sidebar.slider('Area SE', 10.0, 100.0, 40.0),
        'smoothness_se': st.sidebar.slider('Smoothness SE', 0.001, 0.03, 0.005),
        'compactness_se': st.sidebar.slider('Compactness SE', 0.005, 0.1, 0.02),
        'concavity_se': st.sidebar.slider('Concavity SE', 0.005, 0.1, 0.02),
        'concave points_se': st.sidebar.slider('Concave Points SE', 0.0, 0.05, 0.01),
        'symmetry_se': st.sidebar.slider('Symmetry SE', 0.005, 0.05, 0.02),
        'fractal_dimension_se': st.sidebar.slider('Fractal Dimension SE', 0.001, 0.02, 0.003),
        'radius_worst': st.sidebar.slider('Radius Worst', 7.0, 40.0, 16.0),
        'texture_worst': st.sidebar.slider('Texture Worst', 12.0, 50.0, 25.0),
        'perimeter_worst': st.sidebar.slider('Perimeter Worst', 50.0, 250.0, 100.0),
        'area_worst': st.sidebar.slider('Area Worst', 200.0, 4000.0, 800.0),
        'smoothness_worst': st.sidebar.slider('Smoothness Worst', 0.07, 0.3, 0.15),
        'compactness_worst': st.sidebar.slider('Compactness Worst', 0.03, 1.0, 0.3),
        'concavity_worst': st.sidebar.slider('Concavity Worst', 0.0, 1.5, 0.2),
        'concave points_worst': st.sidebar.slider('Concave Points Worst', 0.0, 0.5, 0.1),
        'symmetry_worst': st.sidebar.slider('Symmetry Worst', 0.1, 0.6, 0.3),
        'fractal_dimension_worst': st.sidebar.slider('Fractal Dimension Worst', 0.05, 0.2, 0.08)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_label = le.inverse_transform(prediction)

st.subheader('Prediction')
st.write(f"The tumor is predicted to be **{prediction_label[0]}**.")
