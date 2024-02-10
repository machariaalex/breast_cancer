import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_breast_cancer

# Load the trained model
model = joblib.load('breast_cancer_model.joblib')

# Load the breast cancer dataset
data = load_breast_cancer()
features = data.feature_names
X_sample_data = data.data[:5]  # Extract the first 5 rows for demonstration purposes

# Streamlit App
st.title("Breast Cancer Detection App")

# Display the first 5 rows of the dataset
st.header("First 5 Rows of the Breast Cancer Dataset")
st.write(pd.DataFrame(X_sample_data, columns=features))

# User input for features
st.sidebar.header('Input Features')

user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(0.0), float(10.0))

# Create a dataframe with user input and convert column names to strings
user_input_df = pd.DataFrame([user_input])

# Display user input
st.write("User Input:")
st.write(user_input_df)

# Ensure that the user input has the same number of features as the model expects
if user_input_df.shape[1] == len(features):
    # Preprocess user input
    scaler = StandardScaler()
    X_user_input = scaler.fit_transform(user_input_df)

    # Make predictions
    prediction = model.predict(X_user_input)
    probability = model.predict_proba(X_user_input)

    # Display result
    st.write(f"Predicted Class: {prediction[0]}")
    st.write(f"Probability of Malignancy: {probability[0][1]:.4f}")
else:
    st.error("Number of features in user input does not match the model's expectations.")
