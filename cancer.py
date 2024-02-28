import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_breast_cancer
import requests  # Assuming you'll use requests library for API calls

# Replace with your actual API key
API_KEY = "AIzaSyAfSRafLWuXss-I-IpSvkqUlwTRptx1ilU"

# Load the trained model
model = joblib.load('import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_breast_cancer
import requests  # Assuming you'll use requests library for API calls

# Replace with your actual API key
API_KEY = "AIzaSyAfSRafLWuXss-I-IpSvkqUlwTRptx1ilU"

# Load the trained model
model = joblib.load('/home/ndegwa/cancer/mode.joblib')

# Load the breast cancer dataset
data = load_breast_cancer()
features = data.feature_names
X_sample_data = data.data[:5]  # Extract the first 5 rows for demonstration purposes

# Streamlit App
st.title("Breast Cancer Detection App with Gemini Integration")

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

    # Display prediction result
    st.write(f"Predicted Class: {prediction[0]}")
    st.write(f"Probability of Malignancy: {probability[0][1]:.4f}")
else:
    st.error("Number of features in user input does not match the model's expectations.")

# Interact with Gemini (optional)
if st.checkbox("Interact with Gemini about the results"):
    prompt = st.text_input("Ask Gemini a question related to the prediction")

    if st.button("Generate Response"):
        # API request structure (adjust based on Gemini API specifications)
        url = "https://api.gemini.ai/v1/generate"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        data = {"prompt": prompt}

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for non-2xx status codes

            # Handle successful response (adjust based on Gemini API response format)
            gemini_response = response.json()["generated_text"]
            st.success("Gemini's Response:", gemini_response)
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:  # Handle other potential errors
            st.error(f"Unexpected error: {str(e)}")

# ... additional app logic, dashboards, visualizations, etc. (optional)
')

# Load the breast cancer dataset
data = load_breast_cancer()
features = data.feature_names
X_sample_data = data.data[:5]  # Extract the first 5 rows for demonstration purposes

# Streamlit App
st.title("Breast Cancer Detection App with Gemini Integration")

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

    # Display prediction result
    st.write(f"Predicted Class: {prediction[0]}")
    st.write(f"Probability of Malignancy: {probability[0][1]:.4f}")
else:
    st.error("Number of features in user input does not match the model's expectations.")

# Interact with Gemini (optional)
if st.checkbox("Interact with Gemini about the results"):
    prompt = st.text_input("Ask Gemini a question related to the prediction")

    if st.button("Generate Response"):
        # API request structure (adjust based on Gemini API specifications)
        url = "https://api.gemini.ai/v1/generate"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        data = {"prompt": prompt}

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for non-2xx status codes

            # Handle successful response (adjust based on Gemini API response format)
            gemini_response = response.json()["generated_text"]
            st.success("Gemini's Response:", gemini_response)
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:  # Handle other potential errors
            st.error(f"Unexpected error: {str(e)}")

# ... additional app logic, dashboards, visualizations, etc. (optional)
