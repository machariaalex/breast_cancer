import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_breast_cancer

# Function to train the model
def train_model():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the trained model: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(model, 'breast_cancer_model.joblib')
    st.write("Model has been trained and saved.")

# Function to deploy the Streamlit app
def streamlit_app():
    # Load the trained model
    model = joblib.load('breast_cancer_model.joblib')

    # Streamlit App
    st.title("Breast Cancer Detection App")

    # User input for features
    st.sidebar.header('Input Features')

    # Assuming you have 30 features in your dataset
    features = load_breast_cancer().feature_names

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

# Check if the script is being run for training or deployment
if __name__ == "__main__":
    train_model()
else:
    streamlit_app()
