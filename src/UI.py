import streamlit as st
import joblib
import pandas as pd
import numpy as np
import webbrowser
from datetime import datetime

# Load the trained Random Forest model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.success(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model from {model_path}.")
        st.error(str(e))
        return None

# Function to open a URL in a new tab
def open_url_in_new_tab(url):
    import webbrowser
    webbrowser.open_new_tab(url)

# Main function
def main():
    # Streamlit elements
    st.image('/home/space/Project01/src/Images/pic3.jpeg', caption=' ', use_column_width=True)
    st.title("Crop Analysis and Prediction Model")
    st.write("Smart Farming")
    st.sidebar.title("My Model")
    st.sidebar.image('/home/space/Project01/src/Images/pic6.jpeg', caption=' ', use_column_width=True)

    # Get current date and time
    current_datetime = datetime.now()

    # Format date and time as string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Display date and time on the sidebar
    st.sidebar.write(formatted_datetime)
    # Add text input for search
    search_query = st.sidebar.text_input("Search")

    # Add a button to explore machine learning models on the sidebar
    if st.sidebar.button("Explore Models"):
        # URL for the webpage containing machine learning models for crop analysis
        url = "https://example.com/machine-learning-models-crop-analysis"
        open_url_in_new_tab(url)

    st.sidebar.write("Today")
    st.sidebar.write("Loading history...")

    # Load the trained Random Forest model
    rf_model_path = '/home/space/Project01/docs/random_forest_model01.joblib'
    rf_model = load_model(rf_model_path)

    if rf_model is None:
        return  # Stop further execution if model loading fails

    # User details form
    with st.form("user_details_form"):
        st.subheader("Enter User Details")
        name = st.text_input("Name")
        email = st.text_input("Email")
        national_id = st.text_input("National ID")
        farm_number = st.text_input("Farm Number")
        user_details_submit_button = st.form_submit_button("Submit User Details")

    # Process user details form submission
    if user_details_submit_button:
        # Save user details to a database or file
        st.success("User details submitted successfully.")
        st.write("Name:", name)
        st.write("Email:", email)
        st.write("National ID:", national_id)
        st.write("Farm Number:", farm_number)

    # User input for prediction
    st.subheader("Enter Crop Details for Prediction")
    crop_name = st.text_input("Crop Name")
    target_yield = st.number_input("Target Yield", min_value=0.0)
    field_size = st.number_input("Field Size", min_value=0.0)
    ph_water = st.number_input("pH (water)", min_value=0.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
    total_nitrogen = st.number_input("Total Nitrogen", min_value=0.0)
    phosphorus = st.number_input("Phosphorus (M3)", min_value=0.0)
    potassium = st.number_input("Potassium (exch.)", min_value=0.0)
    soil_moisture = st.number_input("Soil moisture", min_value=0.0)

    prediction_submit_button = st.button("Predict Nutrient Needs")

    if prediction_submit_button:
        input_data = {
            'Crop Name': [crop_name],
            'Target Yield': [target_yield],
            'Field Size': [field_size],
            'pH (water)': [ph_water],
            'Organic Carbon': [organic_carbon],
            'Total Nitrogen': [total_nitrogen],
            'Phosphorus (M3)': [phosphorus],
            'Potassium (exch.)': [potassium],
            'Soil moisture': [soil_moisture]
        }

        input_df = pd.DataFrame(input_data)
        input_df = pd.get_dummies(input_df, columns=['Crop Name'])

        # Ensure the same columns as in the training data
        feature_columns = rf_model.feature_names_in_
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0
        input_df = input_df[feature_columns]  # Ensure the same column order as training data

        # Predict with Random Forest model
        rf_predictions = rf_model.predict(input_df)

        st.subheader("Predicted Nutrient Needs")
        st.write(f"Nitrogen (N) Need: {rf_predictions[0][0]}")
        st.write(f"Phosphorus (P2O5) Need: {rf_predictions[0][1]}")
        st.write(f"Potassium (K2O) Need: {rf_predictions[0][2]}")

if __name__ == "__main__":
    main()

