import streamlit as st
import joblib
import pandas as pd
import webbrowser
from datetime import datetime

# Load the trained Random Forest model
@st.cache_data()
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error("An error occurred while loading the model.")
        st.error(str(e))
        return None

# Function to open a URL in a new tab
def open_url_in_new_tab(url):
    webbrowser.open_new_tab(url)

# Main function
def main():
    # Streamlit elements
    st.image('/home/space/Project01/src/Images/pic3.jpeg', caption=' ', use_column_width=True)
    st.title("Crop Analysis and Prediction Model")
    st.write("Smart Farming")
    st.sidebar.title("My Model")
    st.sidebar.image('/home/space/Project01/src/Images/pic6.jpeg', caption=' ', use_column_width=True,)

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
        #st.write("Redirecting to the webpage...")

    st.sidebar.write("Today")
    st.sidebar.write("Loading history...")

    # Load the trained Random Forest model
    model_path = "/home/space/Project01/src/models/random_forest_model.joblib"
    model = load_model(model_path)

    if model is None:
        return  # Stop further execution if model loading fails

    # Mapping between feature names used in Streamlit application and model
    feature_mapping = {
        "Humidity": "humidity",
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K",
        "Rainfall": "rainfall"

        # Add more mappings as needed
    }

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

    # Soil and weather parameters form
    with st.form("crop_analysis_form"):
        st.header("Soil and Weather Analysis Parameters")
        st.write("Please input the values below:")

        # Create input fields for soil and weather parameters
        nitrogen = st.number_input("Nitrogen", min_value=0.0, step=0.1)
        phosphorus = st.number_input("Phosphorus", min_value=0.0, step=0.1)
        potassium = st.number_input("Potassium", min_value=0.0, step=0.1)
        temperature = st.number_input("Temperature", min_value=-50.0, step=0.1)
        humidity = st.number_input("Humidity", min_value=0.0, step=0.1)
        ph = st.number_input("pH", min_value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall", min_value=0.0, step=0.1)

        # Add a button to submit the crop analysis parameters
        submit_analysis_button = st.form_submit_button("Submit Crop Analysis Data")

    # Process crop analysis form submission
    if submit_analysis_button:
        try:
            # Map feature names to the ones expected by the model
            user_data = pd.DataFrame({
                feature_mapping.get("Nitrogen", "N"): [nitrogen],
                feature_mapping.get("Phosphorus", "P"): [phosphorus],
                feature_mapping.get("Potassium", "K"): [potassium],
                feature_mapping.get("Temperature", "temperature"): [temperature],
                feature_mapping.get("Humidity", "humidity"): [humidity],
                feature_mapping.get("pH", "ph"): [ph],
                feature_mapping.get("Rainfall", "rainfall"): [rainfall]
            })

            # Make predictions using the Random Forest model
            prediction = model.predict(user_data)

            # Display predictions
            st.header("Crop Recommendations")
            st.write("Based on the provided parameters, the recommended crops are:")
            st.write(prediction)
        except Exception as e:
            st.error("An error occurred while processing the analysis.")
            st.error(str(e))

if __name__ == "__main__":
    main()
