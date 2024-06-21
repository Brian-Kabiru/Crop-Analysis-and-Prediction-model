import streamlit as st
import random
import time
import webbrowser
from datetime import datetime

# Function to open a URL in a new tab
def open_url_in_new_tab(url):
    import webbrowser
    webbrowser.open_new_tab(url)

# Function to generate assistant response
def generate_assistant_response():
    return random.choice([
        "Hello there! How can I assist you today?",
        "Hi, human! Is there anything I can help you with?",
        "Do you need help?",
    ])

# Main function
def main():
    # Streamlit elements
    st.title("Simple Chat with Assistant")
    st.sidebar.title("Smart Farming")
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

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input and process chat interaction
    if prompt := st.text_input("You:", key="input_text"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        assistant_response = generate_assistant_response()

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Entry point of the application
if __name__ == "__main__":
    main()
