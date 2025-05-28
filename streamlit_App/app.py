import streamlit as st
import pandas as pd
import joblib

# Load the forecast model
import os

# Construct the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'Model', 'arima_model.pkl')
model = joblib.load(model_path)

# Title of the app
st.title('Forecasting App')

# Upload sample data
uploaded_file = st.file_uploader("Upload your input data (CSV)", type='csv')

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data)

    # Process the data and make predictions
    # Assuming the model expects a specific format, adjust as necessary
    predictions = model.predict(data)

    # Display the predictions
    st.write("Forecasted Values:")
    st.dataframe(predictions)

# Instructions for using the app
st.sidebar.header('Instructions')
st.sidebar.write('1. Upload your CSV file containing the input data.')
st.sidebar.write('2. The app will display the data and the forecasted values.')