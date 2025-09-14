import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model pipeline
# Make sure 'xgb_model_pipeline.pkl' is in the same directory as your app.py file
import pickle
import os
import streamlit as st # Make sure streamlit is imported

try:
    with open("xgb_model_pipeline_pickle.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


st.title("⚡ Sustainable Energy Predictor")
st.write("Predict **Electricity Generated** from farm & digester data.")

# --- Input fields ---
project_type = st.selectbox("Project Type", ["Farm Scale", "Centralized/Regional", "Multiple Farm/Facility"])
city = st.text_input("City")
county = st.text_input("County")
state = st.text_input("State")
digester_type = st.selectbox("Digester Type", ["Covered Lagoon", "Mixed Plug Flow", "Complete Mix", "Unknown or Unspecified", "Horizontal Plug Flow", "Fixed Film/Attached Media", "Vertical Plug Flow", "Induced Blanket Reactor", "Lagoon", "Plug Flow"])
year_operational = st.number_input("Year Operational", min_value=1900, max_value=2024, step=1)
animal_farm_type = st.selectbox("Animal/Farm Type(s)", ["Swine", "Dairy", "Cattle", "Poultry"])
dairy_count = st.number_input("Dairy Count", min_value=0, step=1)
biogas_end_use = st.selectbox("Biogas End Use(s)", ["Flared Full-time", "Pipeline Gas", "CNG", "Cogeneration", "Electricity", "Electricity; Boiler/Furnace fuel", "Boiler/Furnace fuel", "Other", "Flared Part-time", "Electricity; Flared Full-time", "Electricity; Flared Part-time", "Flared Full-time; Other"])


# Put inputs into dataframe (must match training features!)
input_data = pd.DataFrame({
    "Project Type": [project_type],
    "City": [city],
    "County": [county],
    "State": [state],
    "Digester Type": [digester_type],
    "Year Operational": [year_operational],
    "Animal/Farm Type(s)": [animal_farm_type],
    "Dairy": [dairy_count],
    "Biogas End Use(s)": [biogas_end_use]
})

# --- Prediction ---
if st.button("Predict"):
    # Use the pipeline to predict
    prediction_log = model_pipeline.predict(input_data)
    prediction = np.expm1(prediction_log)  # reverse log transform
    st.success(f"⚡ Predicted Electricity Generated: {prediction[0]:,.0f} kWh")
