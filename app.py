import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("electricity_model.pkl")

st.title("⚡ Sustainable Energy Predictor")
st.write("Predict **Electricity Generated** from farm & digester data.")

# --- Input fields ---
state = st.text_input("State")
digester_type = st.selectbox("Digester Type", ["Covered Lagoon", "Plug Flow", "Complete Mix", "Other"])
animal_count = st.number_input("Number of Animals", min_value=0, step=100)
biogas_produced = st.number_input("Biogas Produced (m³)", min_value=0, step=100)

# Put inputs into dataframe (must match training features!)
input_data = pd.DataFrame({
    "State": [state],
    "Digester Type": [digester_type],
    "Animal Count": [animal_count],
    "Biogas Produced": [biogas_produced]
})

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction = np.expm1(prediction)  # reverse log transform
    st.success(f"⚡ Predicted Electricity Generated: {prediction[0]:,.0f} kWh")
