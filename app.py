import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import sys

# -------------------------------
# 1️⃣ Safe model loading
# -------------------------------
def load_model(path):
    """
    Tries to load a model safely using joblib and pickle fallbacks.
    """
    try:
        # Try standard joblib load
        model = joblib.load(path)
        st.success("Model loaded successfully with joblib!")
        return model
    except AttributeError:
        st.warning("Joblib failed. Trying pickle fallback...")
        try:
            with open(path, "rb") as f:
                model = pickle.load(f, encoding='latin1')
            st.success("Model loaded successfully with pickle fallback!")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            sys.exit()
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        sys.exit()

# Load your model
model_pipeline = load_model("xgb_model_pipeline.pkl")

# -------------------------------
# 2️⃣ Streamlit UI
# -------------------------------
st.title("⚡ Sustainable Energy Predictor")
st.write("Predict **Electricity Generated** from farm & digester data.")

# --- Input fields ---
project_type = st.selectbox("Project Type", ["Farm Scale", "Centralized/Regional", "Multiple Farm/Facility"])
city = st.text_input("City")
county = st.text_input("County")
state = st.text_input("State")
digester_type = st.selectbox(
    "Digester Type",
    ["Covered Lagoon", "Mixed Plug Flow", "Complete Mix", "Unknown or Unspecified",
     "Horizontal Plug Flow", "Fixed Film/Attached Media", "Vertical Plug Flow",
     "Induced Blanket Reactor", "Lagoon", "Plug Flow"]
)
year_operational = st.number_input("Year Operational", min_value=1900, max_value=2024, step=1)
animal_farm_type = st.selectbox("Animal/Farm Type(s)", ["Swine", "Dairy", "Cattle", "Poultry"])
dairy_count = st.number_input("Dairy Count", min_value=0, step=1)
biogas_end_use = st.selectbox(
    "Biogas End Use(s)",
    ["Flared Full-time", "Pipeline Gas", "CNG", "Cogeneration", "Electricity",
     "Electricity; Boiler/Furnace fuel", "Boiler/Furnace fuel", "Other", "Flared Part-time",
     "Electricity; Flared Full-time", "Electricity; Flared Part-time", "Flared Full-time; Other"]
)

# --- Prepare input dataframe ---
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

# -------------------------------
# 3️⃣ Make prediction
# -------------------------------
if st.button("Predict"):
    try:
        # Make prediction
        prediction_log = model_pipeline.predict(input_data)
        prediction = np.expm1(prediction_log)  # reverse log transform
        st.success(f"⚡ Predicted Electricity Generated: {prediction[0]:,.0f} kWh")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that input features match the model training data.")
