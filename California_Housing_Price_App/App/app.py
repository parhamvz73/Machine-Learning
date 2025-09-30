# =========================
# app.py - Streamlit frontend
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle

# =========================
# Load artifacts
# =========================
with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("schema.json", "r") as f:
    schema = json.load(f)

RAW_NUM_COLS = schema["raw_numeric_features"]
RAW_CAT_COLS = schema["categorical_features"]
RATIO_COLS   = schema["fe_ratio_cols"]
CAP_COLS     = schema["fe_cap_cols"]

# =========================
# Streamlit UI
# =========================
st.title("🏠 California Housing Price Predictor")

with st.form("input_form"):
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Median Age", value=41)
    total_rooms = st.number_input("Total Rooms", value=880)
    total_bedrooms = st.number_input("Total Bedrooms", value=129)
    population = st.number_input("Population", value=322)
    households = st.number_input("Households", value=126)
    median_income = st.number_input("Median Income", value=8.3252)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","ISLAND"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # The pipeline already handles preprocessing
    pred = model.predict(raw)[0]
    st.success(f"🏡 Predicted House Price: ${pred:,.0f}")
