# =========================
# app.py - Streamlit frontend
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import lightgbm as lgb

# =========================
# Load artifacts
# =========================
bst = lgb.Booster(model_file="housing_model_lgb.txt")
km = joblib.load("kmeans.pkl")

with open("schema.json", "r") as f:
    schema = json.load(f)

RAW_NUM_COLS = schema["raw_numeric_features"]
RAW_CAT_COLS = schema["categorical_features"]
RATIO_COLS   = schema["fe_ratio_cols"]
CAP_COLS     = schema["fe_cap_cols"]

# =========================
# Feature Engineering
# =========================
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, np.nan)
    df["bedrooms_ratio"] = df["total_bedrooms"] / df["total_rooms"].replace(0, np.nan)
    df["population_per_household"] = df["population"] / df["households"].replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)

def add_caps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_age_capped"] = (df["housing_median_age"] == 52).astype(int)
    df["is_income_capped"] = (df["median_income"] == 15.0001).astype(int)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ratios(add_caps(df))
    df["region_cluster"] = km.predict(df[["longitude","latitude"]])

    # one-hot encode categorical cols
    df = pd.get_dummies(df, columns=["ocean_proximity","region_cluster"])

    # align with model features
    df = df.reindex(columns=bst.feature_name(), fill_value=0)
    return df

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

    features = preprocess(raw)
    pred = bst.predict(features)[0]
    st.success(f"🏡 Predicted House Price: ${pred:,.0f}")
