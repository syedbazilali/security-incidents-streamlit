import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Load model
# ----------------------------------
model = joblib.load("model.pkl")

# ----------------------------------
# App title
# ----------------------------------
st.title("Security Incident Risk Prediction")
st.write("Predict whether an incident will result in fatalities")

# ----------------------------------
# Load data (for dropdowns & structure)
# ----------------------------------
df = pd.read_csv("security_incidents.csv")

# ----------------------------------
# User Inputs
# ----------------------------------
st.header("Enter Incident Details")

country = st.selectbox("Country", sorted(df["Country"].dropna().unique()))
region = st.selectbox("Region", sorted(df["Region"].dropna().unique()))
actor_type = st.selectbox("Actor Type", sorted(df["Actor type"].dropna().unique()))
attack_context = st.selectbox("Attack Context", sorted(df["Attack context"].dropna().unique()))

latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)

# ----------------------------------
# Create input dataframe
# ----------------------------------
input_data = pd.DataFrame({
    "Country": [country],
    "Region": [region],
    "Actor type": [actor_type],
    "Attack context": [attack_context],
    "Latitude": [latitude],
    "Longitude": [longitude]
})

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fatal Incident Likely (Risk: {probability:.2f})")
    else:
        st.success(f"✅ No Fatality Likely (Risk: {probability:.2f})")
