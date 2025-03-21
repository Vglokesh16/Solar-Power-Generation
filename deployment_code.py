
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("solar_power_generation_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Solar Power Generation Prediction")

st.sidebar.header("Input Features")

def user_input_features():
    st.sidebar.header("User Input Parameters")  # Sidebar title

    # Sidebar input fields
    distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon")
    temperature = st.sidebar.number_input("Temperature")
    wind_direction = st.sidebar.number_input("Wind Direction")
    wind_speed = st.sidebar.number_input("Wind Speed")
    sky_cover = st.sidebar.number_input("Sky Cover")
    visibility = st.sidebar.number_input("Visibility")
    humidity = st.sidebar.number_input("Humidity")
    avg_wind_speed = st.sidebar.number_input("Average Wind Speed")
    avg_pressure = st.sidebar.number_input("Average Pressure")

    # Create DataFrame with exact column names from the array
    data = pd.DataFrame({
        "distance-to-solar-noon": [distance_to_solar_noon],
        "temperature": [temperature],
        "wind-direction": [wind_direction],
        "wind-speed": [wind_speed],
        "sky-cover": [sky_cover],
        "visibility": [visibility],
        "humidity": [humidity],
        "average-wind-speed-(period)": [avg_wind_speed],
        "average-pressure-(period)": [avg_pressure],
        # "power-generated": [power_generated],
    })

    return data

df = user_input_features()
st.write("### User Input Features:")
st.write(df)

df_scaled = scaler.transform(df)

if st.button("Predict Power Generation"):
    prediction = model.predict(df_scaled)
    st.success(f"Predicted Power Output: {prediction[0]:.2f} kW")
