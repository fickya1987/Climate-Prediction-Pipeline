import streamlit as st
import numpy as np
import pickle

# Load your pre-trained RandomForestRegressor model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set the title of the app
st.title("Predicting Temperature in London")

# header and description
st.header("Predict Weather Conditions Based on Various Features")

# link to the GitHub repository
st.markdown("[GitHub Repo](https://github.com/Netcodez/Climate-Prediction-Pipeline)")

st.write(
    """
This application uses a Random Forest Regression model to predict the mean temperature in London. 
Please enter the required features below to get a prediction.
"""
)
feature_names = [
    "Cloud Cover (oktas)",
    "Sunshine (hrs)",
    "Global Radiation (W/m²)",
    "Max Temp (°C)",
    "Min Temp (°C)",
    "Precipitation (mm)",
    "Pressure (Pa)",
    "Snow Depth (cm)",
    "Month",
]

# feature names with units of measurement and short descriptions
feature_info = {
    "Cloud Cover (oktas)": {
        "range": (0.0, 9.0),
        "description": "Measurement of cloud cover in oktas",
    },
    "Sunshine (hrs)": {
        "range": (0.0, 24.0),
        "description": "Measurement of sunshine in hours per day",
    },
    "Global Radiation (W/m²)": {
        "range": (0.0, 500.0),
        "description": "Measurement of global radiation in Watt per " "square meter",
    },
    "Max Temp (°C)": {
        "range": (-10.0, 40.0),
        "description": "Maximum temperature recorded in degrees Celsius",
    },
    "Min Temp (°C)": {
        "range": (-10.0, 40.0),
        "description": "Minimum temperature recorded in degrees Celsius",
    },
    "Precipitation (mm)": {
        "range": (0.0, 100.0),
        "description": "Measurement of precipitation in millimeters",
    },
    "Pressure (Pa)": {
        "range": (90000.0, 110000.0),
        "description": "Measurement of pressure in Pascals",
    },
    "Snow Depth (cm)": {
        "range": (0.0, 50.0),
        "description": "Measurement of snow depth in centimeters",
    },
    "Month": {"range": (1, 12), "description": "Month of observation"},
}

feature_values = []
for feature_name, info in feature_info.items():
    min_val, max_val = info["range"]
    label = f"{feature_name}: {info['description']}"
    feature_values.append(st.slider(label, min_val, max_val, min_val))


# Prediction button
if st.button("Predict"):
    input_data = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"Mean Temperature in London is predicted to be {prediction[0]}°C")
