import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and encoder
model = joblib.load(r"C:\Users\Korisnik\Desktop\car_score_project\notebooks\gradient_boosting_model_tuned.pkl")
encoder = joblib.load(r"C:\Users\Korisnik\Desktop\car_score_project\notebooks\one_hot_encoder.pkl")

# Define valid input options
valid_buying = ['low', 'med', 'high', 'vhigh']
valid_maint = ['low', 'med', 'high', 'vhigh']
valid_doors = ['2', '3', '4', '5more']
valid_persons = ['2', '4', 'more']
valid_lug_boot = ['small', 'med', 'big']
valid_safety = ['low', 'med', 'high']

def predict_car_class(buying, maint, doors, persons, lug_boot, safety):
    """Encode and predict car class."""
    # Create a DataFrame for encoding
    input_data = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                              columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    # Encode using the saved encoder
    encoded_input = encoder.transform(input_data).toarray()
    # Predict the class
    prediction = model.predict(encoded_input)
    return prediction[0]

# Streamlit App Layout
st.title("Car Classification Application")
st.write("Predict the car class based on its features.")

# User Inputs
buying = st.selectbox("Buying Price", valid_buying)
maint = st.selectbox("Maintenance Cost", valid_maint)
doors = st.selectbox("Number of Doors", valid_doors)
persons = st.selectbox("Passenger Capacity", valid_persons)
lug_boot = st.selectbox("Luggage Boot Size", valid_lug_boot)
safety = st.selectbox("Safety Rating", valid_safety)

# Prediction Button
if st.button("Predict Car Class"):
    result = predict_car_class(buying, maint, doors, persons, lug_boot, safety)
    st.success(f"The predicted car class is: **{result}**")
