#import the necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib as plt
import seaborn as sns
from datetime import datetime
import joblib
import os


# Load model and encoder
clf = joblib.load('start_location_classifier.joblib')
le = joblib.load('start_location_label_encoder.joblib')

# App title
st.title("🚗 Uber Start Location Predictor")
st.write("Enter the hour of the day to predict the most likely start location.")

# Input section
hour_val = st.number_input("Hour (0-23):", min_value=0, max_value=23, value=8)

# Predict button
if st.button("Predict Location"):
    pred_class_idx = clf.predict(np.array([[hour_val]]))[0]
    pred_label = le.inverse_transform([pred_class_idx])[0]
    st.success(f"Predicted Start Location: **{pred_label}**")
