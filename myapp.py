import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.joblib")

st.title("Housing Prices Prediction")
st.write("please give accurately")
longitude = st.number_input("Enter the longitude ", format="%.4f")
latitude = st.number_input("Enter the latitude ", format="%.4f")
housing_median_age = st.number_input("Enter the housing median age ", format="%.4f")
total_rooms = st.number_input("Enter the total rooms ")
total_bedrooms = st.number_input("Enter the total bedrooms ")
population = st.number_input("Enter the population ")
households = st.number_input("Enter the households ")
median_income = st.number_input("Enter the median income ", format="%.4f")

columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
           "median_income"]


def predict():
    arr = np.array(
        [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income])

    X = pd.DataFrame([arr], columns=columns)
    prediction = model.predict(X)
    st.success("the predicted house price is "+prediction)


st.button("predict", on_click=predict)