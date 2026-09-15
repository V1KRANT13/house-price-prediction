import streamlit as st
import pandas as pd
import joblib


# Load trained model and preprocessor
model = joblib.load("house_price_model.pkl")
preprocessor = joblib.load("house_price_preprocessor.pkl")

# Load feature template
sample_input = pd.read_csv("sample_input.csv")


# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# Title
st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house to estimate its sale price."
)

st.divider()


# Input section
st.subheader("Property Details")


overall_qual = st.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=7
)


gr_liv_area = st.number_input(
    "Above Ground Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)


year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)


garage_cars = st.slider(
    "Garage Capacity",
    min_value=0,
    max_value=4,
    value=2
)


total_bsmt_sf = st.number_input(
    "Total Basement Area (sq ft)",
    min_value=0,
    max_value=5000,
    value=800
)


first_flr_sf = st.number_input(
    "First Floor Area (sq ft)",
    min_value=300,
    max_value=5000,
    value=1200
)


neighborhood = st.selectbox(
    "Neighborhood",
    sorted(sample_input["Neighborhood"].dropna().unique())
)


# Prediction
if st.button("🔮 Predict Price"):

    input_data = sample_input.copy()

    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["YearBuilt"] = year_built
    input_data["GarageCars"] = garage_cars
    input_data["TotalBsmtSF"] = total_bsmt_sf
    input_data["1stFlrSF"] = first_flr_sf
    input_data["Neighborhood"] = neighborhood

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]

    st.divider()

    st.success(
        f"🏠 Estimated House Price: ${prediction:,.0f}"
    )

    st.caption(
        "The prediction is generated using the trained Gradient Boosting model."
    )