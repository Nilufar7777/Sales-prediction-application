import streamlit as st
import joblib
import pandas as pd

# Load the trained model and the exact column names/order used in training
model = joblib.load("final_model.pkl")
columns = joblib.load("column.pkl")

st.set_page_config(page_title="Sales Prediction", page_icon="📈")

st.title("📈 Sales Prediction (Random Forest)")
st.write("Enter advertising spend across TV, radio, and newspaper to predict sales.")

tv = st.number_input("TV Advertising Budget ($1000s)", value=100.0)
radio = st.number_input("Radio Advertising Budget ($1000s)", value=20.0)
newspaper = st.number_input("Newspaper Advertising Budget ($1000s)", value=10.0)

if st.button("Predict Sales"):
    input_df = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=columns
    )
    prediction = final_model.pkl.predict(input_df)[0]
    st.success(f"Predicted Sales: {prediction:.2f} units")
