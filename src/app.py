import streamlit as st
import joblib
import pandas as pd

model =joblib.load("model.pkl")

st.title("📉 Customer Churn Prediction")

umur = st.slider("Umur", 18, 60)
langganan = st.slider("Lama Berlangganan (bulan)", 1, 36)
usage = st.slider("Usage", 50, 500)

if st.button("Prediksi"):
    input_data = pd.DataFrame([[umur, langganan, usage]], columns=["umur","langganan","usage"])
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("Customer kemungkinan CHURN ❌")
    else :
        st.success("Customer kemungkinan STAY ✅")