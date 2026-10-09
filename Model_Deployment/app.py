import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

CLUSTER_INFO = {
    0: ("Single-item shopper",
        "Typically one product in one category, averaging about $200 per transaction."),
    1: ("Multi-category buyer",
        "Typically about 3 products across 2 categories, averaging about $442 per transaction."),
}

st.title("Transaction Segment Predictor")
st.write("Enter a transaction's details to see which customer segment it belongs to.")

total_products = st.number_input("Total products", min_value=1, value=1, step=1)
total_units = st.number_input("Total units", min_value=1, value=1, step=1)
total_revenue = st.number_input("Total revenue ($)", min_value=0.0, value=50.0, step=1.0)
categories = st.number_input("Number of categories", min_value=1, value=1, step=1)

if st.button("Predict segment"):
    new_data = pd.DataFrame([{
        "total_products": total_products,
        "total_units": total_units,
        "total_revenue": total_revenue,
        "categories": categories,
    }])
    scaled = scaler.transform(new_data)
    cluster = int(model.predict(scaled)[0])
    name, description = CLUSTER_INFO[cluster]
    st.subheader(f"Cluster {cluster}: {name}")
    st.write(description)