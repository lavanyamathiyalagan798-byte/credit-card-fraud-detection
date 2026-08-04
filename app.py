import streamlit as st
import pandas as pd
import joblib
import random

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
model = joblib.load("fraud_model.pkl")

# Header
st.markdown(
    """
    <h1 style='text-align:center;color:#1E88E5;'>
    💳 Credit Card Fraud Detection System
    </h1>
    <h4 style='text-align:center;color:gray;'>
    AI Powered Secure Banking Transaction Verification
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Customer Details")

    customer = st.text_input("Customer Name")

    card = st.text_input("Card Number")

    amount = st.number_input("Transaction Amount (₹)", min_value=0.0)

    location = st.text_input("Transaction Location")

with col2:

    st.subheader("Transaction Details")

    merchant = st.text_input("Merchant Name")

    date = st.date_input("Transaction Date")

    payment = st.selectbox(
        "Payment Method",
        ["Online","POS Machine","ATM","UPI"]
    )

    device = st.selectbox(
        "Device",
        ["Mobile","Laptop","Desktop"]
    )

st.write("")

if st.button("🔍 Verify Transaction"):

    # Create 30 features for model
    features = [0]*30

    features[0] = random.uniform(-1,1)

    features[29] = amount

    columns = ["Time"] + \
        [f"V{i}" for i in range(1,29)] + \
        ["Amount"]

    data = pd.DataFrame([features],columns=columns)

    prediction = model.predict(data)

    st.divider()

    if prediction[0]==1:

        st.error("⚠ Fraudulent Transaction Detected")

    else:

        st.success("✅ Genuine Transaction")

    st.write("### Transaction Summary")

    st.write("Customer :",customer)

    st.write("Card Number :",card)

    st.write("Amount : ₹",amount)

    st.write("Merchant :",merchant)

    st.write("Location :",location)

    st.write("Payment :",payment)
    