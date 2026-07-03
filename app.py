import streamlit as st
import pandas as pd
import pickle

# =========================
# Load trained model safely
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Online Shopper Intention Predictor", layout="centered")

st.title("🛒 Online Shopper Intention Predictor")
st.write("Predict whether a visitor will generate revenue based on session behavior.")

st.markdown("---")

# =========================
# User Inputs (REAL FEATURES ONLY)
# =========================

administrative = st.number_input("Administrative Pages", min_value=0, value=0)
administrative_duration = st.number_input("Administrative Duration", min_value=0.0, value=0.0)

informational = st.number_input("Informational Pages", min_value=0, value=0)
informational_duration = st.number_input("Informational Duration", min_value=0.0, value=0.0)

product_related = st.number_input("Product Related Pages", min_value=0, value=0)
product_related_duration = st.number_input("Product Related Duration", min_value=0.0, value=0.0)

bounce_rates = st.slider("Bounce Rate", 0.0, 1.0, 0.01)
exit_rates = st.slider("Exit Rate", 0.0, 1.0, 0.01)
page_values = st.number_input("Page Value", min_value=0.0, value=0.0)

special_day = st.slider("Special Day", 0.0, 1.0, 0.0)

# Month encoding (common dataset feature)
month = st.selectbox(
    "Month",
    ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

# Weekend feature
weekend = st.selectbox("Weekend Session?", ["No", "Yes"])

# Visitor type
visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])

st.markdown("---")

# =========================
# Encoding categorical values
# =========================
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "June": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

weekend_val = 1 if weekend == "Yes" else 0

visitor_map = {
    "New_Visitor": 0,
    "Returning_Visitor": 1,
    "Other": 2
}

# =========================
# Create input DataFrame
# IMPORTANT: must match training features
# =========================
input_data = pd.DataFrame([[
    administrative,
    administrative_duration,
    informational,
    informational_duration,
    product_related,
    product_related_duration,
    bounce_rates,
    exit_rates,
    page_values,
    special_day,
    month_map[month],
    weekend_val,
    visitor_map[visitor_type]
]], columns=model.feature_names_in_)

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Purchase Intent"):

    prediction = model.predict(input_data)

    # Probability (if model supports it)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"### 📊 Purchase Probability: {prob * 100:.2f}%")

    if prediction[0] == 1:
        st.success("✅ The visitor is likely to generate revenue!")
    else:
        st.error("❌ The visitor is unlikely to generate revenue.")
