import streamlit as st
import pandas as pd
import joblib
import sklearn

# Load the saved pipeline
# The pipeline handles OneHotEncoding and scaling automatically
model = joblib.load("model.pkl")

st.title("Online Shopper Prediction")

Administrative = st.number_input("Administrative", 0)
Administrative_Duration = st.number_input("Administrative Duration", 0.0)
Informational = st.number_input("Informational", 0)
Informational_Duration = st.number_input("Informational Duration", 0.0)
ProductRelated = st.number_input("Product Related", 0)
ProductRelated_Duration = st.number_input("Product Related Duration", 0.0)
BounceRates = st.number_input("Bounce Rates", 0.0)
ExitRates = st.number_input("Exit Rates", 0.0)
PageValues = st.number_input("Page Values", 0.0)
SpecialDay = st.number_input("Special Day", 0.0)

Month = st.selectbox("Month",
    ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
)

OperatingSystems = st.number_input("Operating Systems", 1)
Browser = st.number_input("Browser", 1)
Region = st.number_input("Region", 1)
TrafficType = st.number_input("Traffic Type", 1)

VisitorType = st.selectbox("Visitor Type",
    ["Returning_Visitor", "New_Visitor", "Other"]
)

# Map the checkbox/selectbox to a boolean for the pipeline
Weekend = st.selectbox("Weekend", [True, False])

input_data = pd.DataFrame([{
    "Administrative": Administrative,
    "Administrative_Duration": Administrative_Duration,
    "Informational": Informational,
    "Informational_Duration": Informational_Duration,
    "ProductRelated": ProductRelated,
    "ProductRelated_Duration": ProductRelated_Duration,
    "BounceRates": BounceRates,
    "ExitRates": ExitRates,
    "PageValues": PageValues,
    "SpecialDay": SpecialDay,
    "Month": Month,
    "OperatingSystems": OperatingSystems,
    "Browser": Browser,
    "Region": Region,
    "TrafficType": TrafficType,
    "VisitorType": VisitorType,
    "Weekend": Weekend
}])

if st.button("Predict"):
    try:
        pred = model.predict(input_data)
        st.success("Will Purchase" if pred[0] else "Will NOT Purchase")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
