
import joblib
import pandas as pd

# Load the model directly
model = joblib.load("backend/xgb_tuned_model.joblib")
import streamlit as st
import requests

st.title("Superkart Sales Prediction")

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=4.0, max_value=22.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.004, max_value=0.298, value=0.056, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=31.0, max_value=266.0, value=146.74)
Store_Size = st.selectbox("Store Size", ["Medium", "High", "Small"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type2", "Departmental Store", "Supermarket Type1", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Code", ["FD", "NC", "DR"])
Store_Age_Years = st.number_input("Store Age Years", min_value=16, max_value=38, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Non Perishables", "Perishables"])

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

if st.button("Predict", type="primary"):
    try:
        # Convert your product_data dictionary to a DataFrame
        input_df = pd.DataFrame([product_data])

        # Predict directly
        predicted_sales = model.predict(input_df)[0]
        st.write(
            f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}"
        )
    except Exception as e:
        st.error(f"Error making prediction: {e}")

