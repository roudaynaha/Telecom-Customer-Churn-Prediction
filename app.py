import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using our Deep Learning model.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load Model & Scaler
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    
    # Rebuild architecture to bypass Keras 2 / Keras 3 config serialization bugs
    model = Sequential([
        Dense(64, activation='relu', input_shape=(33,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Load only the weights
    model.load_weights('ann_model.h5')
    
    return scaler, model

try:
    scaler, model = load_assets()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Helper function to preprocess input
def preprocess_input(data):
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    
    # Feature Engineering (same as training notebook)
    df['AverageMonthlyCost'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Use training mean for MonthlyCharges: 64.76
    df['HighValueCustomer'] = (df['MonthlyCharges'] > 64.76).astype(int)
    df['LongTermCustomer'] = (df['tenure'] >= 24).astype(int)
    
    # Expected columns layout exactly as they were fitted during training
    expected_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'AverageMonthlyCost', 'HighValueCustomer', 'LongTermCustomer',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Initialize a dict with 0s for the expected columns
    encoded_data = {col: 0 for col in expected_cols}
    
    # Map numerical
    encoded_data['SeniorCitizen'] = float(df['SeniorCitizen'].iloc[0])
    encoded_data['tenure'] = float(df['tenure'].iloc[0])
    encoded_data['MonthlyCharges'] = float(df['MonthlyCharges'].iloc[0])
    encoded_data['TotalCharges'] = float(df['TotalCharges'].iloc[0])
    encoded_data['AverageMonthlyCost'] = float(df['AverageMonthlyCost'].iloc[0])
    encoded_data['HighValueCustomer'] = float(df['HighValueCustomer'].iloc[0])
    encoded_data['LongTermCustomer'] = float(df['LongTermCustomer'].iloc[0])
    
    # Map categorical
    if df['gender'].iloc[0] == 'Male': encoded_data['gender_Male'] = 1
    if df['Partner'].iloc[0] == 'Yes': encoded_data['Partner_Yes'] = 1
    if df['Dependents'].iloc[0] == 'Yes': encoded_data['Dependents_Yes'] = 1
    if df['PhoneService'].iloc[0] == 'Yes': encoded_data['PhoneService_Yes'] = 1
    
    ml = df['MultipleLines'].iloc[0]
    if ml == 'No phone service': encoded_data['MultipleLines_No phone service'] = 1
    elif ml == 'Yes': encoded_data['MultipleLines_Yes'] = 1
    
    ise = df['InternetService'].iloc[0]
    if ise == 'Fiber optic': encoded_data['InternetService_Fiber optic'] = 1
    elif ise == 'No': encoded_data['InternetService_No'] = 1
    
    def map_internet_addon(val, prefix):
        if val == 'No internet service': encoded_data[f'{prefix}_No internet service'] = 1
        elif val == 'Yes': encoded_data[f'{prefix}_Yes'] = 1
        
    map_internet_addon(df['OnlineSecurity'].iloc[0], 'OnlineSecurity')
    map_internet_addon(df['OnlineBackup'].iloc[0], 'OnlineBackup')
    map_internet_addon(df['DeviceProtection'].iloc[0], 'DeviceProtection')
    map_internet_addon(df['TechSupport'].iloc[0], 'TechSupport')
    map_internet_addon(df['StreamingTV'].iloc[0], 'StreamingTV')
    map_internet_addon(df['StreamingMovies'].iloc[0], 'StreamingMovies')
    
    contract = df['Contract'].iloc[0]
    if contract == 'One year': encoded_data['Contract_One year'] = 1
    elif contract == 'Two year': encoded_data['Contract_Two year'] = 1
    
    if df['PaperlessBilling'].iloc[0] == 'Yes': encoded_data['PaperlessBilling_Yes'] = 1
    
    pm = df['PaymentMethod'].iloc[0]
    if pm == 'Credit card (automatic)': encoded_data['PaymentMethod_Credit card (automatic)'] = 1
    elif pm == 'Electronic check': encoded_data['PaymentMethod_Electronic check'] = 1
    elif pm == 'Mailed check': encoded_data['PaymentMethod_Mailed check'] = 1
    
    # Important: Create DataFrame ensuring exact order of `scaler.feature_names_in_` if available
    try:
        if hasattr(scaler, 'feature_names_in_'):
            final_features = list(scaler.feature_names_in_)
        else:
            final_features = expected_cols
    except Exception:
        final_features = expected_cols
        
    # Build array based on final_features
    X_input = pd.DataFrame([[encoded_data.get(c, 0) for c in final_features]], columns=final_features)
    
    return X_input

st.subheader("Customer Information")

# Input layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.markdown("#### Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("#### Additional Services")
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")
st.markdown("#### Account & Billing")
col4, col5, col6 = st.columns(3)
with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col5:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=12)

with col6:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges * tenure if tenure > 0 else 0.0)

# Build payload
input_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

st.markdown("---")

if st.button("Predict Churn Risk", use_container_width=True, type="primary"):
    with st.spinner("Analyzing customer profile..."):
        try:
            # 1. Preprocess
            X_input = preprocess_input(input_data)
            
            # 2. Scale
            X_scaled = scaler.transform(X_input)
            
            # 3. Predict
            churn_prob = model.predict(X_scaled)[0][0]
            churn_pred = int(churn_prob > 0.5)
            
            st.markdown("### Prediction Result")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if churn_pred == 1:
                    st.error(f"⚠️ High Risk of Churn")
                else:
                    st.success(f"✅ Low Risk of Churn")
            
            with col_res2:
                st.metric(label="Churn Probability", value=f"{churn_prob:.1%}")
                
            st.progress(float(churn_prob))
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
