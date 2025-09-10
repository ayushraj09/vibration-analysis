import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_csv, predict_faults
import os

def load_models():
    try:
        binary_model = joblib.load("lgbm_binary_model.joblib")
        multi_model = joblib.load("lgbm_multi_model.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        return binary_model, multi_model, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def main():
    st.title("Motor Fault Detection System")
    st.write("Upload a CSV file from the MAFAULDA dataset to predict the fault type.")

    # Load models
    binary_model, multi_model, label_encoder = load_models()
    
    if all([binary_model, multi_model, label_encoder]):
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Process the file and make prediction
                with st.spinner('Processing data...'):
                    df_features = preprocess_csv(uploaded_file)
                    prediction = predict_faults(df_features, binary_model, multi_model, label_encoder)
                
                # Display results
                st.success("Analysis Complete!")
                st.header("Prediction Result:")
                st.write(f"**Detected Condition:** {prediction}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.error("Failed to load models. Please check if model files exist in the correct location.")

if __name__ == "__main__":
    main()