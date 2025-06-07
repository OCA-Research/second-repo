import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load pre-trained models and scalers
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
autoencoder = load_model("lstm_autoencoder.h5")

st.set_page_config(page_title="Meter Tampering Detection", layout="wide")
st.title("AI-powered Meter Tampering Detection System")

uploaded_file = st.file_uploader("Upload smart meter data (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if st.button("Run Tampering Detection"):
        st.info("Processing data and running models. Please wait...")

        # Example: Assume preprocessing function is defined elsewhere
        try:
            from model_pipeline import preprocess_data, detect_tampering

            df_prepared, xgb_features, seq_data, customers = preprocess_data(df)

            xgb_preds, ae_preds, final_preds = detect_tampering(df_prepared, xgb_features, seq_data, customers,
                                                                xgb_model, autoencoder, scaler)

            # Combine results into a dataframe
            results_df = pd.DataFrame({
                'customer_id': customers,
                'XGBoost Prediction': xgb_preds,
                'Autoencoder Prediction': ae_preds,
                'Final Tampering Flag': final_preds
            })

            st.success("Detection completed!")
            st.subheader("Detection Results")
            st.dataframe(results_df)

            tampered_count = np.sum(final_preds)
            st.metric("Tampering Cases Detected", tampered_count)
        
        except Exception as e:
            st.error(f"Error in detection pipeline: {e}")

    st.markdown("---")
    st.caption("Note: This app uses both XGBoost and LSTM Autoencoder models to detect tampering patterns.")
else:
    st.warning("Please upload a CSV file to begin.")
