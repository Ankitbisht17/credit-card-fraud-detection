# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection - Logistic Regression", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    return model

def main():
    st.title("💳 Credit Card Fraud Detection (Logistic Regression)")
    st.write(
        """
        This web app uses a **Logistic Regression** model trained on the  
        **Kaggle Credit Card Fraud Dataset** to classify transactions as  
        **Fraudulent (1)** or **Legitimate (0)**.
        """
    )

    model = load_model()

    st.header("📂 Upload Transactions CSV File")
    st.write(
        """
        **Input format requirements:**
        - Columns should match the original dataset (except `Class` column is optional)  
        - Typically: `Time`, `V1` ... `V28`, `Amount`  
        - If your file contains `Class`, it will be ignored for prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file with transaction data", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            st.subheader("👀 Preview of Uploaded Data")
            st.dataframe(data.head())

            # Drop target column if present
            if "Class" in data.columns:
                st.info("ℹ️ 'Class' column found and will be dropped for prediction.")
                data = data.drop("Class", axis=1)

            # Make sure only numeric columns are used
            if not np.all([np.issubdtype(dt, np.number) for dt in data.dtypes]):
                st.error("❌ Non-numeric columns detected. Ensure all features are numeric.")
                return

            if st.button("🔍 Predict Fraud"):
                # Predict probabilities and classes
                probs = model.predict_proba(data)[:, 1]   # probability of fraud
                preds = (probs >= 0.5).astype(int)

                result_df = data.copy()
                result_df["Fraud_Probability"] = probs
                result_df["Prediction"] = preds  # 1 = Fraud, 0 = Legit

                st.subheader("📊 Prediction Results (Top 50 rows)")
                st.dataframe(result_df.head(50))

                fraud_count = int((preds == 1).sum())
                legit_count = int((preds == 0).sum())

                st.markdown("### 🔢 Summary")
                st.write(f"- Total transactions: **{len(preds)}**")
                st.write(f"- Predicted **Fraudulent**: **{fraud_count}**")
                st.write(f"- Predicted **Legitimate**: **{legit_count}**")

                # Download button
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="logistic_fraud_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")

    st.markdown("---")
    st.caption(
        "This is an academic demo using Logistic Regression with class imbalance handling (`class_weight='balanced'`). "
        "Real banking fraud systems combine ML with business rules, network analysis, and human review."
    )

if __name__ == "__main__":
    main()
