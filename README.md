# credit-card-fraud-detection
# Credit Card Fraud Detection using Logistic Regression

This project is a **Machine Learning–based Credit Card Fraud Detection System** built using **Logistic Regression**.  
It classifies transactions as **Fraudulent** or **Legitimate** using the popular **Kaggle Credit Card Fraud Dataset**.  
The system also includes a **Streamlit web app** for real-time prediction using CSV file upload.

---

## Project Features

- Detects **fraudulent credit card transactions**
- Handles **highly imbalanced dataset**
- Uses **Logistic Regression with Standard Scaling**
- Provides **AUC, confusion matrix & classification report**
- Includes a **user-friendly Streamlit web interface**
- Downloadable **prediction results**

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:**  
  - pandas  
  - scikit-learn  
  - joblib  
  - streamlit  
- **ML Algorithm:** Logistic Regression  
- **Deployment:** Streamlit Web App  
- **Version Control:** Git & GitHub  

---

## Project Structure
credit-card-fraud-detection/
│
├── app.py # Streamlit web app
├── train_logistic_regression.py # Model training script
├── logistic_model.pkl # Trained model (ignored in Git)
├── .gitignore # Ignored files
└── README.md # Project documentation


---

## Dataset

- **Name:** Credit Card Fraud Dataset  
- **Source (Kaggle):**  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- **Records:** 284,807 transactions  
- **Fraud Cases:** 492  
- **Features:**  
  - Time  
  - V1 – V28 (PCA transformed features)  
  - Amount  
  - Class (Target: 0 = Legit, 1 = Fraud)

> The dataset is **not included in this GitHub repository** because it exceeds GitHub’s 100MB file limit.  
> Please download `creditcard.csv` from Kaggle and place it in the project folder before running the training script.


Author
<br>
Ankit Bisht
