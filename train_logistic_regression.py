import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
import joblib

RANDOM_STATE = 42

def load_data(path="D:\Projects\CreditCardFraud\creditcard.csv"):
    """
    Load the Kaggle Credit Card Fraud dataset.
    Assumes 'Class' is the target column: 0 = legit, 1 = fraud.
    """
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def train_test_split_data(X, y):
    """
    Stratified split so that class imbalance is preserved in train & test.
    """
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

def build_logistic_model():
    """
    Builds a pipeline:
      - StandardScaler for feature scaling
      - LogisticRegression with class_weight='balanced' to handle imbalance
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate Logistic Regression model and print metrics.
    Returns AUC score.
    """
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (fraud)
    y_pred = (y_proba >= 0.5).astype(int)        # threshold at 0.5

    auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("\n===== Logistic Regression Evaluation =====")
    print(f"AUC-ROC     : {auc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return auc

def main():
    print("🔹 Loading data...")
    X, y = load_data("creditcard.csv")
    print(f"Dataset shape: {X.shape}, Fraud cases: {y.sum()}, Legit: {(y == 0).sum()}")

    print("\n🔹 Splitting train and test...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("\n🔹 Building Logistic Regression model...")
    log_reg_model = build_logistic_model()

    print("\n🔹 Training model (this may take a few seconds)...")
    log_reg_model.fit(X_train, y_train)

    print("\n🔹 Evaluating model on test set...")
    auc = evaluate_model(log_reg_model, X_test, y_test)

    # Save model as a single pipeline (scaler + LR)
    model_path = "logistic_model.pkl"
    joblib.dump(log_reg_model, model_path)
    print(f"\n✅ Model saved to '{model_path}' with AUC = {auc:.4f}")

if __name__ == "__main__":
    main()
