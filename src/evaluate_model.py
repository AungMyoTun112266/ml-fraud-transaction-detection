import joblib
from sklearn.metrics import classification_report
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data

def evaluate_model():
    # Load the data
    df = load_data('data/bank_transactions.csv')

    # Preprocess the data
    (X_train, X_test, y_train, y_test), preprocessor = preprocess_data(df)

    # Load the trained model
    model = joblib.load('models/fraud_detection_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)

    # Output evaluation metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
