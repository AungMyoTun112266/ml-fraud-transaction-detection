import joblib
import pandas as pd

def predict_transactions(new_transactions_data):
    # Load the trained model
    model = joblib.load('models/fraud_detection_model.pkl')

    # Make predictions
    predictions = model.predict(new_transactions_data)
    return predictions

def analyze_fraud_predictions(new_transactions_data, predictions):
    fraud_transactions = new_transactions_data[predictions == 1]
    print("\nFraudulent Transactions:\n", fraud_transactions)

if __name__ == "__main__":
    # Example new transaction data
    new_transactions_data = pd.DataFrame({
        'amount': [3000, 5000, 1000],
        'transaction_type': ['withdrawal', 'deposit', 'withdrawal'],
        'account_age': [9, 10, 1],
        'location': ['New York', 'Los Angeles', 'Myanmar']
    })

    # Use the prediction function
    predictions = predict_transactions(new_transactions_data)

    for i, prediction in enumerate(predictions, 1):
        print(f"Prediction for Transaction {i} (1=fraud, 0=not fraud): {prediction}")

    # Analyze fraud predictions
    analyze_fraud_predictions(new_transactions_data, predictions)
