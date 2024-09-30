import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from src.data_preprocessing import load_data, preprocess_data

def train_model():
    # Load the data
    df = load_data('data/bank_transactions.csv')

    # Preprocess the data
    (X_train, X_test, y_train, y_test), preprocessor = preprocess_data(df)

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, 'models/fraud_detection_model.pkl')
    print("Model trained and saved to fraud_detection_model.pkl")

if __name__ == "__main__":
    train_model()
