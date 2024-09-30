import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Define the features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['transaction_type', 'location']),
        ],
        remainder='passthrough'  # Keep other columns as they are
    )
    
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor
