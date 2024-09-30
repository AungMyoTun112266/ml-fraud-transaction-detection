import pandas as pd
import random

def generate_data(num_samples):
    data = {
        'amount': [random.uniform(10, 10000) for _ in range(num_samples)],
        'transaction_type': [random.choice(['withdrawal', 'deposit']) for _ in range(num_samples)],
        'account_age': [random.randint(1, 10) for _ in range(num_samples)],
        'location': [random.choice(['New York', 'Los Angeles', 'Chicago', 'Myanmar']) for _ in range(num_samples)],
        'is_fraud': [random.choice([0, 1]) for _ in range(num_samples)]  # 1 = fraud, 0 = not fraud
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data(1000)
    df.to_csv('data/bank_transactions.csv', index=False)
    print("Synthetic data generated and saved to bank_transactions.csv")
