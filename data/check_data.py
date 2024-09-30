import pandas as pd
from src.data_preprocessing import load_data, preprocess_data

# Load the data
data = load_data('data/bank_transactions.csv')

# Step 1: Get unique locations
unique_locations = data['location'].unique()

# Step 2: Loop through each location to find max and min values
for location in unique_locations:
    location_data = data[data['location'] == location]  # Filter by location
    
    if not location_data.empty:  # Check if there are transactions for that location
        max_value = location_data['amount'].max()
        min_value = location_data['amount'].min()
        
        print(f"\nLocation: {location}")
        print(f"Maximum amount: {max_value}")
        print(f"Minimum amount: {min_value}")
    else:
        print(f"\nLocation: {location} has no transactions.")
