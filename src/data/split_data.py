import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load data
df = pd.read_csv('data/raw_data/raw.csv')

# Remove non-feature columns 
columns_to_drop = ['date']  # other non-feature columns can be added here
df_clean = df.drop(columns=columns_to_drop, errors='ignore')  # 'errors=ignore' prevents errors if column doesn't exist

# Define features (X) and target (y). Target 'silica_concentrate' is the last column.
X = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the output directory if it doesn't exist
os.makedirs('data/processed_data', exist_ok=True)

# Save the splits
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("Data splitting completed. Removed columns:", columns_to_drop)
print("Data splitting completed.")
