import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the split data
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# Initialize and fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled datasets and the scaler object
X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl') # Save for later use in production

print("Data normalization completed.")
