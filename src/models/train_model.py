import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the processed data
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').squeeze('columns') # Convert to Series

# Load the best hyperparameters
best_params = joblib.load('models/best_params.pkl')

# Train the model with the best parameters
model = RandomForestRegressor(**best_params)
model.fit(X_train_scaled, y_train)

# Save the trained model
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')

print("Model training completed.")
