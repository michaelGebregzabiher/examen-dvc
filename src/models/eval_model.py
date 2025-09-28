import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# Load the model, test data, and target
model = joblib.load('models/model.pkl')
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').squeeze('columns')

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a dictionary of metrics
metrics = {
    'mean_squared_error': mse,
    'r2_score': r2
}

# Save metrics to a JSON file
os.makedirs('metrics', exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('data/predictions.csv', index=False)

print(f"Model evaluation completed. MSE: {mse:.4f}, R2: {r2:.4f}")
