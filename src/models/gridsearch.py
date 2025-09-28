import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load the training data
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').squeeze('columns')

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 200],
    'max_depth': [5, 20]
}

# Initialize the model and GridSearchCV
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train_scaled, y_train)

# Save the best parameters found
best_params = grid_search.best_params_
joblib.dump(best_params, 'models/best_params.pkl')

print("GridSearch completed. Best parameters:", best_params)
