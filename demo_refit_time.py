"""
Demo script for showing how the refit_time_ attribute would work
"""
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load data
X, y = load_iris(return_X_y=True)

# Setup GridSearchCV
param_grid = {'n_estimators': [2, 3, 4, 5]}
rs = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=3
)

# Time the entire fit process
start_time = time.time()
rs.fit(X, y)
total_time = time.time() - start_time

# Print the fit times
print("Mean fit time per fold for each parameter combination:")
for i, params in enumerate(rs.cv_results_['params']):
    print(f"  {params}: {rs.cv_results_['mean_fit_time'][i]:.4f} seconds")

print("\nMean score time per fold for each parameter combination:")
for i, params in enumerate(rs.cv_results_['params']):
    print(f"  {params}: {rs.cv_results_['mean_score_time'][i]:.4f} seconds")

print(f"\nTotal time to run GridSearchCV: {total_time:.4f} seconds")

# In our implementation, we would have access to:
print("\nWith our implementation, we would have:")
print(f"Time to refit best model: rs.refit_time_")
print("This would show how long it takes to fit the best model on the full dataset")