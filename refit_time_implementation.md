# Implementation of `refit_time_` Attribute in BaseSearchCV

## Overview

This implementation adds a new attribute `refit_time_` to the BaseSearchCV class (and therefore to both GridSearchCV and RandomizedSearchCV) that measures the time it takes to refit the best model on the full dataset after hyperparameter optimization.

## Changes Made

1. Added import of `time` module to `sklearn/model_selection/_search.py`
2. Added timing code around the refit operation in the `fit` method of `BaseSearchCV`
3. Updated documentation for both `GridSearchCV` and `RandomizedSearchCV` to include the new attribute

## Usage Example

```python
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

# Fit the model
rs.fit(X, y)

# Access the refit time
print(f"Time to refit best model: {rs.refit_time_:.4f} seconds")

# Compare with other timing information
print(f"Mean fit time per fold: {np.mean(rs.cv_results_['mean_fit_time']):.4f} seconds")
print(f"Mean score time per fold: {np.mean(rs.cv_results_['mean_score_time']):.4f} seconds")
```

## Implementation Details

The `refit_time_` attribute is set during the `fit` method of `BaseSearchCV`, specifically when `refit=True`. The timing is done using the Python `time` module, measuring the wall clock time it takes to execute:

```python
if self.refit:
    self.best_estimator_ = clone(base_estimator).set_params(
        **self.best_params_)
    refit_start_time = time.time()
    if y is not None:
        self.best_estimator_.fit(X, y, **fit_params)
    else:
        self.best_estimator_.fit(X, **fit_params)
    self.refit_time_ = time.time() - refit_start_time
```

The attribute is only available when `refit=True`, since otherwise no refitting is performed.

## Use Case

This feature is particularly useful for:

1. Benchmarking and profiling hyperparameter optimization workflows
2. Understanding the breakdown of time between cross-validation and final model training
3. Services like OpenML.org that need to report timing information for hyperparameter optimization

It allows users to distinguish between:
- Time spent on hyperparameter optimization (already available via `mean_fit_time` and `mean_score_time`)
- Time spent on training the final model on the full dataset (new `refit_time_` attribute)