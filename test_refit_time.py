"""
Test script to verify the refit_time_ attribute in GridSearchCV
"""
import time
import numpy as np

def test_refit_time():
    # Create some test data
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    
    # Manually import the components to avoid package build issues
    from sklearn.model_selection._search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a GridSearchCV object
    param_grid = {'n_estimators': [2, 3, 4, 5]}
    clf = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=3,
        return_train_score=True
    )
    
    # Fit the model
    clf.fit(X, y)
    
    # Print the refit time
    print("Time to refit best model: {:.4f} seconds".format(clf.refit_time_))
    
    # Also print other timing information for comparison
    print("Mean fit time for all candidates: {:.4f} seconds".format(
        np.mean(clf.cv_results_['mean_fit_time'])))
    print("Mean score time for all candidates: {:.4f} seconds".format(
        np.mean(clf.cv_results_['mean_score_time'])))
    
    # Verify the attribute is set properly
    assert hasattr(clf, 'refit_time_')
    assert clf.refit_time_ > 0
    
    print("All tests passed!")

if __name__ == "__main__":
    test_refit_time()