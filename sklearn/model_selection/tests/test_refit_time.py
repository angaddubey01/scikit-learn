import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def test_refit_time_recorded_gridsearch():
    X, y = make_classification(n_samples=50, n_features=4,
                               n_informative=2, random_state=0)
    clf = GridSearchCV(DecisionTreeClassifier(random_state=0),
                       param_grid={'max_depth': [1, 2]}, cv=2)
    clf.fit(X, y)
    assert hasattr(clf, 'refit_time_')
    assert isinstance(clf.refit_time_, float)
    assert clf.refit_time_ >= 0.0


def test_refit_time_recorded_randomizedsearch():
    X, y = make_classification(n_samples=50, n_features=4,
                               n_informative=2, random_state=0)
    clf = RandomizedSearchCV(DecisionTreeClassifier(random_state=0),
                             param_distributions={'max_depth': [1, 2]},
                             n_iter=1, cv=2, random_state=0)
    clf.fit(X, y)
    assert hasattr(clf, 'refit_time_')
    assert isinstance(clf.refit_time_, float)
    assert clf.refit_time_ >= 0.0
