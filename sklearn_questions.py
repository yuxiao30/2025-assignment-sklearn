"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """K-nearest neighbors classifier (Euclidean distance, uniform weights)."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Store training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y=y, reset=True)

        if not isinstance(self.n_neighbors, (int, np.integer)):
            raise TypeError("n_neighbors must be an integer.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be >= 1.")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_test_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Predicted labels.
        """
        check_is_fitted(self, attributes=["X_", "y_", "classes_"])
        X = validate_data(self, X, reset=False)

        n_train = self.X_.shape[0]
        k = min(int(self.n_neighbors), n_train)

        # Distances shape: (n_test, n_train)
        distances = pairwise_distances(X, self.X_, metric="euclidean")
        neigh_idx = np.argsort(distances, axis=1)[:, :k]
        neigh_y = self.y_[neigh_idx]

        # Majority vote with deterministic tie-break:
        # choose the smallest class (same as argmax over bincount).
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        neigh_int = np.vectorize(class_to_int.get, otypes=[int])(neigh_y)

        n_classes = self.classes_.shape[0]
        pred_int = np.empty(neigh_int.shape[0], dtype=int)
        for i in range(neigh_int.shape[0]):
            counts = np.bincount(neigh_int[i], minlength=n_classes)
            pred_int[i] = int(np.argmax(counts))

        return self.classes_[pred_int]

    def score(self, X, y):
        """Return mean accuracy on (X, y)."""
        X, y = validate_data(self, X, y=y, reset=False)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """Cross-validator that trains on month M and tests on month M+1.

    Parameters
    ----------
    time_col : str, default='index'
        If 'index', uses X.index (must be datetime-like).
        Otherwise uses X[time_col] (must be datetime-like).
    """

    def __init__(self, time_col="index"):  # noqa: D107
        self.time_col = time_col

    def __repr__(self):
        # Test expects EXACT string formatting with single quotes.
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _get_time_values(self, X):
        """Return datetime-like values used for splitting."""
        if self.time_col == "index":
            time_values = X.index
        else:
            time_values = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(time_values):
            raise ValueError("The time column/index must be datetime type.")
        return time_values

    def get_n_splits(self, X, y=None, groups=None):
        """Return number of splits (n_months - 1)."""
        time_values = self._get_time_values(X)
        months = pd.PeriodIndex(time_values, freq="M").unique().sort_values()
        return max(len(months) - 1, 0)

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, test_idx) for successive months."""
        time_values = self._get_time_values(X)
        months = pd.PeriodIndex(time_values, freq="M")
        unique_months = months.unique().sort_values()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            idx_train = np.flatnonzero(months == train_month)
            idx_test = np.flatnonzero(months == test_month)

            yield idx_train, idx_test