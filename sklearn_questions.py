import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """K-nearest neighbors classifier (Euclidean distance, uniform weights)."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Store training data.

        Raises a ValueError when passed a continuous target.
        """
        X, y = validate_data(self, X, y=y, reset=True)

        if not isinstance(self.n_neighbors, (int, np.integer)):
            raise TypeError("n_neighbors must be an integer.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be >= 1.")

        target_type = type_of_target(y)
        if target_type == "continuous":
            raise ValueError(
                "continuous target is not supported for classification"
            )
        if target_type not in ("binary", "multiclass"):
            raise ValueError(f"Unknown label type: {target_type}")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self, attributes=["X_", "y_", "classes_"])
        X = validate_data(self, X, reset=False)

        n_train = self.X_.shape[0]
        k = min(int(self.n_neighbors), n_train)

        distances = pairwise_distances(X, self.X_, metric="euclidean")
        neigh_idx = np.argsort(distances, axis=1)[:, :k]
        neigh_y = self.y_[neigh_idx]

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
        """Return repr used by tests."""
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
