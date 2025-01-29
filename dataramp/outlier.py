"""Module containing outlier detection functionality.

This module provides classes for detecting outliers in data using various statistical methods,
including interquartile range and interval-based detection.
"""

import logging
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Base class for all outlier detectors."""

    def __init__(self):
        self._support = None
        self._is_fitted = False

    @property
    def support(self):
        """Outlier support mask."""
        if not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        return self._support.copy()

    def fit(self, x, y=None):
        """Fit outlier detector.

        Parameters
        ----------
        x : array-like, shape=(n_samples)
            Input data.
        y : array-like, shape=(n_samples), optional (default=None)
            Additional target variable.

        Returns:
        -------
        self : OutlierDetector
            Returns an instance of the outlier detector.
        """
        x = self._validate_input(x)
        self._fit(x)
        self._is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform input data by flagging/marking outliers.

        Parameters
        ----------
        x : np.ndarray, shape=(n_samples)
            Input data.

        Returns:
        -------
        outliers : np.ndarray
            Boolean mask indicating outliers.
        """
        if not self._is_fitted:
            raise NotFittedError("Call 'fit' before transforming data.")
        x = self._validate_input(x)
        return self._detect_outliers(x)

    def fit_predict(self, x: np.ndarray, y=None) -> np.ndarray:
        """Fit detector and return outlier mask."""
        return self.fit(x).transform(x)

    def get_outliers(
        self, indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Get indices or mask of outliers.

        Parameters
        ----------
        indices : bool, optional (default=False)
            If True, return an array of indices; otherwise, return a boolean mask.

        Returns:
        -------
        outliers : np.ndarray or Tuple[np.ndarray]
            Array of indices or boolean mask indicating outliers.
        """
        if not self._is_fitted:
            raise NotFittedError("Call 'fit' before getting outliers.")
        return np.where(self._support)[0] if indices else self._support

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validates input data."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.size == 0:
            raise ValueError("Input array is empty.")
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Input must be numeric.")
        return x

    def _fit(self, x: np.ndarray):
        """Internal method for fitting the outlier detector."""
        raise NotImplementedError("Subclasses must implement _fit method.")

    def _detect_outliers(self, x: np.ndarray) -> np.ndarray:
        """Internal method to detect outliers."""
        raise NotImplementedError("Subclasses must implement _detect_outliers method.")


class RangeDetector(OutlierDetector):
    """Interquartile range or interval-based outlier detection method.

    Parameters
    ----------
    interval_length : float, optional (default=0.5)
        Compute ``interval_length``% credible interval. Must be in [0, 1].
    k : float, optional (default=1.5)
        Tukey's factor.
    method : str, optional (default="ETI")
        Method to compute credible intervals. Supported methods: "ETI" (Equal-tailed interval) or "HDI" (Highest Density Interval).

    Examples:
    --------
    >>> import numpy as np
    >>> from dataramp.outlier import RangeDetector
    >>> X = np.array([1, 2, 3, 4, 100])
    >>> detector = RangeDetector(interval_length=0.5, k=1.5, method="ETI")
    >>> detector.fit(X)
    >>> detector.transform(X)
    array([False, False, False, False,  True])

    >>> detector = RangeDetector(interval_length=0.5, k=1.5, method="HDI")
    >>> detector.fit(X)
    >>> detector.transform(X)
    array([False, False, False, False,  True])
    """

    def __init__(
        self, interval_length: float = 0.5, k: float = 1.5, method: str = "ETI"
    ):
        super().__init__()
        self.interval_length = interval_length
        self.k = k
        self.method = method
        self.lower_bound_ = None
        self.upper_bound_ = None

    def _fit(self, x: np.ndarray):
        """Compute bounds for detecting outliers."""
        if self.method not in {"ETI", "HDI"}:
            raise ValueError('Invalid method. Choose "ETI" or "HDI".')

        if not (0 <= self.interval_length <= 1):
            raise ValueError(
                f"Interval length must be in [0, 1]; got {self.interval_length}."
            )

        if self.method == "ETI":
            lb, ub = self._eti_bounds(x)
        else:
            lb, ub = self._hdi_bounds(x)

        iqr = ub - lb
        self.lower_bound_ = lb - self.k * iqr
        self.upper_bound_ = ub + self.k * iqr

        self._support = (x < self.lower_bound_) | (x > self.upper_bound_)
        logger.debug(
            f"Computed bounds: Lower={self.lower_bound_}, Upper={self.upper_bound_}"
        )

    def _eti_bounds(self, x: np.ndarray) -> Tuple[float, float]:
        """Compute Equal-Tailed Interval (ETI) bounds."""
        lower = 100 * (1 - self.interval_length) / 2
        upper = 100 * (1 + self.interval_length) / 2
        return np.percentile(x, [lower, upper])

    def _hdi_bounds(self, x: np.ndarray) -> Tuple[float, float]:
        """Compute Highest Density Interval (HDI) bounds."""
        x_sorted = np.sort(x)
        n = len(x)
        n_included = max(1, int(np.ceil(self.interval_length * n)))

        start_indices = np.arange(n - n_included + 1)
        end_indices = start_indices + (n_included - 1)

        interval_widths = x_sorted[end_indices] - x_sorted[start_indices]
        min_index = np.argmin(interval_widths)

        return x_sorted[start_indices[min_index]], x_sorted[end_indices[min_index]]

    def _detect_outliers(self, x: np.ndarray) -> np.ndarray:
        """Detect outliers based on computed bounds."""
        return (x < self.lower_bound_) | (x > self.upper_bound_)
