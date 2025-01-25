"""Module containing outlier detection functionality.

This module provides classes for detecting outliers in data using various statistical methods,
including interquartile range and interval-based detection.
"""

import logging
import numbers
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class OutlierDetector:
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
        self._fit(x, y)
        self._is_fitted = True
        return self

    def _fit(self, x, y=None):
        """Internal method for fitting the outlier detector."""
        raise NotImplementedError("Subclasses must implement _fit method.")

    def get_outliers(
        self, indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get indices or mask of outliers.

        Parameters
        ----------
        indices : bool, optional (default=False)
            If True, return an array of integers; otherwise, return a boolean mask.

        Returns:
        -------
        outliers : np.ndarray or Tuple[np.ndarray, np.ndarray]
            Array of indices or boolean mask indicating outliers.
        """
        return (np.where(self._support)[0],) if indices else self._support


class RangeDetector(BaseEstimator, OutlierDetector):
    r"""Interquartile range or interval-based outlier detection method.

    The default settings compute the usual interquartile range method.

    Parameters
    ----------
    interval_length : float, optional (default=0.5)
        Compute ``interval_length``\% credible interval. This is a value in [0, 1].
    k : float, optional (default=1.5)
        Tukey's factor.
    method : str, optional (default="ETI")
        Method to compute credible intervals. Supported methods are Highest
        Density interval (``method="HDI"``) and Equal-tailed interval
        (``method="ETI"``).
    """

    def __init__(
        self, interval_length: float = 0.5, k: float = 1.5, method: str = "ETI"
    ):
        self.interval_length = interval_length
        self.k = k
        self.method = method

    def _fit(self, x, y=None):
        if self.method not in ("ETI", "HDI"):
            raise ValueError(
                "Invalid value for method. Allowed string "
                'values are "ETI" and "HDI".'
            )

        if (
            not isinstance(self.interval_length, numbers.Number)
            or not 0 <= self.interval_length <= 1
        ):
            raise ValueError(
                f"Interval length must a value in [0, 1]; got {self.interval_length}."
            )

        if self.method == "ETI":
            lower = 100 * (1 - self.interval_length) / 2
            upper = 100 * (1 + self.interval_length) / 2

            lb, ub = np.percentile(x, [lower, upper])
        else:
            n = len(x)
            xsorted = np.sort(x)
            n_included = int(np.ceil(self.interval_length * n))
            n_ci = n - n_included
            ci = xsorted[n_included:] - xsorted[:n_ci]
            j = np.argmin(ci)
            hdi_min = xsorted[j]
            hdi_max = xsorted[j + n_included]

            lb = hdi_min
            ub = hdi_max

        iqr = ub - lb
        lower_bound = lb - self.k * iqr
        upper_bound = ub + self.k * iqr

        self._support = (x > upper_bound) | (x < lower_bound)
