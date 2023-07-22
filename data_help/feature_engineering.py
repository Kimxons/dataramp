import platform
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# check for macOs - plt might misbehave
if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")

# pre-processing data - remove noise andd missing data


def drop_missing(data: Union[pd.DataFrame, pd.Series], threshold=95) -> Union[pd.DataFrame, pd.Series]:
    """
    Drops missing columns with a threshold of missing data.

    Parameters:
        data: pandas DataFrame or Series, default None
            The input DataFrame or Series.
        threshold: float, default 95
            The percentage of missing values to be in a column before it is eligible for removal.
    Returns:
        pandas DataFrame or Series
            The modified DataFrame or Series after dropping the missing columns.
    """
    if data is None:
        data = pd.DataFrame()

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"data must be a pandas DataFrame or Series, but got {type(data)}")

    missing_data = data.isna().mean() * 100
    cols_to_drop = missing_data[missing_data >= threshold].index

    if not cols_to_drop.empty:
        n_cols_dropped = len(cols_to_drop)
        n_cols_orig = data.shape[1]
        print(f"Dropped {n_cols_dropped}/{n_cols_orig} ({n_cols_dropped/n_cols_orig:.1%}) columns.")
        data = data.drop(columns=cols_to_drop, axis=1)

    return data


def min_max_scaling(X: np.ndarray) -> np.ndarray:
    """
    Apply min-max scaling to a numpy array.

    Parameters
    ----------
    X : numpy ndarray, shape (n_samples, n_features)
        The data to be scaled.

    Returns
    -------
    X_scaled : numpy ndarray, shape (n_samples, n_features)
        The scaled data.
    """
    if X is None:
        raise ValueError("Expected a numpy array, but got None")

    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(X)}")

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled


def z_score_normalization(X: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to a numpy array.

    Parameters
    ----------
    X : numpy ndarray, shape (n_samples, n_features)
        The data to be normalized.

    Returns
    -------
    X_normalized : numpy ndarray, shape (n_samples, n_features)
        The normalized data.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    return X_normalized


def log_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply log transformation to a numpy array.

    Parameters
    ----------
    X : numpy ndarray, shape (n_samples, n_features)
        The data to be transformed.

    Returns
    -------
    X_transformed : numpy ndarray, shape (n_samples, n_features)
        The transformed data.
    """
    if X is None:
        raise ValueError("Expected a numpy array, but got None")
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(X)}")
    X_transformed = np.log(X)
    return X_transformed


def detect_outliers(data: Union[pd.DataFrame, pd.Series], features, n=2):
    """
    Detects rows with outliers using Tukey's Interquartile Range (IQR) method.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The input data to be checked for outliers.
    features : list of str or None
        The list of features (columns) to check for outliers. If None, all columns will be checked.
    n : int, optional
        The maximum number of outliers allowed per feature. Default is 2.

    Returns
    -------
    pandas.Index
        The indices of rows containing outliers.
    """
    if data is None:
        raise ValueError("Expected a pandas dataframe or series, but got None")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"data must be a pandas DataFrame or Series, but got {type(data)}")

    if features is not None:
        if not isinstance(features, list):
            raise TypeError(f"features must be a list, but got {type(features)}")
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Column '{feature}' not found in data")
    else:
        features = data.columns

    outlier_indices = []

    for col in features:
        Q1, Q3 = np.nanpercentile(data[col], [25, 75])
        IQR = Q3 - Q1

        outlier_list_col = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = pd.Index(outlier_indices)
    outlier_indices = outlier_indices[outlier_indices.duplicated(keep=False)]
    outlier_indices_counts = outlier_indices.value_counts()
    multiple_outliers = outlier_indices_counts[outlier_indices_counts > n].index

    return multiple_outliers
