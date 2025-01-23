from __future__ import annotations

import platform
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# matplotlib backend
if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")


def get_num_vars(df: Union[pd.DataFrame, pd.Series]) -> List[str]:
    """Get a list of numeric columns in a DataFrame or Series.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        The input DataFrame or Series.

    Returns:
    -------
    List[str]
        A list of numeric column names.

    Raises:
    ------
    TypeError
        If the input is not a DataFrame or Series.
    ValueError
        If the input DataFrame or Series is empty.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series.")
    if df.empty:
        raise ValueError("Input DataFrame or Series is empty.")

    return df.select_dtypes(include=np.number).columns.tolist()


def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary of descriptive statistics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics for each column.

    Raises:
    ------
    TypeError
        If the input is not a DataFrame.
    ValueError
        If the input DataFrame is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    with tqdm(
        total=len(df.columns), desc="Describing DataFrame", unit="column"
    ) as pbar:
        numeric_descr = [
            df[col].describe().apply("{0:.3f}".format)
            for col in df.select_dtypes(include=np.number).columns
        ]
        non_numeric_descr = [
            df[col].describe().apply(str)
            for col in df.select_dtypes(exclude=np.number).columns
        ]

        descr = numeric_descr + non_numeric_descr
        pbar.update(len(df.columns))

    return pd.concat(descr, axis=1)


def get_cat_vars(df: Union[pd.DataFrame, pd.Series]) -> List[str]:
    """Get a list of categorical columns in a DataFrame or Series.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        The input DataFrame or Series.

    Returns:
    -------
    List[str]
        A list of categorical column names.

    Raises:
    ------
    TypeError
        If the input is not a DataFrame or Series.
    ValueError
        If the input DataFrame or Series is empty.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series.")
    if df.empty:
        raise ValueError("Input DataFrame or Series is empty.")

    return df.select_dtypes(include="object").columns.tolist()


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Perform one-hot encoding on specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    cols : List[str]
        A list of categorical column names to encode.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with one-hot encoded columns.

    Raises:
    ------
    TypeError
        If the input is not a DataFrame or `cols` is not a list.
    ValueError
        If the input DataFrame is empty or `cols` contains invalid column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(cols, list):
        raise TypeError("`cols` must be a list of column names.")
    if not all(col in df.columns for col in cols):
        raise ValueError("One or more columns in `cols` do not exist in the DataFrame.")

    encoder = OneHotEncoder(sparse=False, drop="first")
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[cols]))
    encoded_cols.columns = encoder.get_feature_names_out(cols)

    df = df.drop(cols, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)

    return df


def target_encode(df: pd.DataFrame, col: str, target_column: str) -> pd.DataFrame:
    """Perform target encoding on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The categorical column to encode.
    target_column : str
        The target column used for encoding.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the target-encoded column.

    Raises:
    ------
    TypeError
        If the input is not a DataFrame or `col`/`target_column` are not strings.
    ValueError
        If the input DataFrame is empty or `col`/`target_column` do not exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(col, str) or not isinstance(target_column, str):
        raise TypeError("`col` and `target_column` must be strings.")
    if col not in df.columns or target_column not in df.columns:
        raise ValueError("`col` or `target_column` do not exist in the DataFrame.")

    target_mean = df.groupby(col)[target_column].mean()
    df[f"{col}_encoded"] = df[col].map(target_mean)

    return df
