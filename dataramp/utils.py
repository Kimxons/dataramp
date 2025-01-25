"""Utility module for data preprocessing and analysis in pandas.

This module provides various utility functions for working with pandas DataFrames,
including functions for identifying numeric and categorical variables, data encoding,
and descriptive statistics.
"""

from __future__ import annotations

import platform
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def display_missing(
    df: pd.DataFrame,
    plot: bool = False,
    exclude_zero: bool = False,
    sort_by: str = "missing_count",
    ascending: bool = False,
) -> Optional[pd.DataFrame]:
    """Analyze missing values in a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to analyze
    plot : bool, optional
        Whether to generate a missing values heatmap (default: False)
    exclude_zero : bool, optional
        Exclude columns with no missing values (default: False)
    sort_by : str, optional
        Column to sort results by ('missing_count' or 'missing_percent', default: 'missing_count')
    ascending : bool, optional
        Sort in ascending order (default: False)

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with missing value analysis or None if plot is True

    Raises:
    -------
    ValueError: If input DataFrame is None
    TypeError: If input is not a DataFrame or Series
    """
    if df is None:
        raise ValueError("Expected a pandas DataFrame, but got None")

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame")

    missing = df.isna().sum().to_frame(name="missing_count")
    missing["missing_percent"] = missing["missing_count"] / len(df) * 100
    missing = missing.reset_index().rename(columns={"index": "variable"})

    if exclude_zero:
        missing = missing[missing["missing_count"] > 0]

    missing = missing.sort_values(by=sort_by, ascending=ascending)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.title("Missing Values Heatmap")
        sns.heatmap(df.isna(), cmap="Reds", cbar=False)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        return None

    return missing


def feature_summary(
    df: Union[pd.DataFrame, pd.Series], visualize: bool = False
) -> pd.DataFrame:
    """Generate a summary of features in a DataFrame or Series.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        The input DataFrame or Series to analyze.
    visualize : bool, optional
        Whether to display visualizations for each column (default: False).

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing summary statistics for each column.

    Raises:
    ------
    ValueError
        If the input is None.
    TypeError
        If the input is not a DataFrame or Series.
    """
    if df is None:
        raise ValueError("Expected a pandas dataframe, but got None")
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame")
    summary_df = pd.DataFrame(
        index=df.columns,
        columns=[
            "Null",
            "Unique_Count",
            "Data_type",
            "Max",
            "Min",
            "Mean",
            "Std",
            "Skewness",
        ],
    )
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            summary_df.at[col, "Unique_Count"] = df[col].nunique()
            summary_df.at[col, "Data_type"] = "categorical"
        else:
            summary_df.at[col, "Unique_Count"] = df[col].nunique()
            summary_df.at[col, "Data_type"] = str(df[col].dtype)
            summary_df.at[col, "Max"] = df[col].max()
            summary_df.at[col, "Min"] = df[col].min()
            summary_df.at[col, "Mean"] = df[col].mean()
            summary_df.at[col, "Std"] = df[col].std()
            summary_df.at[col, "Skewness"] = df[col].skew()
            if visualize:
                if pd.api.types.is_numeric_dtype(df[col]):
                    _fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].hist(df[col])
                    ax[0].set_xlabel(col)
                    ax[0].set_ylabel("Frequency")
                    ax[1].boxplot(df[col], vert=False)
                    ax[1].set_xlabel(col)
                    plt.show()
                elif pd.api.types.is_categorical_dtype(df[col]):
                    _fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    df[col].value_counts().plot(kind="bar", ax=ax)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    plt.show()
        summary_df.at[col, "Null"] = df[col].isnull().sum()
    return summary_df


def get_unique_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Get the count of unique values for each object (categorical) column in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to analyze.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing feature names and their unique value counts.

    Raises:
    ------
    ValueError
        If the input is None.
    TypeError
        If the input is not a DataFrame or Series.
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Expected a DataFrame or Series, but got '{type(data)}'")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    features = data.select_dtypes(include="object").columns.tolist()
    unique_counts = data[features].nunique().reset_index()
    unique_counts.columns = ["Feature", "Unique Count"]
    return unique_counts


def join_train_and_test(
    data_train: pd.DataFrame, data_test: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """Join training and test DataFrames and return the combined data with counts.

    Parameters
    ----------
    data_train : pd.DataFrame
        The training DataFrame.
    data_test : pd.DataFrame
        The test DataFrame.

    Returns:
    -------
    Tuple[pd.DataFrame, int, int]
        A tuple containing:
        - The combined DataFrame
        - Number of training samples
        - Number of test samples

    Raises:
    ------
    ValueError
        If either data_train or data_test is None.
    TypeError
        If either input is not a DataFrame.
    """
    if data_train is None or data_test is None:
        raise ValueError("Both 'data_train' and 'data_test' must be provided.")
    if not isinstance(data_train, pd.DataFrame) or not isinstance(
        data_test, pd.DataFrame
    ):
        raise TypeError("Both 'data_train' and 'data_test' should be DataFrames.")
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    all_data = pd.concat([data_train, data_test], ignore_index=True, sort=False)
    return all_data, n_train, n_test


def check_train_test_set(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    index: Optional[str] = None,
    col: Optional[str] = None,
) -> None:
    """Check and visualize the relationship between training and test datasets.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset.
    test_data : pd.DataFrame
        The test dataset.
    index : Optional[str]
        The name of the ID column to check for uniqueness and overlap.
    col : Optional[str]
        The column name to use for grouping and visualization.

    Returns:
    -------
    None
        Displays analysis results and a visualization plot.
    """
    if index:
        if train_data[index].nunique() == train_data.shape[0]:
            print("ID field is unique in the training set.")
        else:
            print("ID field is not unique in the training set.")
        if len(np.intersect1d(train_data[index].values, test_data[index].values)) == 0:
            print("Train and test sets have distinct IDs.")
        else:
            print("Train and test sets share some IDs.")
        print("\n")
        plt.plot(train_data.groupby(col).count()[[index]], "o-", label="Train")
        plt.plot(test_data.groupby(col).count()[[index]], "o-", label="Test")
        plt.title("Train and test instances overlap.")
        plt.legend(loc="best")
        plt.xlabel(col)
        plt.ylabel("Number of records")
        plt.show()
