from __future__ import annotations

import platform
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

"""
Feature Engineering: Transform, Binning Temporal, Image Feature selection. Also feature extraction comes into play here.
"""

if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")


def is_df(obj: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Returns True if `obj` is a pandas DataFrame.
    Note that this function is simply doing `isinstance(obj, pd.DataFrame)`.
    Using that `isinstance` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of `is_df`.

    Args:
        obj (Object): Object to test
    Example::

        >>> x = pd.DataFrame()
        >>> is_df(x)
        True
    """
    return isinstance(obj, pd.DataFrame)


def get_num_vars(df: Union[pd.DataFrame, pd.Series]) -> list:
    """
    Returns the list of numerical features in a DataFrame or Series object.

    Parameters:
    -----------
    df : pandas DataFrame or Series object
        The input DataFrame or Series object to extract the numerical features from.

    Returns:
    --------
    list
        The list of numerical feature column names in the input DataFrame or Series object.
    """
    if not is_df(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    num_vars = df.select_dtypes(include=np.number).columns.tolist()

    return num_vars


def describe_df(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Describes a DataFrame or Series and returns a DataFrame with the descriptions.

    Parameters:
    -----------
    df : pandas DataFrame or Series object
        The input DataFrame or Series object to describe.

    Returns:
    --------
    pandas DataFrame
        A DataFrame containing the descriptions of the input DataFrame or Series.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    with tqdm(
        total=len(df.columns), desc="Describing DataFrame", unit="column"
    ) as pbar:
        # Handle numerical features
        numeric_descr = [
            df[col].describe().apply("{0:.3f}".format)
            for col in df.select_dtypes(include=np.number).columns
        ]
        # Handle categorical features
        non_numeric_descr = [
            df[col].describe().apply(str)
            for col in df.select_dtypes(exclude=np.number).columns
        ]

        descr = numeric_descr + non_numeric_descr
        pbar.update(len(df.columns))

    result_df = pd.concat(descr, axis=1)

    return result_df


def get_cat_vars(df: Union[pd.DataFrame, pd.Series]) -> list:
    """
    Returns the list of categorical features in a DataFrame or Series object.

    Parameters:
    -----------
    df : pandas DataFrame or Series object
        The input DataFrame or Series object to extract the categorical features from.

    Returns:
    --------
    list
        The list of categorical feature column names in the input DataFrame or Series object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame or Series")

    cat_vars = df.select_dtypes(include="object").columns.tolist()

    return cat_vars


def get_cat_counts(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Gets the unique count of categorical features.

    Parameters:
        df: pandas DataFrame
            The input dataframe containing categorical features.
    Returns:
        pandas DataFrame
            Unique value counts of the categorical features in the dataframe.
    """

    if not isinstance(df):
        raise TypeError("df must be a pandas DataFrame")

    cat_vars = get_cat_vars(df)
    counts = {var: df[var].value_counts().shape[0] for var in cat_vars}
    return pd.DataFrame(
        {"Feature": list(counts.keys()), "Unique Count": list(counts.values())}
    )


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical columns in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame containing categorical columns to encode.
    columns : list of str
        The list of column names to perform one-hot encoding.

    Returns
    -------
    pandas DataFrame
        The DataFrame with one-hot encoded columns.
    """
    if not is_df(df):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(cols, list):
        raise TypeError("columns must be a list of column names")

    df[cols] = df[cols].astype("category")

    encoder = OneHotEncoder()
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[cols]))
    encoded_cols.columns = encoder.get_feature_names_out(cols)

    df = df.drop(cols, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)

    return df

def target_encode(df: pd.DataFrame, col:str, target_column:str) -> pd.DataFrame:
    """
    Perform target encoding on a categorical column in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame containing the categorical column and the target column.
    col : str
        The name of the categorical column to encode.
    target_column : str
        The name of the target column.

    Returns
    -------
    pandas DataFrame
        The DataFrame with the categorical column replaced by target encoding.
    """

    if not is_df(df):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(col, str) or not isinstance(target_column, str):
        raise TypeError("col and target_column must be strings")

    target_mean = df.groupby(col)[target_column].mean()
    df[col + '_encoded'] = df[col].map(target_mean)

    return df


def plot_feature_importance(
    estimator: object, feature_names: List[str], show_plot: bool = True
) -> Optional[plt.Figure]:
    """
    Plots the feature importance from a trained scikit-learn estimator
    as a bar chart.

    Parameters:
    -----------
    estimator : scikit-learn estimator
        A fitted estimator that has a `feature_importances_` attribute.
    feature_names : list of str
        The names of the columns in the same order as the feature importances.
    show_plot : bool, optional (default=True)
        Whether to display the plot immediately.

    Returns:
    --------
    fig : matplotlib Figure or None
        The figure object containing the plot or None if show_plot is False.
    """
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            "The estimator does not have a 'feature_importances_' attribute."
        )
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != estimator.n_features_
    ):
        raise ValueError(
            "The 'feature_names' argument should be a list of the same length as the number of features."
        )

    feature_importances = estimator.feature_importances_
    feature_importances_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    )
    feature_importances_df = feature_importances_df.sort_values(
        by="importance", ascending=False
    )

    fig, ax = plt.subplots()
    sns.barplot(x="importance", y="feature", data=feature_importances_df, ax=ax)
    ax.set_title("Feature importance plot")

    if show_plot:
        plt.show()
    else:
        return fig


def feature_summary(
    df: Union[pd.DataFrame, pd.Series], visualize: bool = False
) -> pd.DataFrame:
    """
    Provides a summary of the features in a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame to summarize.
    visualize : bool, optional
        Whether to generate visualizations or not, by default False.

    Returns
    -------
    pandas DataFrame
        The summary DataFrame with columns for the number of null values, unique value counts, data types,
        maximum and minimum values, mean, standard deviation, and skewness.
    """
    if df is None:
        raise ValueError("Expected a pandas dataframe, but got None")

    if not isinstance(df, pd.DataFrame):
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

            if visualize and pd.api.types.is_numeric_dtype(df[col]):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].hist(df[col])
                ax[0].set_xlabel(col)
                ax[0].set_ylabel("Frequency")
                ax[1].boxplot(df[col], vert=False)
                ax[1].set_xlabel(col)
                plt.show()
            elif visualize and pd.api.types.is_categorical_dtype(df[col]):
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                plt.show()

        summary_df.at[col, "Null"] = df[col].isnull().sum()

    return summary_df


def display_missing(
    df: pd.DataFrame,
    plot: bool = False,
    exclude_zero: bool = False,
    sort_by: str = "missing_count",
    ascending: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Display missing values in a pandas DataFrame as a DataFrame or a heatmap.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame to analyze.
    plot : bool, default False
        Whether to display the missing values as a heatmap or not.
    exclude_zero : bool, default False
        Whether to exclude features with zero missing values or not.
    sort_by : str, default 'missing_count'
        Whether to sort the features by missing counts or missing percentages.
    ascending : bool, default False
        Whether to sort the features in ascending or descending order.

    Returns
    -------
    pandas DataFrame or None
        If plot=False, returns a DataFrame with the missing counts and percentages for each feature.
        If plot=True, returns None and displays a heatmap of the missing values.

    """
    if df is None:
        raise ValueError("Expected a pandas DataFrame, but got None")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    dfs = (
        df.isna()
        .sum()
        .to_frame(name="missing_count")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    dfs["missing_percent"] = dfs["missing_count"] / len(df) * 100

    if exclude_zero:
        dfs = dfs[dfs["missing_count"] > 0]

    if sort_by == "missing_percent":
        dfs = dfs.sort_values(by="missing_percent", ascending=ascending)
    else:
        dfs = dfs.sort_values(by="missing_count", ascending=ascending)

    # Display heatmap if plot=True
    if plot:
        plt.figure(figsize=(12, 6))
        plt.title("Missing Values Heatmap")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        sns.heatmap(dfs.isna(), cmap="Reds", cbar=False)
        plt.show()
    else:
        return dfs


def get_unique_counts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the unique count of categorical features in a data set.

    Parameters
    -----------
    data: DataFrame or named Series
        The dataset for which to calculate unique value counts.

    Returns
    -------
    DataFrame
        Unique value counts of the features in the dataset.
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if not is_df(data, (pd.DataFrame)):
        raise TypeError(
            "data: Expecting a DataFrame or Series, got '{}'".format(type(data))
        )

    if isinstance(data, pd.Series):
        data = data.to_frame()

    features = data.select_dtypes(include="object").columns.tolist()
    unique_counts = data[features].nunique().reset_index()
    unique_counts.columns = ["Feature", "Unique Count"]

    return unique_counts


def join_train_and_test(data_train: pd.DataFrame, data_test: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Joins two data sets and returns a dictionary containing their sizes and the concatenated data.
    Used mostly before feature engineering to combine train and test set together.

    Parameters:
    ----------
    data_train: pd.DataFrame
        First dataset, usually called "train_data", to join.

    data_test: pd.DataFrame
        Second dataset, usually called "test_data", to join.

    Returns:
    -------
    pd.DataFrame
        Merged data containing both train and test sets.
    int
        Number of rows in the train set.
    int
        Number of rows in the test set.
    """

    if data_train is None or data_test is None:
        raise ValueError("Both 'data_train' and 'data_test' must be provided.")

    if not is_df(data_train, pd.DataFrame) or not is_df(
        data_test, pd.DataFrame
    ):
        raise TypeError("Both 'data_train' and 'data_test' should be DataFrames.")

    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    all_data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

    return all_data, n_train, n_test


def check_train_test_set(train_data: pd.DataFrame, test_data: pd.DataFrame, index: Optional[str] = None, col: Optional[str] = None) -> None:
    """
    Checks the distribution of train and test for uniqueness to determine
    the best feature engineering strategy.

    Parameters:
    -------------------
    train_data: pd.DataFrame
        The train dataset.

    test_data: pd.DataFrame
        The test dataset.

    index: str, Default None
        An index column present in both datasets to be used in plotting.

    col: str, Default None
        A feature present in both datasets used in plotting.
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
