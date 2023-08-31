from __future__ import annotations

import platform
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")


def get_num_vars(df: Union[pd.DataFrame, pd.Series]) -> None:
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
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    num_vars = df.select_dtypes(include=np.number).columns.tolist()

    return num_vars


def get_cat_vars(df: Union[pd.DataFrame, pd.Series]) -> None:
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
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    cat_vars = df.select_dtypes(include="object").columns.tolist()

    return cat_vars


# TODO - Rename one of the get_num_counts or get_cat_counts functions to avoid confusion.

def get_cat_counts(df: Union[pd.DataFrame, pd.Series]) -> None:
    """
    Gets the unique count of categorical features.

    Parameters:
        df: pandas DataFrame
            The input dataframe containing categorical features.
    Returns:
        pandas DataFrame
            Unique value counts of the categorical features in the dataframe.
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    cat_vars = get_cat_vars(df)
    counts = {var: df[var].value_counts().shape[0] for var in cat_vars}
    return pd.DataFrame({"Feature": list(counts.keys()), "Unique Count": list(counts.values())})


def plot_feature_importance(vi: np.ndarray, feature_names: list) -> None:
    """
    Plots the feature importance as a bar chart.

    Parameters:
    -----------
    vi : numpy ndarray
        The feature vi from a trained model.
    feature_names : list of str
        The names of the features in the same order as the vi.

    Returns:
    --------
    None
    """
    if not isinstance(vi, np.ndarray) or not isinstance(feature_names, list):
        raise TypeError("vi should be a numpy ndarray and feature_names should be a list.")

    if len(vi) != len(feature_names):
        raise ValueError("vi and feature_names should have the same length.")

    sorted_indices = vi.argsort()[::-1]
    plt.bar(range(len(vi)), vi[sorted_indices])
    plt.xticks(range(len(vi)), np.array(feature_names)[sorted_indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature vi")
    plt.show()


def feature_summary(df: Union[pd.DataFrame, pd.Series], visualize: bool = False) -> None:
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
        if df[col].dtype == "category":
            summary_df.at[col, "Unique_Count"] = df[col].value_counts().count()
            summary_df.at[col, "Data_type"] = "categorical"
        else:
            summary_df.at[col, "Unique_Count"] = df[col].nunique()
            summary_df.at[col, "Data_type"] = str(df[col].dtype)  # Use str(df[col].dtype)
            summary_df.at[col, "Max"] = df[col].max()  # Remove .astype(str)
            summary_df.at[col, "Min"] = df[col].min()  # Remove .astype(str)
            summary_df.at[col, "Mean"] = df[col].mean()
            summary_df.at[col, "Std"] = df[col].std()
            summary_df.at[col, "Skewness"] = df[col].skew()

            if visualize and df[col].dtype.name in ["int64", "float64"]:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].hist(df[col])
                ax[0].set_xlabel(col)
                ax[0].set_ylabel("Frequency")
                ax[1].boxplot(df[col], vert=False)
                ax[1].set_xlabel(col)
                plt.show()
            elif visualize and df[col].dtype.name == "category":
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
        raise ValueError("Expected a pandas dataframe, but got None")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    df = df.isna().sum().to_frame(name="missing_count")
    df["missing_percent"] = df["missing_count"] / len(df) * 100

    if exclude_zero:
        df = df[df["missing_count"] > 0]

    if sort_by == "missing_percent":
        df = df.sort_values(by="missing_percent", ascending=ascending)
    else:
        df = df.sort_values(by="missing_count", ascending=ascending)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.title("Missing Values Heatmap")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        sns.heatmap(df.isna(), cmap="Reds", cbar=False)
        plt.show()
    else:
        return df


def get_unique_counts(data=None):
    '''
    Gets the unique count of categorical features in a data set.

    Parameters
    -----------
    data: DataFrame or named Series
        The dataset for which to calculate unique value counts.

    Returns
    -------
    DataFrame
        Unique value counts of the features in the dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data: Expecting a DataFrame or Series, got '{}'".format(type(data)))

    if isinstance(data, pd.Series):
        data = data.to_frame()

    features = data.select_dtypes(include='object').columns.tolist()
    unique_counts = data[features].nunique().reset_index()
    unique_counts.columns = ['Feature', 'Unique Count']

    return unique_counts


def join_train_and_test(data_train=None, data_test=None):
    '''
    Joins two data sets and returns a dictionary containing their sizes and the concatenated data.
    Used mostly before feature engineering to combine train and test set together.

    Parameters:
    ----------
    data_train: DataFrame
        First dataset, usually called "train_data", to join.

    data_test: DataFrame
        Second dataset, usually called "test_data", to join.

    Returns:
    -------
    DataFrame
        Merged data containing both train and test sets.
    int
        Number of rows in the train set.
    int
        Number of rows in the test set.
    '''

    if data_train is None or data_test is None:
        raise ValueError("Both 'data_train' and 'data_test' must be provided.")

    if not isinstance(data_train, pd.DataFrame) or not isinstance(data_test, pd.DataFrame):
        raise TypeError("Both 'data_train' and 'data_test' should be DataFrames.")

    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    all_data = pd.concat([data_train, data_test], sort=False).reset_index(drop=True)

    return all_data, n_train, n_test


def check_train_test_set(train_data, test_data, index=None, col=None):
    '''
    Checks the distribution of train and test for uniqueness to determine
    the best feature engineering strategy.

    Parameters:
    -------------------
    train_data: DataFrame
        The train dataset.

    test_data: DataFrame
        The test dataset.

    index: str, Default None
        An index column present in both datasets to be used in plotting.

    col: str, Default None
        A feature present in both datasets used in plotting.
    '''

    if index:
        if train_data[index].nunique() == train_data.shape[0]:
            print('Id field is unique.')
        else:
            print('Id field is not unique')

        if len(np.intersect1d(train_data[index].values, test_data[index].values)) == 0:
            print('Train and test sets have distinct Ids.')
        else:
            print('Train and test sets IDs are the same.')

        print('\n')
        plt.plot(train_data.groupby(col).count()[[index]], 'o-', label='train')
        plt.plot(test_data.groupby(col).count()[[index]], 'o-', label='test')
        plt.title('Train and test instances overlap.')
        plt.legend(loc=0)
        plt.ylabel('Number of records')
        plt.show()

def is_df(obj):
    r"""Returns True if `obj` is a pandas dataFrame
    Note that this function is simply doing ```isinstance(obj, df)```
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_df``.

     Args:
        obj (Object): Object to test
    Example::

        >>> x = pd.df()
        >>> torch.is_df(x)
        True
    """
    return isinstance(obj, df)
