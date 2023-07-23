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


def plot_feature_importance(importances: np.ndarray, feature_names: list) -> None:
    """
    Plots the feature importance as a bar chart.

    Parameters:
    -----------
    importances : numpy ndarray
        The feature importances from a trained model.
    feature_names : list of str
        The names of the features in the same order as the importances.

    Returns:
    --------
    None
    """
    if not isinstance(importances, np.ndarray) or not isinstance(feature_names, list):
        raise TypeError("importances should be a numpy ndarray and feature_names should be a list.")

    if len(importances) != len(feature_names):
        raise ValueError("importances and feature_names should have the same length.")

    sorted_indices = importances.argsort()[::-1]
    plt.bar(range(len(importances)), importances[sorted_indices])
    plt.xticks(range(len(importances)), np.array(feature_names)[sorted_indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
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
