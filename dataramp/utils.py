from __future__ import annotations

import platform
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

"""
Feature Engineering: Transform, Binning Temporal, Image Feature selection. Also feature extraction comes into play here.
"""

if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")


def get_num_vars(df: Union[pd.DataFrame, pd.Series]) -> list:
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    return df.select_dtypes(include=np.number).columns.tolist()


def describe_df(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame")

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


def get_cat_vars(df: Union[pd.DataFrame, pd.Series]) -> list:
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    return df.select_dtypes(include="object").columns.tolist()


def get_cat_counts(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame")

    cat_vars = get_cat_vars(df)
    counts = {var: df[var].value_counts().shape[0] for var in cat_vars}
    return pd.DataFrame(
        {"Feature": list(counts.keys()), "Unique Count": list(counts.values())}
    )


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(cols, list):
        raise TypeError("cols must be a list of column names")

    df[cols] = df[cols].astype("category")

    encoder = OneHotEncoder()
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[cols]))
    encoded_cols.columns = encoder.get_feature_names_out(cols)

    df = df.drop(cols, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)

    return df


def target_encode(df: pd.DataFrame, col: str, target_column: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not isinstance(col, str) or not isinstance(target_column, str):
        raise TypeError("col and target_column must be strings")

    target_mean = df.groupby(col)[target_column].mean()
    df[f"{col}_encoded"] = df[col].map(target_mean)

    return df


def plot_feature_importance(
    estimator: object, feature_names: List[str], show_plot: bool = True
) -> Optional[plt.Figure]:
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

    return None  # When show_plot is True


def feature_summary(
    df: Union[pd.DataFrame, pd.Series], visualize: bool = False
) -> pd.DataFrame:
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


def display_missing(
    df: pd.DataFrame,
    plot: bool = False,
    exclude_zero: bool = False,
    sort_by: str = "missing_count",
    ascending: bool = False,
) -> Optional[pd.DataFrame]:
    if df is None:
        raise ValueError("Expected a pandas DataFrame, but got None")

    if not isinstance(df, (pd.DataFrame, pd.Series)):
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
    return None


def get_unique_counts(data: pd.DataFrame) -> pd.DataFrame:
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
