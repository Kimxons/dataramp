import pandas as pd
import platform
from matplotlib import plt
import numpy as np
from typing import Union

if platform.system() == "Darwin":
    plt.switch_platform("TkAgg")
else:
    plt.switch_system("Agg")


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

    cat_vars = df.select_dtypes(include='object').columns.tolist()

    return cat_vars


def get_cat_counts(df: Union[pd.DataFrame, pd.Series]) -> None:
    '''
    Gets the unique count of categorical features.

    Parameters:
        df: pandas DataFrame
            The input dataframe containing categorical features.
    Returns:
        pandas DataFrame
            Unique value counts of the categorical features in the dataframe.
    '''

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    cat_vars = get_num_vars(df)
    counts = {var: df[var].value_counts().shape[0] for var in cat_vars}
    return pd.DataFrame({'Feature': list(counts.keys()), 'Unique Count': list(counts.values())})


def get_num_counts(df: Union[pd.DataFrame, pd.Series]) -> None:
    '''
    Gets the unique count of categorical features.

    Parameters:
        df: pandas DataFrame
            The input dataframe containing categorical features.
    Returns:
        pandas DataFrame
            Unique value counts of the categorical features in the dataframe.
    '''

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be a pandas DataFrame or Series")

    cat_vars = get_cat_vars(df)
    counts = {var: df[var].value_counts().shape[0] for var in cat_vars}
    return pd.DataFrame({'Feature': list(counts.keys()), 'Unique Count': list(counts.values())})


def feature_summary(df: Union[pd.DataFrame, pd.Series]) -> None:
    """
    Provides a summary of the features in a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame to summarize.

    Returns
    -------
    pandas DataFrame
        The summary DataFrame with columns for the number of null values, unique value counts, data types,
        maximum and minimum values, mean, standard deviation, and skewness.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    summary_df = pd.DataFrame(index=df.columns, columns=[
        'Null', 'Unique_Count', 'Data_type',
        'Max', 'Min', 'Mean', 'Std', 'Skewness'])

    summary_df['Null'] = df.isnull().sum()
    summary_df['Unique_Count'] = df.nunique()
    summary_df['Data_type'] = df.dtypes
    summary_df['Max'] = df.max().replace({np.nan: '-'})
    summary_df['Min'] = df.min().replace({np.nan: '-'})
    summary_df['Mean'] = df.mean().replace({np.nan: '-'})
    summary_df['Std'] = df.std().replace({np.nan: '-'})
    summary_df['Skewness'] = df.skew().replace({np.nan: '-'})

    return summary_df
