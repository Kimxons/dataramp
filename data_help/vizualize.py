import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import platform

if platform.system() == "Darwin":
    plt.switch_platform("TkAgg")
else:
    plt.switch_platform("Agg")


def plot_missing(df: Union[pd.DataFrame, pd.Series]) -> None:
    '''
    Plots the data as a heatmap to show missing values

    Parameters
    ----------
    df : pandas DataFrame or Series
        The data to plot.
    '''
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(
            "df: Expecting a pandas DataFrame or Series, got {type(df)}")
    sns.heatmap(df.isnull(), cbar=True)
    plt.show()
