import platform
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if platform.system() == "Darwin":
    plt.switch_platform("TkAgg")
else:
    plt.switch_platform("Agg")


def plot_missing(df: Union[pd.DataFrame, pd.Series]) -> None:
    """
    Plots the data as a heatmap to show missing values

    Parameters
    ----------
    df : pandas DataFrame or Series
        The data to plot.
    """
    if df is not None:
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError("df: Expecting a pandas DataFrame or Series, got {type(df)}")
        sns.heatmap(df.isnull(), cbar=True)
        plt.show()
    raise ValueError("EXpected a pandas dataframe or series")
