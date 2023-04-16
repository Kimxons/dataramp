import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import platform

if platform.system() == "Darwin":
    plt.switch_backend("TkAgg")
else:
    plt.switch_backend("Agg")

from dateutil.parser import parse

def drop_missing(data=None, threshold=95):
    '''
    Drops missing columns with threshold of missing data.
    
    Parameters:
        data: Pandas DataFrame or Series, default None
            The input DataFrame or Series.
        threshold: float, default 95
            The percentage of missing values to be in a column before it is eligible for removal.
    Returns:
        Pandas DataFrame or Series
            The modified DataFrame or Series after dropping the missing columns.
    '''
    if data is None:
        data = pd.DataFrame()

    if isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"data must be a pandas DataFrame or Series, but got {type(data)}")

    missing_data = data.isna().mean() * 100
    cols_to_drop = missing_data[missing_data >= threshold].index

    if not cols_to_drop.empty:
        n_cols_dropped = len(cols_to_drop)
        n_cols_orig = data.shape[1]
        print(f"Dropped {n_cols_dropped}/{n_cols_orig} ({n_cols_dropped/n_cols_orig:.1%}) columns.")
        data = data.drop(columns=cols_to_drop, axis=1)

     return data
