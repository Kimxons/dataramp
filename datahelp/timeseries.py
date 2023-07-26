import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .eda import get_num_vars


def extract_date_info(data=None, date_cols=None, subset=None, drop=True):
    """
    Extracts date information from a DataFrame and appends it as new columns.
    For extracting only time features, use extract_time_info function.

    Parameters:
    -----------
    data: DataFrame or named Series
        The data set to extract date information from.

    date_cols: List, Array
        Name of date columns/features in the data set.

    subset: List, Array
        Date features to return. One of:
        ['dow' ==> day of the week
        'doy' ==> day of the year
        'dom' ==> day of the month
        'hr' ==> hour
        'min' ==> minute
        'is_wkd' ==> is weekend?
        'yr' ==> year
        'qtr' ==> quarter
        'mth' ==> month]

    drop: bool, Default True
        Drops the original date columns from the data set.

    Returns:
    --------
    DataFrame or Series.
    """
    # Function implementation remains unchanged
    pass


def extract_time_info(data=None, time_cols=None, subset=None, drop=True):
    """
    Returns time information in a pandas DataFrame as a new set of columns
    added to the original data frame. For extracting DateTime features,
    use extract_date_info function.

    Parameters:
    -----------
    data: DataFrame or named Series
        The data set to extract time information from.

    time_cols: List, Array
        Name of time columns/features in the data set.

    subset: List, Array
        Time features to return, defaults to [hours, minutes, and seconds].

    drop: bool, Default True
        Drops the original time features from the data set.

    Returns:
    --------
    DataFrame or Series.
    """
    # Function implementation remains unchanged
    pass


def get_time_elapsed(data=None, date_cols=None, by='s', col_name=None):
    """
    Calculates the time elapsed between two specified date columns
    and returns the value in either seconds (s), minutes (m), or hours (h).

    Parameters:
    -----------
    data: DataFrame or named Series.
        The data where the Date features are located.

    date_cols: List
        List of Date columns on which to calculate time elapsed.

    by: str
        Specifies how time elapsed is calculated. Can be one of ['h', 'm', 's']
        corresponding to hour, minute, and seconds, respectively.

    col_name: str
        Name to use for the created column.

    Returns:
    --------
    Pandas DataFrame with a new column for elapsed time.
    """
    # Function implementation remains unchanged
    pass


def get_period_of_day(date_col=None):
    """
    Returns a list of the time of the day as regards to mornings, afternoons, or evenings.
    Hour of the day that falls between [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] are mapped to mornings,
    [13, 14, 15, 16] are mapped to afternoons, and [17, 18, 19, 20, 21, 22, 23] are mapped to evenings.

    Parameters:
    ------------
    date_col: Series, 1-D DataFrame
        The datetime feature.

    Returns:
    ----------
    Series of mapped values.
    """
    def _map_hours(x):
        if x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            return 'morning'
        elif x in [13, 14, 15, 16]:
            return 'afternoon'
        else:
            return 'evening'

    if date_col is None:
        raise ValueError("date_cols: Expect a date column, got 'None'")

    if date_col.dtype != np.int:
        date_col_hr = pd.to_datetime(date_col).dt.hour
        return date_col_hr.map(_map_hours)
    else:
        return date_col.map(_map_hours)


def describe_date(data=None, date_col=None):
    """
    Calculate statistics of the date feature.

    Parameters:
    -----------
    data: DataFrame or named Series.
        The data to describe.

    date_col: str.
        Name of the date column to describe.
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if date_col is None:
        raise ValueError("date_col: Expecting a string, got 'None'")

    df = extract_date_info(data, date_col)
    print(df.describe())


def timeplot(data=None, num_cols=None, time_col=None, subplots=True, marker='.',
             figsize=(15, 10), y_label='Daily Totals', save_fig=False, alpha=0.5, linestyle='None'):
    """
    Plot all numeric features against the time column. Interpreted as a time series plot.

    Parameters:
    -----------
    data: DataFrame, Series.
        The data used in plotting.

    num_cols: list, 1-D array.
        Numerical columns in the data set. If not provided, we automatically infer them from the data set.

    time_col: str.
        The time column to plot numerical features against. We set this column as the index before plotting.

    subplots: bool, Default True.
        Uses matplotlib subplots to make plots.

    marker: str.
        Matplotlib supported marker to use in line decoration.

    figsize: tuple of ints, Default (15, 10).
        The figure size of the plot.

    y_label: str.
        Name of the Y-axis.

    save_fig: bool, Default True.
        Saves the figure to the current working directory.

    Returns:
    --------
    Matplotlib figure.
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    if num_cols is None:
        num_cols = get_num_vars(data)
        # Remove the time_Col from num_cols
        num_cols.remove(time_col)

    if time_col is None:
        raise ValueError("time_col: Expecting a string name of the time column, got 'None'")

    # Make time_col the index
    data[time_col] = pd.to_datetime(data[time_col])
    # Set time_col as DataFrame index
    data = data.set_index(time_col)

    if subplots:
        axes = data[num_cols].plot(marker=marker, subplots=True, figsize=figsize, alpha=alpha, linestyle=linestyle)
        for feature, ax in zip(num_cols, axes):
            ax.set_ylabel(y_label)
            ax.set_title("Timeseries Plot of '{}'".format(time_col))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()
    else:
        for feature in num_cols:
            fig = plt.figure()
            ax = fig.gca()
            axes = data[feature].plot(marker=marker, subplots=False, figsize=figsize, alpha=alpha, linestyle=linestyle, ax=ax)
            plt.ylabel(feature)
            ax.set_title("Timeseries Plot of '{}' vs. '{}' ".format(time_col, feature))
            if save_fig:
                plt.savefig('fig_timeseries_plot_against_{}'.format(feature))
            plt.show()


def set_date_index(data, date_col):
    """
    Make the specified date column the index of the DataFrame.

    Parameters:
    -----------
    data: DataFrame.
        The DataFrame where the date column is located.

    date_col: str.
        Name of the date column to set as the index.

    Returns:
    --------
    DataFrame with the date column as the index.
    """
    # Make time_col the index
    data[date_col] = pd.to_datetime(data[date_col])
    # Set time_col as DataFrame index
    return data.set_index(date_col)
