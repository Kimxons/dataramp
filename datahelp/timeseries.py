import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    if data is None or date_cols is None:
        raise ValueError("Both 'data' and 'date_cols' must be provided.")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series.")

    if not isinstance(date_cols, list):
        raise TypeError("date_cols must be a list.")

    if not date_cols:
        raise ValueError("date_cols must not be an empty list.")

    if subset is None:
        subset = ["dow", "doy", "dom", "hr", "min", "is_wkd", "yr", "qtr", "mth"]

    if not set(subset).issubset(
        ["dow", "doy", "dom", "hr", "min", "is_wkd", "yr", "qtr", "mth"]
    ):
        raise ValueError("Invalid values in the 'subset' parameter.")

    for col in date_cols:
        if col not in data.columns:
            raise ValueError(f"{col} not found in the DataFrame.")

    df = data.copy()

    if "dow" in subset:
        df["dow"] = df[date_cols].dt.dayofweek

    if "doy" in subset:
        df["doy"] = df[date_cols].dt.dayofyear

    if "dom" in subset:
        df["dom"] = df[date_cols].dt.day

    if "hr" in subset:
        df["hr"] = df[date_cols].dt.hour

    if "min" in subset:
        df["min"] = df[date_cols].dt.minute

    if "is_wkd" in subset:
        df["is_wkd"] = df[date_cols].dt.dayofweek < 5

    if "yr" in subset:
        df["yr"] = df[date_cols].dt.year

    if "qtr" in subset:
        df["qtr"] = df[date_cols].dt.quarter

    if "mth" in subset:
        df["mth"] = df[date_cols].dt.month

    if drop:
        df.drop(date_cols, axis=1, inplace=True)

    return df


def extract_time_info(data, time_cols, subset=None, drop=True):
    """
    Returns time information as new columns in a pandas DataFrame.

    Parameters:
    -----------
    data: DataFrame or named Series
        The data set to extract time information from.

    time_cols: List, Array
        Name of time columns/features in the data set.

    subset: List, Array, Default None
        Time features to return, defaults to [hours, minutes, and seconds].

    drop: bool, Default True
        Drops the original time features from the data set.

    Returns:
    --------
    DataFrame or Series.
    """
    # Input validation
    if data is None or time_cols is None:
        raise ValueError("Both 'data' and 'time_cols' must be provided.")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series.")

    if not isinstance(time_cols, list):
        raise TypeError("time_cols must be a list.")

    if not time_cols:
        raise ValueError("time_cols must not be an empty list.")

    subset = subset or ["hours", "minutes", "seconds"]

    if not set(subset).issubset(["hours", "minutes", "seconds"]):
        raise ValueError("Invalid values in the 'subset' parameter.")

    for col in time_cols:
        if col not in data.columns:
            raise ValueError(f"{col} not found in the DataFrame.")

    # Copy the input data
    df = data.copy()

    # Extract time information
    for time_unit in subset:
        df[time_unit] = getattr(df[time_cols].dt, time_unit)

    # Drop original time columns if specified
    if drop:
        df.drop(time_cols, axis=1, inplace=True)


def get_time_elapsed(data=None, date_cols=None, by="s", col_name=None):
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
    if data is None or date_cols is None:
        raise ValueError("Both 'data' and 'date_cols' must be provided.")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series.")

    if not isinstance(date_cols, list):
        raise TypeError("date_cols must be a list.")

    if len(date_cols) != 2:
        raise ValueError("Exactly two date columns are required.")

    if col_name is None:
        raise ValueError("col_name must be provided.")

    for col in date_cols:
        if col not in data.columns:
            raise ValueError(f"{col} not found in the DataFrame.")

    if by not in ["h", "m", "s"]:
        raise ValueError("Invalid value for 'by'. It should be one of ['h', 'm', 's'].")

    df = data.copy()
    df[col_name] = (df[date_cols[1]] - df[date_cols[0]]).astype(
        "timedelta64[{}]".format(by)
    )

    return df


def get_period_of_day(date_col=None):
    if date_col is None:
        raise ValueError("date_col: Expecting a date column, got 'None'")

    if date_col.dtype != np.int:
        date_col_hr = pd.to_datetime(date_col).dt.hour
        return date_col_hr.map(
            lambda x: "morning"
            if x in range(13)
            else ("afternoon" if x in range(13, 17) else "evening")
        )
    else:
        return date_col.map(
            lambda x: "morning"
            if x in range(13)
            else ("afternoon" if x in range(13, 17) else "evening")
        )


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
    if data is None or date_col is None:
        raise ValueError("Both 'data' and 'date_col' must be provided.")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series.")

    if date_col not in data.columns:
        raise ValueError(f"{date_col} not found in the DataFrame.")

    df = extract_date_info(data, [date_col])
    print(df.describe())


def timeplot(
    data=None,
    num_cols=None,
    time_col=None,
    subplots=True,
    marker=".",
    figsize=(15, 10),
    y_label="Daily Totals",
    save_fig=False,
    alpha=0.5,
    linestyle="None",
):
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
    if data is None or time_col is None:
        raise ValueError("Both 'data' and 'time_col' must be provided.")

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("data must be a pandas DataFrame or Series.")

    if time_col not in data.columns:
        raise ValueError(f"{time_col} not found in the DataFrame.")

    if num_cols is None:
        num_cols = get_num_vars(data)
        # Remove the time_col from num_cols
        num_cols.remove(time_col)

    # Make time_col the index
    data[time_col] = pd.to_datetime(data[time_col])
    # Set time_col as DataFrame index
    data = data.set_index(time_col)

    if subplots:
        axes = data[num_cols].plot(
            marker=marker,
            subplots=True,
            figsize=figsize,
            alpha=alpha,
            linestyle=linestyle,
        )
        for feature, ax in zip(num_cols, axes):
            ax.set_ylabel(y_label)
            ax.set_title("Timeseries Plot of '{}'".format(time_col))
            if save_fig:
                plt.savefig("fig_timeseries_plot_against_{}".format(feature))
        plt.show()
    else:
        for feature in num_cols:
            fig = plt.figure()
            ax = fig.gca()
            axes = data[feature].plot(
                marker=marker,
                subplots=False,
                figsize=figsize,
                alpha=alpha,
                linestyle=linestyle,
                ax=ax,
            )
            plt.ylabel(feature)
            ax.set_title("Timeseries Plot of '{}' vs. '{}' ".format(time_col, feature))
            if save_fig:
                plt.savefig("fig_timeseries_plot_against_{}".format(feature))
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
    data[date_col] = pd.to_datetime(data[date_col])
    return data.set_index(date_col)


def plot_time_series(
    data, time_col, value_cols, figsize=(15, 8), title="Time Series Plot"
):
    """
    Plot time series data.

    Parameters:
    -----------
    data: DataFrame.
        The time series data.

    time_col: str.
        The name of the time column.

    value_cols: list.
        List of column names containing values to be plotted.

    figsize: tuple, Default (15, 8).
        The figure size of the plot.

    title: str, Default "Time Series Plot".
        The title of the plot.

    Returns:
    --------
    Matplotlib figure.
    """
    data.set_index(time_col)[value_cols].plot(figsize=figsize)
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel("Values")
    plt.show()


def plot_seasonal_decomposition(data, time_col, value_col, freq=None):
    """
    Plot seasonal decomposition of time series data.

    Parameters:
    -----------
    data: DataFrame.
        The time series data.

    time_col: str.
        The name of the time column.

    value_col: str.
        The name of the column containing values to be decomposed.

    freq: int, Default None.
        The frequency of the time series. If not provided, it will be inferred.

    Returns:
    --------
    Matplotlib figure.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(data[value_col], freq=freq, model="additive")
    result.plot()
    plt.suptitle("Seasonal Decomposition of {}".format(value_col))
    plt.show()


def autocorrelation_plot(data, value_col, title="Autocorrelation Plot"):
    """
    Plot autocorrelation of time series data.

    Parameters:
    -----------
    data: DataFrame.
        The time series data.

    value_col: str.
        The name of the column containing values.

    title: str, Default "Autocorrelation Plot".
        The title of the plot.

    Returns:
    --------
    Matplotlib figure.
    """
    from pandas.plotting import autocorrelation_plot

    autocorrelation_plot(data[value_col])
    plt.title(title)
    plt.show()


def stationarity_check(data, value_col, window=12):
    """
    Check stationarity of time series data using rolling statistics.

    Parameters:
    -----------
    data: DataFrame.
        The time series data.

    value_col: str.
        The name of the column containing values.

    window: int, Default 12.
        The size of the rolling window.

    Returns:
    --------
    Matplotlib figure.
    """
    rolmean = data[value_col].rolling(window=window).mean()
    rolstd = data[value_col].rolling(window=window).std()

    plt.figure(figsize=(15, 6))
    plt.plot(data[value_col], label="Original")
    plt.plot(rolmean, label="Rolling Mean")
    plt.plot(rolstd, label="Rolling Std")
    plt.legend()
    plt.title("Rolling Mean & Standard Deviation")
    plt.show()
