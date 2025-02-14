"""Visualization module providing functions for creating various plots and charts using matplotlib and seaborn.

Includes histogram, scatter plot, and correlation matrix visualization capabilities.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .exceptions import VisualizationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (10, 6)
LARGE_DATASET_THRESHOLD = 10_000


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame for visualization functions."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns")


def handle_plot_errors(func):
    """Decorator for centralized error handling and logging."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise VisualizationError from e

    return wrapper


@handle_plot_errors
def plot_histogram(
    df: pd.DataFrame,
    column: str,
    ax: Optional[Axes] = None,
    return_figure: bool = False,
    **kwargs,
) -> Optional[Tuple[Figure, Axes]]:
    """Create a histogram with automatic binning and outlier handling.

    Args:
        df: Input DataFrame
        column: Column name to plot
        ax: Existing matplotlib axes (optional)
        return_figure: Return figure/axes objects
        **kwargs: Additional seaborn histplot arguments

    Returns:
        Tuple[Figure, Axes] if return_figure=True, else None
    """
    try:
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        sns.histplot(df[column], **kwargs)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
        logger.info(f"Histogram plotted for column: {column}")
    except Exception as e:
        logger.error(f"Error plotting histogram for column {column}: {e}")
        raise


def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str, **kwargs):
    """Create a scatter plot comparing two columns in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame containing the data to plot
    x_column: str
        Name of the column to plot on x-axis
    y_column: str
        Name of the column to plot on y-axis
    **kwargs
        Additional keyword arguments to pass to seaborn's scatterplot

    Returns:
    -------
    None
        Displays the plot using matplotlib
    """
    try:
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        sns.scatterplot(x=x_column, y=y_column, data=df, **kwargs)
        plt.title(f"Scatter plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
        logger.info(f"Scatter plot created for {x_column} vs {y_column}")
    except Exception as e:
        logger.error(f"Error plotting scatter plot for {x_column} vs {y_column}: {e}")
        raise


def plot_correlation_matrix(df: pd.DataFrame, **kwargs):
    """Create a correlation matrix heatmap for numerical columns in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame containing the numerical data to analyze
    **kwargs
        Additional keyword arguments to pass to seaborn's heatmap

    Returns:
    -------
    None
        Displays the correlation matrix plot using matplotlib
    """
    try:
        plt.figure(figsize=kwargs.get("figsize", (12, 8)))
        correlation_matrix = df.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap=kwargs.get("cmap", "coolwarm"),
            **kwargs,
        )
        plt.title("Correlation Matrix")
        plt.show()
        logger.info("Correlation matrix plotted")
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        raise
