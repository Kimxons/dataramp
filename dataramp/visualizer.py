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
from scipy.stats import pearsonr

from .exceptions import VisualizationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (10, 6)
LARGE_DATASET_THRESHOLD = 10_000


def validate_df(df: pd.DataFrame) -> None:
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
    validate_df(df)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Plotting histogram for non-numeric column: {column}")

    figsize = kwargs.pop("figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)

    data = df[column]
    if len(df) > LARGE_DATASET_THRESHOLD:
        data = data.sample(LARGE_DATASET_THRESHOLD)
        logger.info(f"Sampled {LARGE_DATASET_THRESHOLD} points for large dataset")

    sns.histplot(data, ax=ax, **{"kde": True, "bins": "auto", **kwargs})

    ax.set_title(f"Histogram of {column}", pad=20)
    ax.set_xlabel(column, labelpad=15)
    ax.set_ylabel("Frequency", labelpad=15)
    plt.tight_layout()

    logger.info(f"Histogram created for column: {column}")
    return (fig, ax) if return_figure else None


@handle_plot_errors
def plot_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    ax: Optional[Axes] = None,
    return_figure: bool = False,
    **kwargs,
) -> Optional[Tuple[Figure, Axes]]:
    """Create a scatter plot with density estimation for large datasets.

    Args:
        df: Input DataFrame
        x_column: X-axis column name
        y_column: Y-axis column name
        ax: Existing matplotlib axes (optional)
        return_figure: Return figure/axes objects
        **kwargs: Additional seaborn scatterplot arguments

    Returns:
        Tuple[Figure, Axes] if return_figure=True, else None
    """
    validate_df(df)

    for col in [x_column, y_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    figsize = kwargs.pop("figsize", DEFAULT_FIGSIZE)
    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)

    if len(df) > LARGE_DATASET_THRESHOLD:
        logger.info("Using hexbin plot for large dataset")
        ax.hexbin(df[x_column], df[y_column], gridsize=50, cmap="Blues", mincnt=1)
        plt.colorbar(ax.collections[0], ax=ax, label="Count")
    else:
        sns.scatterplot(
            x=x_column,
            y=y_column,
            data=df,
            ax=ax,
            **{"alpha": 0.6, "edgecolor": "none", **kwargs},
        )

    ax.set_title(f"{y_column} vs {x_column}", pad=20)
    ax.set_xlabel(x_column, labelpad=15)
    ax.set_ylabel(y_column, labelpad=15)
    plt.tight_layout()

    logger.info(f"Scatter plot created for {x_column} vs {y_column}")
    return (fig, ax) if return_figure else None


@handle_plot_errors
def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    ax: Optional[Axes] = None,
    return_figure: bool = False,
    **kwargs,
) -> Optional[Tuple[Figure, Axes]]:
    """Create a correlation matrix with statistical significance indicators.

    Args:
        df: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        ax: Existing matplotlib axes (optional)
        return_figure: Return figure/axes objects
        **kwargs: Additional seaborn heatmap arguments

    Returns:
        Tuple[Figure, Axes] if return_figure=True, else None
    """
    validate_df(df)

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        raise ValueError("No numeric columns for correlation matrix")

    corr_matrix = numeric_df.corr(method=method)
    p_values = numeric_df.corr(method=lambda x, y: pearsonr(x, y)[1])

    figsize = kwargs.pop("figsize", (12, 8))
    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) | (p_values > 0.05)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        **{
            "annot_kws": {"size": 10},
            "cbar_kws": {"label": f"{method.title()} Correlation"},
            **kwargs,
        },
    )

    ax.set_title(
        f"Correlation Matrix ({method.title()} Method)\n"
        "White = Non-significant (p > 0.05) or Duplicate",
        pad=25,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    logger.info("Correlation matrix plotted with significance filtering")
    return (fig, ax) if return_figure else None
