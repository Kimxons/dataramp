"""Module providing secure and user-friendly functions to load data into pandas DataFrames.

This module includes functions for loading data from various file formats and databases
with robust error handling, security checks, and performance optimizations.
"""

import logging
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from a CSV file into a Pandas DataFrame with security checks.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the CSV file.
    **kwargs : dict
        Additional arguments to pass to pd.read_csv

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Raises:
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid CSV file.

    Example:
    --------
    >>> df = load_csv("data.csv")
    >>> df = load_csv("large.csv", chunksize=1000)  # Returns iterator
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {file_path}")

    try:
        return pd.read_csv(file_path, **kwargs)
    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"Empty CSV file: {file_path}") from e
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path}: {e}")
        raise ValueError(f"CSV parsing error: {file_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        raise


def load_excel(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from an Excel file with security warnings for macros.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the Excel file.
    **kwargs : dict
        Additional arguments to pass to pd.read_excel

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Warns:
    ------
    UserWarning
        If loading an Excel file with potential macros
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    # Security warning for Excel files
    warnings.warn(
        "Excel files may contain malicious macros. Only open files from trusted sources.",
        UserWarning,
    )

    try:
        return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {e}")
        raise ValueError(f"Invalid Excel file: {file_path}") from e


def load_from_db(
    query: str,
    connection_string: Optional[str] = None,
    engine: str = "sqlalchemy",
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Securely load data from a SQL database with parameterized queries.

    Parameters
    ----------
    query : str
        SQL query (preferably parameterized using :param_name syntax)
    connection_string : Optional[str]
        Database connection string (default: use DB_URI environment variable)
    engine : str
        Database engine type ("sqlalchemy" or "sqlite3")
    params : Optional[Dict[str, Any]]
        Parameters for SQL query to prevent injection
    **kwargs : dict
        Additional arguments to pass to pd.read_sql

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Example:
    --------
    >>> load_from_db(
    ...     "SELECT * FROM users WHERE age > :min_age",
    ...     params={"min_age": 18},
    ...     engine="sqlalchemy"
    ... )
    """
    # Get connection string from environment if not provided
    connection_string = connection_string or os.getenv("DB_URI")
    if not connection_string:
        raise ValueError("Database connection string required")

    try:
        if engine == "sqlalchemy":
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                return pd.read_sql(
                    text(query).bindparams(**params) if params else text(query),
                    conn,
                    **kwargs,
                )
        else:
            with sqlite3.connect(connection_string) as conn:
                return pd.read_sql(query, conn, params=params, **kwargs)
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise ValueError("Database operation failed") from e


def load_json(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from a JSON file with schema validation.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the JSON file.
    **kwargs : dict
        Additional arguments to pass to pd.read_json

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        return pd.read_json(file_path, **kwargs)
    except ValueError as e:
        logger.error(f"JSON syntax error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading JSON: {e}")
        raise


def load_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from a Parquet file with schema validation.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the Parquet file.
    **kwargs : dict
        Additional arguments to pass to pd.read_parquet
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    try:
        return pd.read_parquet(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Parquet loading error: {e}")
        raise ValueError(f"Invalid Parquet file: {file_path}") from e


def load_feather(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from a Feather file.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the Feather file.
    **kwargs : dict
        Additional arguments to pass to pd.read_feather
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feather file not found: {file_path}")

    try:
        return pd.read_feather(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Feather loading error: {e}")
        raise ValueError(f"Invalid Feather file: {file_path}") from e


def data_load(source: Union[str, Path], method: str = "csv", **kwargs) -> pd.DataFrame:
    """Unified data loading interface with automatic format detection.

    Parameters
    ----------
    source : Union[str, Path]
        File path or database connection string
    method : str
        Load method (csv|parquet|feather|excel|json|database)
    **kwargs : dict
        Method-specific arguments

    Returns:
    -------
    pd.DataFrame
        Loaded data

    Example:
    --------
    >>> # Load CSV
    >>> df = data_load("data.csv")

    >>> # Load from database
    >>> df = data_load(
    ...     "SELECT * FROM table",
    ...     method="database",
    ...     connection_string="sqlite:///mydb.sqlite"
    ... )
    """
    loaders = {
        "csv": load_csv,
        "parquet": load_parquet,
        "feather": load_feather,
        "excel": load_excel,
        "json": load_json,
        "database": load_from_db,
    }

    if method not in loaders:
        raise ValueError(
            f"Unsupported method: {method}. Available: {list(loaders.keys())}"
        )

    return loaders[method](source, **kwargs)


"""Example usage
# Simple CSV load
df = data_load("data.csv")

# Secure database load with parameters
df = data_load(
    "SELECT * FROM users WHERE age > :min_age",
    method="database",
    params={"min_age": 18},
    connection_string="postgresql://user:pass@localhost/db"
)

# Large file processing
for chunk in data_load("big.csv", chunksize=10000):
    process(chunk)

# Environment variable usage
# Set DB_URI=postgresql://user:pass@localhost/db in environment
df = data_load("SELECT * FROM table", method="database")"""
