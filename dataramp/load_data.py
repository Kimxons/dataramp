import logging
import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd

# TODO: Add support for other db configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the CSV file.

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
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV file from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file from {file_path}: {e}")
        raise


def load_excel(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from an Excel file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the Excel file.

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Raises:
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid Excel file.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Excel file from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file from {file_path}: {e}")
        raise


def load_from_db(connection_string: str, query: str) -> pd.DataFrame:
    """Load data from a database into a Pandas DataFrame using a SQL query.

    Parameters
    ----------
    connection_string : str
        The connection string for the database.
    query : str
        The SQL query to execute.

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Raises:
    ------
    ValueError
        If the connection string or query is invalid.
    """
    try:
        if not connection_string or not query:
            raise ValueError("Connection string and query must be provided.")

        # Use a context manager to ensure the connection is properly closed
        with sqlite3.connect(connection_string) as conn:
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded data from database with query: {query}")
            return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from a JSON file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the JSON file.

    Returns:
    -------
    pd.DataFrame
        The loaded data as a Pandas DataFrame.

    Raises:
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid JSON file.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_json(file_path)
        logger.info(f"Loaded JSON file from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading JSON file from {file_path}: {e}")
        raise
