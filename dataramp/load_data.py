import logging
import sqlite3

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loading CSV file from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file from {file_path}: {e}")
        raise

def load_excel(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Excel file from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file from {file_path}: {e}")
        raise

def load_from_db(connection_string: str, query: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(connection_string)
        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Loaded data from database with query: {query}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise
