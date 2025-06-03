import io
from functools import partials
import csv
import logging
import os
import pickle
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
from typing import cast
import filetype
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, exc, text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_exponential
# import polars as pl

# TODO: Add polars support

from .exceptions import DataLoadError, EmptyDataError, SecurityValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
MAX_CATEGORY_CARDINALITY = 1000
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", os.cpu_count() or 4))
MEMORY_SAFETY_FACTOR = float(os.getenv("MEMORY_SAFETY_FACTOR", 0.7))
PARALLEL_CHUNK_SIZE = int(os.getenv("PARALLEL_CHUNK_SIZE", 10**5))  # 100k rows
MAX_FILE_SIZE = 1024**3 * 10  # 10GB
MIN_PARALLEL_SIZE = 1024**2 * 100  # 100MB

class ConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.pools = {}
        return cls._instance

    def get_engine(self, conn_str: str) -> Engine:
        with self._lock:
            engine = self.pools.get(conn_str)
            if not engine or not self._is_healthy(engine):
                engine = self._create_engine(conn_str)
                self.pools[conn_str] = engine
            return engine

    def _is_healthy(self, engine: Engine) -> bool:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except exc.SQLAlchemyError:
            return False

    # To users; Add these env variables in your .env file
    def _create_engine(self, conn_str: str) -> Engine:
        return create_engine(
            conn_str,
            pool_size=int(os.getenv("DB_POOL_SIZE", 10)),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 20)),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", 3600)),
            pool_pre_ping=True,
            connect_args={"application_name": "DataLoader"},
        )

class DataOptimizer:
    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size
        self.type_rules = {
            "category_threshold": 0.1,
            "float_precision": 32,
            "int_threshold": 0.2,
        }

    def _is_datetime(self, col_data: pd.Series) -> bool:
        """Detect datetime columns with coercion validation."""
        try:
            pd.to_datetime(col_data, errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def _optimize_numeric(self, col_data: pd.Series) -> pd.Series:
        if pd.api.types.is_integer_dtype(col_data):
             return cast(pd.Series, pd.to_numeric(col_data, downcast="integer"))
        elif pd.api.types.is_float_dtype(col_data):
            return cast(pd.Series, pd.to_numeric(col_data, downcast="float"))
        return col_data

    def _optimize_string(self, col_data: pd.Series) -> pd.Series:
        """Convert strings to categories when cardinality is low."""
        if 1 < col_data.nunique() <= MAX_CATEGORY_CARDINALITY:
            return col_data.astype("category")
        return col_data

    def _validate_optimization_integrity(
        self, original: pd.DataFrame, optimized: pd.DataFrame
    ):
        """Ensure data integrity after transformations."""
        if original.shape != optimized.shape:
            raise RuntimeError(
                f"Shape mismatch: original {original.shape}, optimized {optimized.shape}"
            )

        sample = original.sample(min(100, len(original)))
        for col in original.columns:
            if not original[col].equals(optimized[col].loc[sample.index]):
                raise RuntimeError(f"Data corruption detected in column {col}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()

        try:
            logger.debug(f"Original memory usage: {original_memory / 1024**2:.2f} MB")
            for col in optimized_df.columns:
                col_data = optimized_df[col]

                if self._is_datetime(col_data):
                    optimized_df[col] = pd.to_datetime(col_data, errors="coerce")
                elif pd.api.types.is_numeric_dtype(col_data):
                    optimized_df[col] = self._optimize_numeric(col_data)
                elif pd.api.types.is_string_dtype(col_data):
                    optimized_df[col] = self._optimize_string(col_data)

            self._validate_optimization_integrity(df, optimized_df)
            return optimized_df

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return df

def _validate_file_type(file_path: Path, expected_type: str):
    if expected_type.lower() == "csv":
        if file_path.suffix.lower() != ".csv":
            raise SecurityValidationError("Invalid CSV file extension")
        _validate_csv_structure(file_path)
    else:
        guess = filetype.guess(str(file_path))
        if not guess or guess.extension != expected_type.lower():
            raise SecurityValidationError(
                f"Invalid {expected_type} file signature. Detected: {guess.extension if guess else 'unknown'}"
            )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def load_csv(
    file_path: Union[str, Path],
    low_memory: bool = False,
    dtype: Optional[Dict] = None,
    parse_dates: Optional[list] = None,
    chunksize: Optional[int] = None,
    encoding: str = "utf-8",
    parallel: bool = False,
    use_dask: bool = False,
    optimizer: Optional[DataOptimizer] = None,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    original_file_path_str = str(file_path)
    file_path = Path(file_path)
    _validate_file(file_path)
    _validate_file_signature(file_path, "csv")

    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise SecurityValidationError("File size exceeds maximum allowed limit")

        effective_parallel = parallel
        if file_size < MIN_PARALLEL_SIZE and parallel:
            effective_parallel = False  # Disable parallel for small files

        if effective_parallel:
            return _parallel_csv_load(file_path, use_dask, optimizer, **kwargs)

        if dtype is None:
            schema = infer_data_schema(file_path, encoding=encoding, **kwargs)
            dtype = schema.get("dtypes")
            if parse_dates is None and "parse_dates" in schema:
                parse_dates = schema.get("parse_dates")

        effective_chunksize = _calculate_optimal_chunksize(file_path, chunksize)

        reader = pd.read_csv(
            file_path,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=low_memory,
            chunksize=effective_chunksize,
            on_bad_lines="warn",
            encoding=encoding,
            **kwargs,
        )
        return _process_chunks(reader, effective_parallel, optimizer=optimizer)

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path.name}: {str(e)}")
        raise DataLoadError(f"CSV parsing failed: {file_path.name}") from e


def _validate_file_signature(file_path: Path, expected_type: str):
    """Validate file type using magic numbers."""
    guess = filetype.guess(str(file_path))
    if not guess or guess.extension != expected_type.lower():
        raise SecurityValidationError(
            f"Invalid file type. Expected {expected_type}, got {guess.extension if guess else 'unknown'}"
        )


def _parallel_csv_load(
    file_path: Path, use_dask: bool, optimizer: Optional[DataOptimizer], **kwargs
) -> pd.DataFrame:
    try:
        if use_dask:
            import dask.dataframe as dd

            return dd.read_csv(file_path, **kwargs).compute()
    except ImportError:
        logger.warning("Dask not installed, falling back to ThreadPool")

    return _multiprocess_load(file_path, pd.read_csv, optimizer, **kwargs)


def _multiprocess_load(
    file_path: Path, 
    loader: callable,  
    optimizer: Optional[DataOptimizer], 
    **kwargs  
) -> pd.DataFrame:
    opt = optimizer or DataOptimizer()
    load_with_kwargs = partial(loader, **kwargs)
    file_encoding = kwargs.get('encoding', 'utf-8')

    raw_df_chunks = []
    chunk_iterator = _chunked_file_reader(file_path, PARALLEL_CHUNK_SIZE, encoding=file_encoding)

    try:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            raw_df_chunks = list(executor.map(load_with_kwargs, chunk_iterator))
    except pickle.PicklingError:
        logger.warning("PicklingError with ProcessPoolExecutor, falling back to ThreadPoolExecutor for CSV loading.")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            raw_df_chunks = list(executor.map(load_with_kwargs, chunk_iterator))
    except Exception as e:
        logger.error(f"Error during parallel CSV loading for {file_path}: {e}")
        raise DataLoadError(f"Parallel CSV loading failed for {file_path}") from e

    if not raw_df_chunks:
        logger.warning(f"No data chunks were loaded from {file_path}. The file might be empty or contain only a header.")
        try:
            header_kwargs = {k: v for k, v in kwargs.items() if k != 'chunksize'}
            return pd.read_csv(file_path, nrows=0, **header_kwargs)
        except Exception as e_header:
            logger.warning(f"Could not read header for empty file {file_path}: {e_header}")
            return pd.DataFrame()

    optimized_chunks = [opt.optimize(chunk) for chunk in raw_df_chunks if not chunk.empty]

    if not optimized_chunks:
        logger.warning(f"All loaded chunks from {file_path} were empty after processing.")
        try:
            header_kwargs = {k: v for k, v in kwargs.items() if k != 'chunksize'}
            return pd.read_csv(file_path, nrows=0, **header_kwargs)
        except Exception as e_header:
            logger.warning(f"Could not read header for file with all empty chunks {file_path}: {e_header}")
            return pd.DataFrame()
            
    return pd.concat(optimized_chunks, ignore_index=True)

def _chunked_file_reader(file_path: Path, chunksize: int, encoding: str = "utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        header = f.readline()
        if not header: 
            logger.warning(f"CSV file {file_path} is empty or has no header.")
            return

        if not header.endswith('\n'):
            header += '\n'

        while True:
            current_chunk_lines = []
            for _ in range(chunksize):
                line = f.readline()
                if not line: 
                    break
                current_chunk_lines.append(line)

            if not current_chunk_lines: 
                break
            
            yield io.StringIO(header + "".join(current_chunk_lines))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def load_excel(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    dtype: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    file_path = Path(file_path)
    _validate_file(file_path)
    _validate_excel_file(file_path)
    _validate_file_type

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=dtype, **kwargs)
        return optimize_memory(df)
    except Exception as e:
        logger.error(f"Excel loading error: {str(e)}")
        raise DataLoadError(f"Failed to load Excel file: {file_path}") from e

def load_json(
    file_path: Union[str, Path],
    dtype: Optional[Dict] = None,
    precise_float: bool = False,
    **kwargs,
) -> pd.DataFrame:
    file_path = Path(file_path)
    _validate_file(file_path)

    try:
        df = pd.read_json(file_path, dtype=dtype, precise_float=precise_float, **kwargs)
        return optimize_memory(df)
    except ValueError as e:
        logger.error(f"JSON syntax error: {str(e)}")
        raise DataLoadError("Invalid JSON structure") from e

def load_parquet(
    file_path: Union[str, Path],
    columns: Optional[list] = None,
    memory_map: bool = True,
    **kwargs,
) -> pd.DataFrame:
    file_path = Path(file_path)
    _validate_file(file_path)

    try:
        df = pd.read_parquet(
            file_path, columns=columns, memory_map=memory_map, **kwargs
        )
        return optimize_memory(df)
    except Exception as e:
        logger.error(f"Parquet error: {str(e)}")
        raise DataLoadError("Parquet loading failed") from e


def load_feather(
    file_path: Union[str, Path], columns: Optional[list] = None, **kwargs
) -> pd.DataFrame:
    file_path = Path(file_path)
    _validate_file(file_path)

    try:
        df = pd.read_feather(file_path, columns=columns, **kwargs)
        return optimize_memory(df)
    except Exception as e:
        logger.error(f"Feather error: {str(e)}")
        raise DataLoadError("Feather loading failed") from e

def load_orc(
    file_path: Union[str, Path], columns: Optional[list] = None, **kwargs
) -> pd.DataFrame:
    file_path = Path(file_path)
    _validate_file(file_path)

    try:
        df = pd.read_orc(file_path, columns=columns, **kwargs)
        return optimize_memory(df)
    except Exception as e:
        logger.error(f"ORC error: {str(e)}")
        raise DataLoadError("ORC loading failed") from e


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def load_from_db(
    query: str,
    connection_string: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    streaming: bool = False,
    chunk_size: int = 10_000,
    parallel: bool = False,
    index_col: Optional[str] = None,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    if parallel and index_col:
        return _parallel_db_load(query, connection_string, index_col, **kwargs)

    conn_str = connection_string or os.getenv("DB_URI")

    pool = ConnectionPool()
    engine = pool.get_engine(conn_str)

    try:
        if streaming:
            return _stream_db_results(engine, query, params, chunk_size)

        with engine.connect() as conn:
            df = pd.read_sql(text(query).bindparams(**(params or {})), conn, **kwargs)
            if df.empty:
                logger.warning("Query returned empty result set")
            return optimize_memory(df)

    except exc.SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise DataLoadError(f"Database operation failed: {str(e)}") from e


def _calculate_optimal_chunksize(
    file_path: Path, user_chunksize: Optional[int]
) -> Optional[int]:
    """Memory-aware chunk size calculation with CPU core consideration."""
    if user_chunksize:
        return user_chunksize

    try:
        import psutil

        file_size = file_path.stat().st_size
        mem_available = psutil.virtual_memory().available
        cpu_cores = os.cpu_count() or 1

        chunk_size = max(
            1,
            int(
                (mem_available * MEMORY_SAFETY_FACTOR)
                / (cpu_cores * (file_size / 1000))
            ),
        )

        logger.info(f"Calculated chunk size: {chunk_size} rows")
        return chunk_size

    except ImportError:
        logger.warning("psutil not installed, using conservative defaults")
        return PARALLEL_CHUNK_SIZE

def _estimate_row_size(file_path: Path) -> float:
    import random

    sample_points = 5
    total_bytes = 0
    total_rows = 0

    with open(file_path, "rb") as f:
        file_size = f.seek(0, 2)
        for _ in range(sample_points):
            pos = random.randint(0, file_size - 1024)
            f.seek(pos)
            sample = f.read(1024)
            total_bytes += len(sample)
            total_rows += sample.count(b"\n")

    return total_bytes / (total_rows or 1)


def _parallel_db_load(
    query: str, connection_string: str, index_col: str, chunks: int = 4, **kwargs
) -> pd.DataFrame:
    try:
        pool = ConnectionPool()
        engine = _get_db_engine(connection_string)

        with engine.connect() as conn:
            # Verify index exists
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS tmp_idx ON ({query}) ({index_col})"
            )

            min_max = conn.execute(
                f"""
                SELECT MIN({index_col}), MAX({index_col})
                FROM ({query}) AS sub
            """
            ).fetchone()

        ranges = np.linspace(min_max[0], min_max[1], chunks + 1)
        queries = [
            f"{query} WHERE {index_col} BETWEEN {start} AND {end}"
            for start, end in zip(ranges[:-1], ranges[1:])
        ]

        with ThreadPoolExecutor(max_workers=chunks) as executor:
            futures = []
            for q in queries:
                futures.append(
                    executor.submit(load_from_db, q, connection_string, **kwargs)
                )

            results = [f.result() for f in futures]

        return pd.concat(results, ignore_index=True)

    except exc.SQLAlchemyError as e:
        logger.error(f"Parallel load failed: {str(e)}")
        raise DataLoadError(f"Parallel database operation failed: {str(e)}")
    finally:
        with engine.connect() as conn:
            conn.execute("DROP INDEX IF EXISTS tmp_idx")


def _process_chunks(
    reader: Iterable[pd.DataFrame],
    parallel: bool,
    optimizer: Optional[DataOptimizer] = None,
) -> Iterable[pd.DataFrame]:
    opt = optimizer or DataOptimizer()
    if parallel:
        return _parallel_chunk_processing(reader, opt)

    for chunk in reader:
        yield opt.optimize(chunk)


def _parallel_chunk_processing(
    reader: Iterable[pd.DataFrame], optimizer: DataOptimizer
) -> Iterable[pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        yield from executor.map(optimizer.optimize, reader)

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.columns:
            col_type = df[col].dtype
            if pd.api.types.is_object_dtype(col_type):
                unique_count = df[col].nunique()
                if 1 < unique_count <= MAX_CATEGORY_CARDINALITY:
                    df[col] = df[col].astype("category")
            elif pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="float")
        return df
    except Exception as e:
        logger.error(f"Memory optimization failed: {str(e)}")
        return df


def infer_data_schema(
    file_path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE, **reader_args
) -> Dict[str, Any]:
    sample = pd.read_csv(file_path, nrows=sample_size, **reader_args)
    optimized = optimize_memory(sample)
    return {
        "dtypes": optimized.dtypes.to_dict(),
        "parse_dates": [
            col
            for col, dtype in optimized.dtypes.items()
            if pd.api.types.is_datetime64_any_dtype(dtype)
        ],
    }


def _validate_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise EmptyDataError(f"Empty file: {file_path}")

    if file_path.suffix.lower() == ".csv":
        _validate_csv_structure(file_path)


def _validate_csv_structure(file_path: Path, sample_size: int = 1024):
    """Validate CSV structure and encoding."""
    try:
        with open(file_path, "rb") as f:
            sample = f.read(sample_size)
            if b"\0" in sample:
                raise SecurityValidationError(
                    "File contains null bytes (possible binary file)"
                )

            try:
                csv.Sniffer().sniff(sample.decode("utf-8", errors="replace"))
            except csv.Error as e:
                raise SecurityValidationError(f"CSV format error: {str(e)}")

    except UnicodeDecodeError as e:
        raise SecurityValidationError(f"Encoding error: {str(e)}") from e


def _validate_excel_file(file_path: Path):
    """Excel file security checks."""
    warnings.warn(
        "Excel files may contain malicious macros. Only open trusted files can be used.",
        UserWarning,
    )

    if file_path.suffix.lower() in (".xlsb", ".xlsm"):
        raise SecurityValidationError("Potentially unsafe Excel file format")

def _stream_db_results(
    engine: Engine, query: str, params: Optional[Dict], chunk_size: int
) -> Iterable[pd.DataFrame]:
    """Server-side result streaming."""
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(
            text(query), params or {}
        )
        while True:
            chunk = result.fetchmany(chunk_size)
            if not chunk:
                break
            df = pd.DataFrame(chunk, columns=result.keys())
            yield optimize_memory(df)

def data_load(
    source: Union[str, Path], method: str = "csv", **kwargs
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Unified data loading interface.

    Supported methods:
    - csv, excel, json, parquet, feather, orc, database
    """
    loaders = {
        "csv": load_csv,
        "excel": load_excel,
        "json": load_json,
        "parquet": load_parquet,
        "feather": load_feather,
        "orc": load_orc,
        "database": load_from_db,
    }

    if method not in loaders:
        raise ValueError(
            f"Unsupported method: {method}. Available: {list(loaders.keys())}"
        )

    try:
        return loaders[method](source, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load {method} data: {str(e)}")
        raise DataLoadError(f"{method} loading failed") from e
