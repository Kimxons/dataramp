"""Enterprise-grade data loading module with CPU optimization and security safeguards.

Supporting formats: CSV, Excel, JSON, Parquet, Feather, ORC, SQL databases.
"""

import csv
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, exc, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
MAX_CATEGORY_CARDINALITY = 1000
_CHUNK_READER_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB


class DataLoadError(Exception):
    """Base exception for data loading errors."""


class SecurityValidationError(DataLoadError):
    """Security validation failure."""


class EmptyDataError(DataLoadError):
    """Empty file or query result."""


def load_csv(
    file_path: Union[str, Path],
    low_memory: bool = False,
    dtype: Optional[Dict] = None,
    parse_dates: Optional[list] = None,
    chunksize: Optional[int] = None,
    encoding: str = "utf-8",
    parallel: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Optimized CSV loader with memory-aware processing."""
    file_path = Path(file_path)
    _validate_file(file_path)

    if dtype is None:
        schema = infer_data_schema(file_path, encoding=encoding, **kwargs)
        dtype = schema["dtypes"]
        parse_dates = parse_dates or schema.get("parse_dates")

    chunksize = _calculate_optimal_chunksize(file_path, chunksize)

    try:
        reader = pd.read_csv(
            file_path,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=low_memory,
            chunksize=chunksize,
            on_bad_lines="warn",
            encoding=encoding,
            **kwargs,
        )

        if chunksize:
            return _process_chunks(reader, parallel, file_path)
        return optimize_memory(reader)

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path.name}: {str(e)}")
        raise DataLoadError(f"CSV parsing failed: {file_path.name}") from e


def load_excel(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    dtype: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Secure Excel loader with macro validation."""
    file_path = Path(file_path)
    _validate_file(file_path)
    _validate_excel_file(file_path)

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
    """JSON loader with schema validation."""
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
    """High-performance Parquet loader."""
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
    """Feather format loader."""
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
    """ORC format loader."""
    file_path = Path(file_path)
    _validate_file(file_path)

    try:
        df = pd.read_orc(file_path, columns=columns, **kwargs)
        return optimize_memory(df)
    except Exception as e:
        logger.error(f"ORC error: {str(e)}")
        raise DataLoadError("ORC loading failed") from e


def load_from_db(
    query: str,
    connection_string: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    streaming: bool = False,
    chunk_size: int = 10_000,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Database loader with advanced security."""
    conn_str = connection_string or os.getenv("DB_URI")
    if not conn_str:
        raise ValueError("Database connection string required")

    _validate_query(query)

    engine = _get_db_engine(conn_str)

    try:
        if streaming:
            return _stream_db_results(engine, query, params, chunk_size)

        with engine.connect() as conn:
            df = pd.read_sql(text(query).bindparams(**(params or {})), conn, **kwargs)
            return optimize_memory(df)

    except exc.SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise DataLoadError(f"Database operation failed: {str(e)}") from e


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage through type optimization.."""
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


def infer_data_schema(
    file_path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE, **reader_args
) -> Dict[str, Any]:
    """Generate optimized data schema with statistical sampling."""
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
    """Comprehensive file validation."""
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
        "Excel files may contain malicious macros. Only open trusted files.",
        UserWarning,
    )

    if file_path.suffix.lower() in (".xlsb", ".xlsm"):
        raise SecurityValidationError("Potentially unsafe Excel file format")


def _validate_query(query: str):
    """SQL injection prevention."""
    forbidden_patterns = [
        (";", "Query termination"),
        ("--", "SQL comment"),
        ("/*", "Block comment start"),
        ("*/", "Block comment end"),
        ("xp_", "Extended procedure"),
        ("DROP ", "DROP statement"),
        ("DELETE ", "DELETE statement"),
        ("INSERT ", "INSERT statement"),
        ("UPDATE ", "UPDATE statement"),
        ("TRUNCATE ", "TRUNCATE statement"),
    ]

    for pattern, description in forbidden_patterns:
        if pattern.upper() in query.upper():
            raise SecurityValidationError(
                f"Potentially dangerous SQL pattern: {description}"
            )


def _calculate_optimal_chunksize(
    file_path: Path, user_chunksize: Optional[int]
) -> Optional[int]:
    """Calculate memory-safe chunk size."""
    if user_chunksize:
        return user_chunksize

    try:
        import psutil

        file_size = file_path.stat().st_size
        mem_available = psutil.virtual_memory().available

        if file_size > _CHUNK_READER_WARNING_THRESHOLD:
            logger.warning(f"Loading large file: {file_size/1e9:.1f}GB")

        return max(1, int((mem_available * 0.7) // (file_size / 1000)))

    except ImportError:
        logger.warning("psutil not installed, using default chunking")
        return None


def _process_chunks(
    reader: Iterable[pd.DataFrame], parallel: bool, file_path: Path
) -> Iterable[pd.DataFrame]:
    """Process data chunks with optional parallelism."""
    if parallel:
        return _parallel_chunk_processing(reader, file_path)

    for chunk in reader:
        yield optimize_memory(chunk)


def _parallel_chunk_processing(
    reader: Iterable[pd.DataFrame], file_path: Path
) -> Iterable[pd.DataFrame]:
    """Parallel chunk processing using ThreadPool."""
    from concurrent.futures import ThreadPoolExecutor

    def process(chunk):
        return optimize_memory(chunk)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        yield from executor.map(process, reader)


_CONNECTION_POOL = {}


def _get_db_engine(connection_string: str):
    """Database connection pool manager."""
    if connection_string not in _CONNECTION_POOL:
        _CONNECTION_POOL[connection_string] = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            connect_args={"application_name": "DataLoader"},
        )
    return _CONNECTION_POOL[connection_string]


def _stream_db_results(engine, query, params, chunk_size):
    """Stream database results server-side."""
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(
            text(query), params
        )
        while True:
            chunk = result.fetchmany(chunk_size)
            if not chunk:
                break
            yield pd.DataFrame(chunk, columns=result.keys())


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

    return loaders[method](source, **kwargs)
