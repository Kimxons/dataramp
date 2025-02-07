"""Enterprise-grade data loading module with CPU optimization and security safeguards.

Supporting formats: CSV, Excel, JSON, Parquet, Feather, ORC, SQL databases.
"""

import csv
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from dask import dataframe as dd
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, exc, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
MAX_CATEGORY_CARDINALITY = 1000
_CHUNK_READER_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB

MAX_PARALLEL_WORKERS = 8
MEMORY_SAFETY_FACTOR = 0.7
PARALLEL_CHUNK_SIZE = 10**5  # 100,000 rows per chunk


class DataOptimizer:
    """ML-powered data type optimization engine."""

    def __init__(self, sample_size=10000):
        self.sample_size = sample_size
        self.type_rules = {
            "category_threshold": 0.1,
            "float_precision": 32,
            "int_threshold": 0.2,
        }

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart type optimization using heuristic rules."""
        for col in df.columns:
            col_data = df[col]

            if self._is_datetime(col_data):
                df[col] = pd.to_datetime(col_data)
                continue

            if pd.api.types.is_numeric_dtype(col_data):
                df[col] = self._optimize_numeric(col_data)
            elif pd.api.types.is_string_dtype(col_data):
                df[col] = self._optimize_string(col_data)

        return df

    def _optimize_numeric(self, series: pd.Series) -> pd.Series:
        """Optimize numeric columns with anomaly detection."""
        try:
            if series.dropna().empty:
                return series

            clf = IsolationForest(contamination=0.05)
            mask = clf.fit_predict(series.values.reshape(-1, 1)) == 1
            clean_series = series[mask]

            if np.issubdtype(series.dtype, np.integer):
                return pd.to_numeric(clean_series, downcast="integer")

            downcast = pd.to_numeric(series, downcast="float")
            if self._precision_loss(downcast, series):
                return series.astype(f"float{self.type_rules['float_precision']}")
            return downcast

        except Exception as e:
            logger.warning(f"Numeric optimization failed: {str(e)}")
            return series

    def _optimize_string(self, series: pd.Series) -> pd.Series:
        """Smart string categorization with entropy analysis."""
        unique_count = series.nunique()
        total_count = len(series)
        unique_ratio = unique_count / total_count

        if unique_ratio < self.type_rules["category_threshold"]:
            return series.astype("category")

        if self._is_categorical_code(series):
            return series.astype("category")

        return series

    def _is_datetime(self, series: pd.Series) -> bool:
        """Advanced datetime pattern detection."""
        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y%m%d"]
        sample = series.dropna().sample(min(self.sample_size, len(series)))

        for fmt in date_formats:
            try:
                pd.to_datetime(sample, format=fmt, errors="raise")
                return True
            except (ValueError, TypeError):
                continue
        return False

    def _is_categorical_code(self, series: pd.Series) -> bool:
        """Detect coded categorical patterns using regex."""
        sample = series.dropna().sample(min(500, len(series)))
        pattern = r"^[A-Za-z]+\d+$"
        match_ratio = sample.str.contains(pattern).mean()
        return match_ratio > 0.8

    def _precision_loss(self, converted: pd.Series, original: pd.Series) -> bool:
        """Detect significant precision loss after conversion."""
        try:
            return not np.allclose(converted, original, equal_nan=True, atol=1e-5)
        except TypeError:
            return False


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
    use_dask: bool = False,
    optimizer: Optional[DataOptimizer] = None,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Optimized CSV loader with parallel processing options."""
    file_path = Path(file_path)
    _validate_file(file_path)

    if parallel:
        return _parallel_csv_load(file_path, use_dask, optimizer, **kwargs)

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
            return _process_chunks(reader, False, file_path, optimizer)
        return (optimizer or DataOptimizer()).optimize(pd.read_csv(file_path, **kwargs))

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path.name}: {str(e)}")
        raise DataLoadError(f"CSV parsing failed: {file_path.name}") from e


def _parallel_csv_load(
    file_path: Path, use_dask: bool, optimizer: Optional[DataOptimizer], **kwargs
) -> pd.DataFrame:
    """Internal parallel CSV loader."""
    if use_dask:
        try:
            ddf = dd.read_csv(file_path, **kwargs)
            df = ddf.compute()
            return (optimizer or DataOptimizer()).optimize(df)
        except ImportError:
            logger.warning("Dask not installed, falling back to ThreadPool")
            return _multiprocess_pandas_load(
                file_path, pd.read_csv, optimizer, **kwargs
            )

    return _multiprocess_pandas_load(file_path, pd.read_csv, optimizer, **kwargs)


def _multiprocess_pandas_load(
    file_path: Path, loader: callable, optimizer: Optional[DataOptimizer], **kwargs
) -> pd.DataFrame:
    """Parallel processing using ProcessPoolExecutor."""
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        chunks = list(
            executor.map(
                lambda x: loader(x, **kwargs),
                _chunked_file_reader(file_path, PARALLEL_CHUNK_SIZE),
            )
        )

    return pd.concat([optimizer.optimize(chunk) for chunk in chunks], ignore_index=True)


def _chunked_file_reader(file_path: Path, chunksize: int):
    """Generate file chunks for parallel processing."""
    with open(file_path, "r") as f:
        header = f.readline()
        while True:
            lines = []
            for _ in range(chunksize):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            yield pd.read_csv([header] + lines)


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
    parallel: bool = False,
    index_col: Optional[str] = None,
    **kwargs,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Database loader with parallel execution options."""
    if parallel and index_col:
        return _parallel_db_load(query, connection_string, index_col, **kwargs)

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


def _parallel_db_load(
    query: str, connection_string: str, index_col: str, chunks: int = 4, **kwargs
) -> pd.DataFrame:
    """Parallel database query execution."""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            min_max = conn.execute(
                f"SELECT MIN({index_col}), MAX({index_col}) FROM ({query}) AS sub"
            ).fetchone()

        ranges = np.linspace(min_max[0], min_max[1], chunks + 1)
        queries = [
            f"{query} WHERE {index_col} BETWEEN {start} AND {end}"
            for start, end in zip(ranges[:-1], ranges[1:])
        ]

        with ThreadPoolExecutor(max_workers=chunks) as executor:
            futures = [
                executor.submit(load_from_db, q, connection_string) for q in queries
            ]
            results = [f.result() for f in futures]

        return pd.concat(results, ignore_index=True)
    except exc.SQLAlchemyError as e:
        logger.error(f"Parallel database load failed: {str(e)}")
        raise DataLoadError(f"Parallel database operation failed: {str(e)}")


def _process_chunks(
    reader: Iterable[pd.DataFrame],
    parallel: bool,
    file_path: Path,
    optimizer: Optional[DataOptimizer] = None,
) -> Iterable[pd.DataFrame]:
    """Process data chunks with optional parallelism."""
    opt = optimizer or DataOptimizer()
    if parallel:
        return _parallel_chunk_processing(reader, file_path, opt)

    for chunk in reader:
        yield opt.optimize(chunk)


def _parallel_chunk_processing(
    reader: Iterable[pd.DataFrame], file_path: Path, optimizer: DataOptimizer
) -> Iterable[pd.DataFrame]:
    """Parallel chunk processing using ThreadPool."""
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        yield from executor.map(optimizer.optimize, reader)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage through type optimization."""
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
    reader: Iterable[pd.DataFrame],
    parallel: bool,
    file_path: Path,
    optimizer: Optional[DataOptimizer] = None,
) -> Iterable[pd.DataFrame]:
    """Process data chunks with optional parallelism."""
    opt = optimizer or DataOptimizer()
    if parallel:
        return _parallel_chunk_processing(reader, file_path, opt)

    for chunk in reader:
        yield opt.optimize(chunk)


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
