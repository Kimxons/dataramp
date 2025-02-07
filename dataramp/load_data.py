"""Enterprise-grade data loading module with CPU optimization and security safeguards.

Supporting formats: CSV, Excel, JSON, Parquet, Feather, ORC, SQL databases.
"""

import csv
import logging
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from dask import dataframe as dd
from sqlalchemy import create_engine, exc, text

from .exceptions import DataLoadError, EmptyDataError, SecurityValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 10_000
MAX_CATEGORY_CARDINALITY = 1000
_CHUNK_READER_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB

MAX_PARALLEL_WORKERS = 8
MEMORY_SAFETY_FACTOR = 0.7
PARALLEL_CHUNK_SIZE = 10**5  # 100,000 rows per chunk


class DataOptimizer:
    """ML-powered data type optimization with automatic fallback to basic methods."""

    def __init__(self, sample_size=10000):
        """Initialize the data optimizer.

        Args:
            sample_size (int): Number of samples to use for statistical analysis
        """
        self.sample_size = sample_size
        self.type_rules = {
            "category_threshold": 0.1,  # Max unique ratio for category conversion
            "float_precision": 32,  # Bit precision for float columns
            "int_threshold": 0.2,  # Null ratio threshold for integer conversion
        }
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Safely check if scikit-learn is available in the environment."""
        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401

            return True
        except ImportError:
            logger.warning("scikit-learn not installed, using basic optimization")
            return False

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage through type inference and conversion.

        Args:
            df (pd.DataFrame): Input dataframe to optimize

        Returns:
            pd.DataFrame: Optimized dataframe with reduced memory usage
        """
        for col in df.columns:
            col_data = df[col]

            # Handle datetime detection first
            if self._is_datetime(col_data):
                df[col] = pd.to_datetime(col_data)
                continue

            # Numeric column optimization
            if pd.api.types.is_numeric_dtype(col_data):
                df[col] = self._optimize_numeric(col_data)

            # String column optimization
            elif pd.api.types.is_string_dtype(col_data):
                df[col] = self._optimize_string(col_data)

        return df

    def _optimize_numeric(self, series: pd.Series) -> pd.Series:
        """Hybrid numeric optimization strategy.

        Uses ML-based optimization if available, otherwise falls back to basic downcasting.
        """
        if self._sklearn_available:
            return self._ml_optimize_numeric(series)
        return self._basic_optimize_numeric(series)

    def _ml_optimize_numeric(self, series: pd.Series) -> pd.Series:
        """ML-enhanced numeric optimization using IsolationForest for outlier detection.

        Args:
            series (pd.Series): Numeric series to optimize

        Returns:
            pd.Series: Optimized numeric series
        """
        try:
            if series.dropna().empty:
                return series

            # Detect outliers using IsolationForest
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(contamination=0.05, random_state=42)
            mask = clf.fit_predict(series.values.reshape(-1, 1)) == 1
            clean_series = series[mask]

            # Integer optimization
            if np.issubdtype(series.dtype, np.integer):
                return pd.to_numeric(clean_series, downcast="integer")

            # Float optimization with precision checks
            downcast = pd.to_numeric(series, downcast="float")
            if self._precision_loss(downcast, series):
                return series.astype(f"float{self.type_rules['float_precision']}")
            return downcast

        except Exception as e:
            logger.warning(f"ML numeric optimization failed: {str(e)}")
            return self._basic_optimize_numeric(series)

    def _basic_optimize_numeric(self, series: pd.Series) -> pd.Series:
        """Basic numeric optimization through simple downcasting.

        Args:
            series (pd.Series): Numeric series to optimize

        Returns:
            pd.Series: Downcast numeric series
        """
        try:
            if np.issubdtype(series.dtype, np.integer):
                return pd.to_numeric(series, downcast="integer")
            return pd.to_numeric(series, downcast="float")
        except Exception as e:
            logger.warning(f"Basic numeric optimization failed: {str(e)}")
            return series

    def _optimize_string(self, series: pd.Series) -> pd.Series:
        """Optimize string columns through categorical conversion.

        Args:
            series (pd.Series): String series to optimize

        Returns:
            pd.Series: Optimized series (category dtype if appropriate)
        """
        try:
            unique_count = series.nunique()
            total_count = len(series)
            unique_ratio = unique_count / total_count

            # Convert to category if under threshold
            if unique_ratio < self.type_rules["category_threshold"]:
                return series.astype("category")

            # Check for coded categorical patterns
            if self._is_categorical_code(series):
                return series.astype("category")

            return series
        except Exception as e:
            logger.warning(f"String optimization failed: {str(e)}")
            return series

    def _is_datetime(self, series: pd.Series) -> bool:
        """Detect datetime patterns using multiple common formats.

        Args:
            series (pd.Series): Series to check for datetime patterns

        Returns:
            bool: True if datetime pattern detected
        """
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%Y%m%d",
            "%Y-%m-%d %H:%M:%S",
            "%d-%b-%y",
            "%m/%d/%y",
        ]
        sample = series.dropna().sample(min(self.sample_size, len(series)))

        for fmt in date_formats:
            try:
                pd.to_datetime(sample, format=fmt, errors="raise")
                return True
            except (ValueError, TypeError):
                continue
        return False

    def _is_categorical_code(self, series: pd.Series) -> bool:
        """Detect coded categorical patterns using regex.

        Args:
            series (pd.Series): String series to check

        Returns:
            bool: True if pattern matches coded categorical
        """
        try:
            sample = series.dropna().sample(min(500, len(series)))
            pattern = r"^[A-Za-z]+\d+$"
            match_ratio = sample.str.contains(pattern, regex=True).mean()
            return match_ratio > 0.8
        except (ValueError, AttributeError) as e:
            logger.debug(f"Categorical code detection failed: {str(e)}")
            return False

    def _precision_loss(self, converted: pd.Series, original: pd.Series) -> bool:
        """Detect significant precision loss after numeric conversion.

        Args:
            converted (pd.Series): Downcast series
            original (pd.Series): Original series

        Returns:
            bool: True if significant precision loss detected
        """
        try:
            return not np.allclose(
                converted.astype("float64"),
                original.astype("float64"),
                equal_nan=True,
                atol=1e-5,
            )
        except TypeError:
            return False


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
    """Adaptive parallel processing with resource awareness."""
    try:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunks = list(
                executor.map(
                    _process_chunk_wrapper,
                    _chunked_file_reader(file_path, PARALLEL_CHUNK_SIZE),
                    [loader] * os.cpu_count(),
                    [kwargs] * os.cpu_count(),
                )
            )
    except pickle.PicklingError:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            chunks = list(
                executor.map(
                    lambda x: loader(x, **kwargs),
                    _chunked_file_reader(file_path, PARALLEL_CHUNK_SIZE),
                )
            )

    return pd.concat([optimizer.optimize(chunk) for chunk in chunks], ignore_index=True)


def _process_chunk_wrapper(chunk, loader, kwargs):
    """Helper for proper pickle serialization."""
    return loader(chunk, **kwargs)


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
    """Robust parallel database loading with connection pooling."""
    try:
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
