"""Core functionality for managing data science project structures and model persistence.

This module provides utilities for creating standardized data science project directories,
managing file paths, and saving machine learning models using different serialization methods.
"""

import hashlib
import json
import logging
import os
import pickle as pk
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path, PurePath
from typing import IO, Any, Dict, Generator, List, Optional, Tuple, Union

import fasteners
import google.protobuf.json_format as protobuf_json
import joblib as jb
import msgpack
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DISABLE_PICKLE = os.getenv("DISABLE_PICKLE", "false").lower() == "true"

SUPPORTED_MODEL_METHODS = {
    "joblib": (jb.dump, "joblib"),
    "msgpack": (msgpack.packb, "msgpack"),
}

if not DISABLE_PICKLE:
    SUPPORTED_MODEL_METHODS["pickle"] = (pk.dump, "pkl")

SUPPORTED_DATA_METHODS = {
    "parquet": (pd.DataFrame.to_parquet, "parquet"),
    "feather": (pd.DataFrame.to_feather, "feather"),
    "csv": (pd.DataFrame.to_csv, "csv"),
    "hdf5": (pd.DataFrame.to_hdf, "hdf5"),
    "orc": (pd.DataFrame.to_orc, "orc"),
    "sql": (pd.DataFrame.to_sql, "sql"),
    "protobuf": (protobuf_json.MessageToJson, "proto"),
    "msgpack": (msgpack.packb, "msgpack"),
}

SUPPORTED_COMPRESSION = {
    "gzip": "gzip",
    "snappy": "snappy",
    "zstd": "zstd",
    "lz4": "lz4",
    "brotli": "brotli",
}

@dataclass
class DataVersion:
    """Class representing a version of a dataset."""

    version_id: str
    author: str
    data_hash: str
    timestamp: str
    description: str
    file_path: Path
    metadata: dict
    dataset_name: str
    compression: Optional[str] = None


@contextmanager
def atomic_write(
    file_path: Path, mode: str = "w", encoding: str = "utf-8"
) -> Generator[IO, Any, None]:
    """Secure atomic file writes with permissions."""
    temp = file_path.with_suffix(".tmp")
    try:
        with open(temp, mode, encoding=encoding) as file:
            yield file
            os.chmod(temp, 0o600)
            os.replace(temp, file_path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


class DataVersioner:
    """Manager for dataset versions with metadata tracking."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the DataVersioner."""
        self.base_path = base_path or Path(get_path("processed_data_path")) / "versions"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / "version_history.json"
        self.lock_file = self.history_file.with_suffix(".lock")
        self.versions = {}
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Ensure version history file exists and is valid JSON."""
        if not self.history_file.exists():
            with atomic_write(self.history_file) as f:
                json.dump({}, f)

    def create_version(
        self,
        data: Union[pd.DataFrame, pd.Series],
        name: str,
        description: str = "",
        author: Optional[str] = None,
        version_format: str = "timestamp",
        metadata: Optional[dict] = None,
        method: str = "parquet",
        compression: Optional[str] = None,
        compression_level: Optional[int] = None,
    ) -> DataVersion:
        """Create a new dataset version."""
        if data is None or data.empty:
            raise ValueError("Cannot version empty dataset")

        if compression and compression not in SUPPORTED_COMPRESSION:
            raise ValueError(f"Unsupported compression codec: {compression}")

        safe_name = PurePath(name).name
        data_hash = self._calculate_hash(data)
        version_id = self._generate_version_id(data_hash, version_format, safe_name)
        version_path = self.base_path / safe_name / version_id
        version_path.mkdir(parents=True, exist_ok=True)

        if method not in SUPPORTED_DATA_METHODS:
            raise ValueError(f"Unsupported data format: {method}")

        file_ext = SUPPORTED_DATA_METHODS[method][1]
        data_file = version_path / f"data.{file_ext}"

        compression_opts = {}
        if compression:
            compression_opts["compression"] = compression
            if compression_level is not None:
                compression_opts["compression_level"] = compression_level

        # Determine write mode based on method requirements
        mode = "wb" if method in ["parquet", "feather", "hdf5", "orc", "msgpack"] else "w"
        with atomic_write(data_file, mode=mode) as file:
            if method == "protobuf":
                if not hasattr(data, "SerializeToString"):
                    raise ValueError("Data object must be a protobuf message")
                file.write(data.SerializeToString())
            elif method == "msgpack":
                packed = msgpack.packb(data.to_dict(orient="records"))
                file.write(packed)
            elif method == "csv":
                data.to_csv(file, index=False, **compression_opts)
            elif method == "parquet":
                data.to_parquet(file, **compression_opts)
            elif method == "feather":
                data.to_feather(file, compression=compression)
            elif method == "hdf5":
                data.to_hdf(file, key="data", mode="w", format="table", **compression_opts)
            elif method == "orc":
                data.to_orc(file, engine="pyarrow", compression=compression)
            elif method == "sql":
                raise NotImplementedError("SQL serialization requires a connection parameter")
            else:
                raise ValueError(f"Unsupported method: {method}")

        version_metadata = {
            "dataset_name": safe_name,
            "author": author or os.getenv("USER", "unknown"),
            "created": datetime.now().isoformat(),
            "description": description,
            "columns": list(data.columns) if isinstance(data, pd.DataFrame) else [data.name],
            "shape": data.shape,
            "data_hash": data_hash,
            "custom": metadata or {},
            "compression": compression,
            "compression_level": compression_level,
            "format": method,
        }

        metadata_file = version_path / "metadata.json"
        with atomic_write(metadata_file, mode="w") as file:
            json.dump(version_metadata, file, indent=2)

        version = DataVersion(
            version_id=version_id,
            author=version_metadata["author"],
            data_hash=data_hash,
            timestamp=version_metadata["created"],
            description=description,
            file_path=data_file,
            metadata=version_metadata,
            dataset_name=safe_name,
            compression=compression,
        )

        with fasteners.InterProcessLock(self.lock_file):
            self.versions = self._load_history()
            self.versions[version_id] = version
            self._save_history()

        logger.info(f"Created version {version_id} of dataset {safe_name}")
        return version

    def _calculate_hash(
        self,
        data: Union[pd.DataFrame, pd.Series],
        chunk_size: int = 10000,
        num_workers: int = 4,
    ) -> str:
        if isinstance(data, pd.Series):
            data = data.to_frame()

        data = data.sort_index(axis=1)
        data = data.fillna("NULL_PLACEHOLDER")

        def hash_chunk(chunk: pd.DataFrame) -> bytes:
            chunk_hash = (
                pd.util.hash_pandas_object(chunk, index=True)
                .values.astype(np.int64)
                .tobytes()
            )
            return hashlib.sha256(chunk_hash).digest()

        chunk_hashes = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(hash_chunk, data.iloc[i : i + chunk_size])
                for i in range(0, len(data), chunk_size)
            ]
            for future in futures:
                chunk_hashes.append(future.result())

        final_hash = hashlib.sha256(b"".join(chunk_hashes)).hexdigest()
        return final_hash

    def _generate_version_id(
        self, data_hash: str, version_format: str, name: str
    ) -> str:
        if version_format == "timestamp":
            version_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        elif version_format == "hash":
            version_id = f"hash_{data_hash[:8]}"
        elif version_format == "increment":
            existing = [v for v in self.versions.values() if v.dataset_name == name]
            version_id = f"v{len(existing) + 1}"
        else:
            raise ValueError(f"Invalid version format: {version_format}")
        if version_id in self.versions:
            raise ValueError(f"Version ID collision detected: {version_id}")
        return version_id

    def _load_history(self) -> Dict[str, DataVersion]:
        if not self.history_file.exists():
            return {}
        with open(self.history_file, "r") as f:
            history = json.load(f)
        return {k: DataVersion(**v) for k, v in history.items()}

    def _save_history(self):
        with atomic_write(self.history_file, mode="w") as f:
            history = {
                k: {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(v).items()
                }
                for k, v in self.versions.items()
            }
            json.dump(history, f, indent=2)

    def get_version(self, version_id: str) -> DataVersion:
        if version_id not in self.versions:
            raise KeyError(f"Version {version_id} not found")
        return self.versions[version_id]

    def list_versions(self, dataset_name: Optional[str] = None) -> Dict[str, DataVersion]:
        if dataset_name:
            return {k: v for k, v in self.versions.items() if v.dataset_name == dataset_name}
        return self.versions

    def validate_version(self, version_id: str) -> bool:
        version = self.get_version(version_id)
        try:
            ext = version.file_path.suffix[1:]
            read_method = getattr(pd, f"read_{ext}", None)
            if not read_method:
                logger.error(f"Unsupported validation format: {ext}")
                return False
            current_data = read_method(version.file_path)
            current_hash = self._calculate_hash(current_data)
            return current_hash == version.data_hash
        except Exception as e:
            logger.error(f"Error validating version {version_id}: {e}")
            return False

def get_project_root(filepath: Optional[str] = None) -> Path:
    start_path = Path(filepath or os.getcwd()).resolve()
    for path in [start_path, *start_path.parents]:
        if (path / ".git").exists() or (path / ".dataramprc").exists():
            return path
    return start_path

def get_path(dir_key: str) -> Path:
    homedir = get_project_root()
    config_path = homedir / ".dataramprc"
    if not config_path.exists():
        default_config = {
            "data_path": "datasets",
            "raw_data_path": "datasets/raw",
            "processed_data_path": "datasets/processed",
            "output_path": "outputs",
            "models_path": "outputs/models",
        }
        with atomic_write(config_path, mode="w") as f:
            json.dump(default_config, f, indent=2)
    config = json.loads(config_path.read_text())
    if dir_key not in config:
        raise KeyError(f"Invalid config key: {dir_key}")
    return homedir / config[dir_key]

def create_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_project(
    project_name: str,
    extra_dirs: Optional[List[str]] = None,
    init_git: bool = False,
    packages: Optional[List[str]] = None,
):
    if not project_name or any(c in project_name for c in "/\\"):
        raise ValueError("Invalid project name")
    if (Path.cwd() / project_name).exists():
        raise ValueError(f"Project '{project_name}' already exists.")

    base_path = Path.cwd() / project_name
    config_path = base_path / ".dataramprc"
    directories = [
        base_path / "datasets/raw",
        base_path / "datasets/processed/versions",
        base_path / "outputs/models",
        base_path / "src/scripts/ingest",
        base_path / "src/scripts/tests",
        base_path / "src/notebooks",
    ]
    if extra_dirs:
        directories.extend(base_path / d for d in extra_dirs)

    for directory in directories:
        create_directory(directory)

    if not config_path.exists():
        config_content = {
            "data_path": "datasets",
            "model_path": "outputs/models",
            "logging_level": "INFO",
        }
        with atomic_write(config_path, mode="w") as f:
            json.dump(config_content, f, indent=4)

    readme_template = f"""# {project_name}

    ## Description
    A data science project structure.

    ## Installation
    Install dependencies:
    ```
    pip install -r requirements.txt
    ```

    ## Usage
    Run scripts from `src/scripts`.
    """
    readme = base_path / "README.md"
    with atomic_write(readme) as temp_path:
        temp_path.write(readme_template)

    _generate_requirements_file(base_path)
    _generate_environment_file(base_path)

    if init_git:
        subprocess.run(["git", "init"], cwd=base_path, check=True)

    logger.info(f"Created project at {base_path}")


def _generate_requirements_file(project_path: Path):
    core_packages = ["pandas", "numpy", "scikit-learn", "joblib", "pyarrow"]
    versions = {}
    for pkg in core_packages:
        try:
            versions[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None

    lines = [f"{pkg}>={ver}" for pkg, ver in versions.items() if ver]
    req_file = project_path / "requirements.txt"
    with atomic_write(req_file, mode="w") as f:
        f.write("\n".join(lines))


def _generate_environment_file(project_path: Path):
    environment_content = """name: data_science_env
channels:
- conda-forge
- defaults
dependencies:
- python>=3.8
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- joblib>=1.1.0
- pyarrow>=6.0.0
- pip
"""
    environment_path = project_path / "environment.yml"
    with atomic_write(environment_path, mode="w") as f:
        f.write(environment_content)


def model_save(
    model: object,
    name: str = "model",
    method: str = "joblib",
    version: Optional[str] = None,
    version_format: str = "timestamp",
    models_dir: Optional[Path] = None,
    overwrite: bool = False,
    metadata: Optional[dict] = None,
    compression: Optional[Union[int, str]] = None,
) -> Path:
    method_info = SUPPORTED_MODEL_METHODS.get(method)
    if not method_info:
        raise ValueError(f"Unsupported method: {method}")

    safe_name = PurePath(name).name
    models_dir = models_dir or get_path("models_path")
    models_dir.mkdir(parents=True, exist_ok=True)

    lock = models_dir / "versions.lock"
    with fasteners.InterProcessLock(lock):
        version_id = _resolve_version(safe_name, version, version_format, models_dir, method)

    file_ext = method_info[1]
    model_path = models_dir / f"{safe_name}_{version_id}.{file_ext}"

    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Model exists at {model_path}. Use overwrite=True.")

    mode = "wb" if method in ["joblib", "pickle", "msgpack"] else "w"
    try:
        with atomic_write(model_path, mode=mode) as file:
            dump_method = method_info[0]
            if method == "joblib":
                jb.dump(model, file, compress=compression)
            elif method == "pickle":
                pk.dump(model, file)
            else:
                dump_method(model, file)

        if metadata:
            meta_path = model_path.with_suffix(".json")
            with atomic_write(meta_path, mode="w") as file:
                json.dump(metadata, file, indent=2)
        return model_path
    except Exception as e:
        logger.error(f"Model save failed: {e}")
        if model_path.exists():
            model_path.unlink()
        raise

def _resolve_version(
    name: str,
    version: Optional[str],
    version_format: str,
    models_dir: Path,
    method: str,
) -> str:
    if version:
        return version
    if version_format == "timestamp":
        return datetime.now().strftime("%Y%m%dT%H%M%S")
    if version_format == "increment":
        pattern = f"{name}_v*.{SUPPORTED_MODEL_METHODS[method][1]}"
        existing = []
        for path in models_dir.glob(pattern):
            try:
                version_num = int(path.stem.split("_v")[-1])
                existing.append(version_num)
            except ValueError:
                continue
        return f"v{max(existing) + 1 if existing else 1}"
    raise ValueError(f"Invalid version format: {version_format}")

def data_save(
    data: Union[pd.DataFrame, pd.Series],
    name: str = "data",
    method: str = "parquet",
    versioning: bool = False,
    compression: Optional[str] = None,
    **version_kwargs,
) -> Union[Path, DataVersion]:
    if data is None or data.empty:
        raise ValueError("Cannot save empty dataset")

    method = method.lower()
    if method not in SUPPORTED_DATA_METHODS:
        raise ValueError(f"Unsupported format: {method}")

    data_path = get_path("processed_data_path") / f"{name}.{SUPPORTED_DATA_METHODS[method][1]}"
    mode = "wb" if method in ["parquet", "feather", "hdf5", "orc", "msgpack"] else "w"
    with atomic_write(data_path, mode=mode) as file:
        if method == "csv":
            data.to_csv(file, index=False, compression=compression)
        elif method == "parquet":
            data.to_parquet(file, compression=compression)
        elif method == "feather":
            data.to_feather(file, compression=compression)
        else:
            raise ValueError(f"Unsupported method: {method}")

    if versioning:
        versioner = DataVersioner()
        return versioner.create_version(data, name, method=method, compression=compression, **version_kwargs)
    return data_path

def update_dependencies(requirements_file: str = "requirements.txt"):
    """Modern dependency version updating."""
    core_packages = ["pandas", "numpy", "scikit-learn", "joblib", "pyarrow"]
    current_versions = {pkg: pkg_version(pkg) for pkg in core_packages}
    try:
        with open(requirements_file, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                parts = line.strip().split("#")[0].split(">=")
                if len(parts) < 1:
                    f.write(line)
                    continue
                pkg = parts[0].strip().lower()
                if pkg in current_versions:
                    f.write(f"{pkg}>={current_versions[pkg]}\n")
                else:
                    f.write(line)
            f.truncate()
    except Exception as e:
        logger.error(f"Dependency update failed: {str(e)}")
        raise


def dataframe_hash(df: pd.DataFrame) -> str:
    """Stable hash for DataFrame content."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
        + str(df.shape).encode()
        + str(df.columns.tolist()).encode()
    ).hexdigest()


@lru_cache(maxsize=128)
def expensive_operation(data_hash: str, data_shape: Tuple[int, int]) -> float:
    """Cached operation with content validation."""
    logger.info(f"Processing dataset {data_hash[:8]}...")
    return pd.Series(np.random.randn(1000)).mean()


def create_pipeline(steps: list) -> Pipeline:
    """Validated scikit-learn pipeline creation."""
    for idx, (name, estimator) in enumerate(steps):
        required = ["fit"] + (["transform"] if idx != len(steps) - 1 else [])
        for method in required:
            if not hasattr(estimator, method):
                raise ValueError(f"Estimator {name} missing {method} method")
    return Pipeline(steps)


def register_model(
    model: object,
    name: str,
    version: str,
    metadata: dict,
    registry_file: str = "model_registry.json",
):
    """Model registration with conflict checking."""
    registry_path = Path(get_path("models_path")) / registry_file
    lock = registry_path.with_suffix(".lock")
    with fasteners.InterProcessLock(lock):
        registry = {}
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())

        if name in registry and version in registry[name]:
            raise ValueError(f"Model {name} version {version} exists")

        registry.setdefault(name, {})[version] = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "path": str(Path(get_path("models_path")) / f"{name}_{version}.joblib"),
        }

        with atomic_write(registry_path) as temp_path:
            with open(temp_path, "w") as f:
                json.dump(registry, f, indent=2)
