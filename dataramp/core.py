"""Core functionality for managing data science project structures and model persistence.

This module provides utilities for creating standardized data science project directories,
managing file paths, and saving machine learning models using different serialization methods.
"""

import hashlib
import json
import logging
import os
import pickle as pk
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path, PurePath
from typing import Dict, Optional, Tuple, Union

import fasteners
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Security configuration
DISABLE_PICKLE = os.getenv("DISABLE_PICKLE", "true").lower() == "true"

# Serialization methods
SUPPORTED_MODEL_METHODS = {
    "joblib": (jb.dump, "joblib"),
    "pickle": (pk.dump, "pkl") if not DISABLE_PICKLE else None,
}
SUPPORTED_DATA_METHODS = {
    "parquet": (pd.DataFrame.to_parquet, "parquet"),
    "feather": (pd.DataFrame.to_feather, "feather"),
    "csv": (pd.DataFrame.to_csv, "csv"),
}


@dataclass
class DataVersion:
    """Class representing a version of a dataset."""

    version_id: str
    timestamp: str
    description: str
    author: str
    data_hash: str
    file_path: Path
    metadata: dict
    dataset_name: str


class DataVersioner:
    """Manager for dataset versions with metadata tracking and integrity checks."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(get_path("processed_data_path")) / "versions"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / "version_history.json"
        self.lock_file = self.history_file.with_suffix(".lock")
        self.versions = self._load_history()

    def _load_history(self) -> Dict[str, DataVersion]:
        """Load version history with error handling."""
        with fasteners.InterProcessLock(self.lock_file):
            if not self.history_file.exists():
                return {}
            try:
                with open(self.history_file) as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading version history: {e}")
                return {}

            versions = {}
            for k, v in history.items():
                try:
                    versions[k] = DataVersion(
                        version_id=k,
                        timestamp=v["timestamp"],
                        description=v["description"],
                        author=v["author"],
                        data_hash=v["data_hash"],
                        file_path=Path(v["file_path"]),
                        metadata=v["metadata"],
                        dataset_name=v["metadata"]["dataset_name"],
                    )
                except KeyError as e:
                    logger.error(f"Invalid version entry {k}: {e}")
            return versions

    def _save_history(self):
        """Save version history with atomic write."""
        with atomic_write(self.history_file) as temp_path:
            with open(temp_path, "w") as f:
                json.dump(
                    {k: vars(v) for k, v in self.versions.items()},
                    f,
                    indent=2,
                    default=str,
                )

    def _calculate_hash(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """Calculate SHA-256 hash of the dataset."""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return hashlib.sha256(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()

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
    ) -> DataVersion:
        """Create a new dataset version with atomic writes and validation."""
        if data is None or data.empty:
            raise ValueError("Cannot version empty dataset")
        safe_name = PurePath(name).name
        data_hash = self._calculate_hash(data)
        version_id = self._generate_version_id(data_hash, version_format, safe_name)
        version_path = self.base_path / safe_name / version_id
        version_path.mkdir(parents=True, exist_ok=True)

        if method not in SUPPORTED_DATA_METHODS:
            raise ValueError(f"Unsupported data format: {method}")

        file_ext = SUPPORTED_DATA_METHODS[method][1]
        data_file = version_path / f"data.{file_ext}"
        with atomic_write(data_file) as temp_path:
            if method == "csv":
                data.to_csv(temp_path, index=False, compression=compression)
            elif method == "parquet":
                data.to_parquet(temp_path, compression=compression)
            elif method == "feather":
                data.to_feather(temp_path, compression=compression)
            else:
                getattr(data, f"to_{method}")(temp_path)

        # metadata
        version_metadata = {
            "dataset_name": safe_name,
            "author": author or os.getenv("USER", "unknown"),
            "created": datetime.now().isoformat(),
            "description": description,
            "columns": (
                list(data.columns) if isinstance(data, pd.DataFrame) else [data.name]
            ),
            "shape": data.shape,
            "data_hash": data_hash,
            "custom": metadata or {},
            "compression": compression,
        }

        metadata_file = version_path / "metadata.json"
        with atomic_write(metadata_file) as temp_path:
            with open(temp_path, "w") as f:
                json.dump(version_metadata, f, indent=2)

        version = DataVersion(
            version_id=version_id,
            timestamp=version_metadata["created"],
            description=description,
            author=version_metadata["author"],
            data_hash=data_hash,
            file_path=data_file,
            metadata=version_metadata,
            dataset_name=safe_name,
        )

        with fasteners.InterProcessLock(self.lock_file):
            self.versions[version_id] = version
            self._save_history()
        logger.info(f"Created version {version_id} of dataset {safe_name}")
        return version

    def _generate_version_id(
        self, data_hash: str, version_format: str, name: str
    ) -> str:
        """Generate version identifier with collision checks."""
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

    def get_version(self, version_id: str) -> DataVersion:
        """Retrieve a specific dataset version."""
        if version_id not in self.versions:
            raise KeyError(f"Version {version_id} not found")
        return self.versions[version_id]

    def list_versions(
        self, dataset_name: Optional[str] = None
    ) -> Dict[str, DataVersion]:
        """List versions with optional filtering by dataset name."""
        if dataset_name:
            return {
                k: v for k, v in self.versions.items() if v.dataset_name == dataset_name
            }
        return self.versions

    def validate_version(self, version_id: str) -> bool:
        """Validate data integrity with enhanced error handling."""
        version = self.get_version(version_id)
        try:
            if version.file_path.suffix == ".parquet":
                current_data = pd.read_parquet(version.file_path)
            elif version.file_path.suffix == ".feather":
                current_data = pd.read_feather(version.file_path)
            else:
                current_data = pd.read_csv(version.file_path)
            current_hash = self._calculate_hash(current_data)
            if current_hash != version.data_hash:
                logger.error(f"Data hash mismatch in version {version_id}")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation failed for {version_id}: {str(e)}")
            return False


@contextmanager
def atomic_write(file_path: Path):
    """Secure atomic file writes with permissions."""
    temp = file_path.with_suffix(".tmp")
    try:
        yield temp
        os.chmod(temp, 0o600)
        temp.replace(file_path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def get_project_root(filepath: Optional[str] = None) -> Path:
    """Reliable project root detection using marker files."""
    start_path = Path(filepath or os.getcwd()).resolve()
    for path in [start_path, *start_path.parents]:
        if (path / ".git").exists() or (path / ".dataramprc").exists():
            return path
    return start_path


def get_path(dir_key: str) -> Path:
    """Safe path resolution with auto-config creation."""
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
        with atomic_write(config_path) as temp_path:
            json.dump(default_config, temp_path.open("w"), indent=2)
    config = json.loads(config_path.read_text())
    if dir_key not in config:
        raise KeyError(f"Invalid config key: {dir_key}")
    return homedir / config[dir_key]


def create_directory(path: Path):
    """Safe directory creation with exist_ok."""
    path.mkdir(parents=True, exist_ok=True)


def create_project(project_name: str):
    """Project initialization with dependency files."""
    if not project_name or any(c in project_name for c in "/\\"):
        raise ValueError("Invalid project name")
    base_path = Path.cwd() / project_name
    dirs = [
        base_path / "datasets/raw",
        base_path / "datasets/processed/versions",
        base_path / "outputs/models",
        base_path / "src/scripts/ingest",
        base_path / "src/scripts/tests",
        base_path / "src/notebooks",
    ]
    for dir in dirs:
        create_directory(dir)

    # Create config file
    config_path = base_path / ".dataramprc"
    if not config_path.exists():
        get_path("data_path")  # Triggers auto-config

    # Create basic documentation
    readme = base_path / "README.md"
    with atomic_write(readme) as temp_path:
        temp_path.write_text(f"# {project_name}\n\nData science project structure.")

    # Create dependency files
    _generate_requirements_file(base_path)
    _generate_environment_file(base_path)
    logger.info(f"Created project structure at {base_path}")


def _generate_requirements_file(project_path: Path):
    """Generate requirements.txt with current versions."""
    core_packages = ["pandas", "numpy", "scikit-learn", "joblib", "pyarrow"]
    versions = {}
    for pkg in core_packages:
        try:
            versions[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None

    lines = [f"{pkg}>={ver}" for pkg, ver in versions.items() if ver is not None]

    req_file = project_path / "requirements.txt"
    with atomic_write(req_file) as temp_path:
        temp_path.write_text("\n".join(lines))


def _generate_environment_file(project_path: Path):
    """Generate environment.yml for Conda users."""
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
  - pyarrow>=6.0.0  # Parquet support
  - pip
  # Optional dependencies
  # - matplotlib>=3.5.0
  # - seaborn>=0.11.2
  - pip:
    # Add pip-only packages here
    # - package>=1.0.0
"""
    environment_path = project_path / "environment.yml"
    with atomic_write(environment_path) as temp_path:
        temp_path.write_text(environment_content)


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
    """Secure model saving with versioning and atomic writes."""
    method_info = SUPPORTED_MODEL_METHODS.get(method)
    if method_info is None:
        raise ValueError(f"Unsupported model serialization method: {method}")

    safe_name = PurePath(name).name
    models_dir = Path(models_dir or get_path("models_path"))
    models_dir.mkdir(parents=True, exist_ok=True)

    lock = models_dir / "versions.lock"
    with fasteners.InterProcessLock(lock):
        version_id = _resolve_version(
            safe_name, version, version_format, models_dir, method
        )

    file_ext = method_info[1]
    model_path = models_dir / f"{safe_name}_{version_id}.{file_ext}"

    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Model exists at {model_path}. Use overwrite=True.")

    try:
        with atomic_write(model_path) as temp_path:
            dump_method = method_info[0]
            if method == "joblib":
                dump_method(model, temp_path, compress=compression)
            elif method == "pickle":
                if compression:
                    raise ValueError("Pickle doesn't support compression")
                dump_method(model, temp_path)
            else:
                dump_method(model, temp_path)

        if metadata:
            meta_path = model_path.with_suffix(".json")
            with atomic_write(meta_path) as temp_path:
                json.dump(metadata, temp_path.open("w"), indent=2)

        return model_path
    except Exception as e:
        logger.error(f"Model save failed: {str(e)}")
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
    """Version resolution with conflict checking."""
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
                continue  # Skip invalid formats
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
    """Data saving with optional versioning."""
    if data is None or data.empty:
        raise ValueError("Cannot save empty dataset")
    method = method.lower()
    if method not in SUPPORTED_DATA_METHODS:
        raise ValueError(f"Unsupported format: {method}")

    data_path = (
        Path(get_path("processed_data_path"))
        / f"{name}.{SUPPORTED_DATA_METHODS[method][1]}"
    )
    with atomic_write(data_path) as temp_path:
        if method == "csv":
            data.to_csv(temp_path, index=False, compression=compression)
        elif method == "parquet":
            data.to_parquet(temp_path, compression=compression)
        else:
            getattr(data, f"to_{method}")(temp_path)

    if versioning:
        versioner = DataVersioner()
        return versioner.create_version(
            data, name, method=method, compression=compression, **version_kwargs
        )
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
