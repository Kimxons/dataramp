"""Core functionality for managing data science project structures and model persistence.

This module provides utilities for creating standardized data science project directories,
managing file paths, and saving machine learning models using different serialization methods.
"""

import hashlib
import json
import logging
import os
import pickle as pk
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import fasteners
import joblib as jb
import pandas as pd
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# security config
DISABLE_PICKLE = os.getenv("DISABLE_PICKLE", "true").lower() == "true"

# Model serialization methods
SUPPORTED_MODEL_METHODS = {
    "joblib": (jb.dump, "joblib"),
    "pickle": (pk.dump, "pkl"),
}

# Data serialization methods
SUPPORTED_DATA_METHODS = {
    "parquet": (pd.DataFrame.to_parquet, "parquet"),
    "feather": (pd.DataFrame.to_feather, "feather"),
    "csv": (pd.DataFrame.to_csv, "csv"),
}


def _calculate_hash(self, data: Union[pd.DataFrame, pd.Series]) -> str:
    """Calculate cryptographic hash of the dataset using SHA-256."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()


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


class DataVersioner:
    """Manager for dataset versions with metadata tracking and integrity checks."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(get_path("processed_data_path")) / "versions"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / "version_history.json"
        self.versions = self._load_history()

    def _load_history(self) -> Dict[str, DataVersion]:
        """Load version history from JSON file."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                history = json.load(f)
                return {
                    k: DataVersion(
                        version_id=k,
                        timestamp=v["timestamp"],
                        description=v["description"],
                        author=v["author"],
                        data_hash=v["data_hash"],
                        file_path=Path(v["file_path"]),
                        metadata=v["metadata"],
                    )
                    for k, v in history.items()
                }
        return {}

    def _save_history(self):
        """Save version history to JSON file."""
        with open(self.history_file, "w") as f:
            json.dump(
                {k: vars(v) for k, v in self.versions.items()}, f, indent=2, default=str
            )

    def create_version(
        self,
        data: Union[pd.DataFrame, pd.Series],
        name: str,
        description: str = "",
        author: Optional[str] = None,
        version_format: str = "timestamp",
        metadata: Optional[dict] = None,
        method: str = "parquet",
    ) -> DataVersion:
        """Create a new version of a dataset with full metadata tracking.

        Args:
            data: The dataset to version
            name: Base name for the dataset
            description: Human-readable description of changes
            author: Author of the version (default: current user)
            version_format: Version ID strategy (timestamp|hash|increment)
            metadata: Custom key-value metadata
            method: Storage format (parquet|feather|csv)

        Returns:
            DataVersion: Created version object
        """
        if data is None:
            raise ValueError("Cannot version None data")

        # Generate version ID
        data_hash = self._calculate_hash(data)
        version_id = self._generate_version_id(data_hash, version_format, name)

        # Create version directory
        version_path = self.base_path / name / version_id
        version_path.mkdir(parents=True, exist_ok=True)

        # Save data
        file_ext = SUPPORTED_DATA_METHODS[method][1]
        data_file = version_path / f"data.{file_ext}"
        data_save(data, data_file.stem, method=method)

        # Save metadata
        metadata_file = version_path / "metadata.json"
        version_metadata = {
            "author": author or os.getenv("USER", "unknown"),
            "created": datetime.now().isoformat(),
            "description": description,
            "columns": (
                list(data.columns) if isinstance(data, pd.DataFrame) else [data.name]
            ),
            "shape": data.shape,
            "data_hash": data_hash,
            "custom": metadata or {},
        }
        with open(metadata_file, "w") as f:
            json.dump(version_metadata, f, indent=2)

        # Create version record
        version = DataVersion(
            version_id=version_id,
            timestamp=version_metadata["created"],
            description=description,
            author=version_metadata["author"],
            data_hash=data_hash,
            file_path=data_file,
            metadata=version_metadata,
        )

        # Update history
        self.versions[version_id] = version
        self._save_history()

        logger.info(f"Created new version {version_id} of dataset {name}")
        return version

    def _calculate_hash(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """Calculate cryptographic hash of the dataset."""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

    def _generate_version_id(
        self, data_hash: str, version_format: str, name: str
    ) -> str:
        """Generate version identifier based on selected strategy."""
        if version_format == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        if version_format == "hash":
            return f"hash_{data_hash[:8]}"
        if version_format == "increment":
            existing = [
                v for v in self.versions.values() if name in v.file_path.parent.name
            ]
            return f"v{len(existing) + 1}"
        raise ValueError(f"Invalid version format: {version_format}")

    def get_version(self, version_id: str) -> DataVersion:
        """Retrieve a specific dataset version."""
        if version_id not in self.versions:
            raise KeyError(f"Version {version_id} not found")
        return self.versions[version_id]

    def list_versions(
        self, dataset_name: Optional[str] = None
    ) -> Dict[str, DataVersion]:
        """List all available versions, optionally filtered by dataset name."""
        if dataset_name:
            return {
                k: v
                for k, v in self.versions.items()
                if dataset_name in v.file_path.parent.name
            }
        return self.versions

        def validate_version(self, version_id: str) -> bool:
            """Validate data integrity for a specific version."""
            version = self.get_version(version_id)

            # Read data based on stored format
            if version.file_path.suffix == ".parquet":
                current_data = pd.read_parquet(version.file_path)
            elif version.file_path.suffix == ".feather":
                current_data = pd.read_feather(version.file_path)
            else:
                current_data = pd.read_csv(version.file_path)

            current_hash = self._calculate_hash(current_data)
            if current_hash != version.data_hash:
                logger.error(f"Data corruption detected in version {version_id}")
                return False
            return True


def get_project_root(filepath: str) -> str:
    """Determine the project root directory from a given file path.

    Args:
        filepath (str): The file path to evaluate.

    Returns:
        str: The project root directory path.

    Raises:
        ValueError: If the filepath is empty or None, or if an error occurs during processing.
    """
    if not filepath:
        raise ValueError("Empty or None filepath provided.")
    try:
        # Check if the filepath points to a file and get its directory
        if os.path.isfile(filepath):
            filepath = os.path.dirname(filepath)
        paths = [
            "src",
            "src/scripts/ingest",
            "src/scripts/tests",
            "src/notebooks",
            "src/outputs",
            "src/datasets",
        ]
        # Convert paths to OS-specific separators
        paths = [p.replace("/", os.path.sep) for p in paths]
        for path in paths:
            # Check if the directory ends with the path
            if filepath.endswith(path):
                parts = path.split(os.path.sep)
                depth = len(parts)
                project_root = Path(filepath).parents[depth - 1]
                return str(project_root)
        return filepath
    except Exception as e:
        raise ValueError(f"Error in get_project_root: {e}") from e


def get_path(dir: str) -> str:
    """Retrieve a specific path from a configuration file.

    Args:
        dir (str): The key of the path to retrieve.

    Returns:
        str: The path associated with the provided key.

    Raises:
        ValueError: If any errors occur during the path retrieval process.
    """
    try:
        homedir = get_project_root(os.getcwd())
        config_path = os.path.join(homedir, ".dataramprc")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No config file found at {config_path}")
        with open(config_path) as configfile:
            config = json.load(configfile)
        if dir not in config:
            raise KeyError(f"No key {dir} in config file {config_path}")
        # Join the project root with the relative path from config
        path = os.path.join(homedir, config[dir])
        return path
    except Exception as e:
        raise ValueError(f"Error in get_path: {e}") from e


def create_directory(path: Path):
    """Create a directory if it does not exist already.

    Args:
        path (Path): The path of the directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def create_project(project_name: str):
    """Create a standard data science project directory structure with dependency files.

    Args:
        project_name (str): The name of the project.

    Raises:
        ValueError: If the project name contains invalid characters.
    """
    if not project_name or any(c in project_name for c in "/\\"):
        raise ValueError(
            "Invalid project name. Avoid special characters like '/' or '\\'."
        )

    base_path = Path.cwd() / project_name
    data_path = base_path / "datasets"
    raw_data_path = data_path / "raw"
    processed_data_path = data_path / "processed"
    versions_path = processed_data_path / "versions"
    output_path = base_path / "outputs"
    models_path = output_path / "models"
    src_path = base_path / "src"
    scripts_path = src_path / "scripts"
    ingest_path = scripts_path / "ingest"
    test_path = scripts_path / "tests"
    notebooks_path = src_path / "notebooks"

    dirs = [
        base_path,
        data_path,
        raw_data_path,
        processed_data_path,
        output_path,
        models_path,
        src_path,
        scripts_path,
        ingest_path,
        test_path,
        notebooks_path,
        versions_path,
    ]

    for dir in dirs:
        create_directory(dir)

    # Store relative paths in the config
    config = {
        "description": "Configure the project settings",
        "data_path": "datasets",
        "raw_data_path": "datasets/raw",
        "processed_data_path": "datasets/processed",
        "output_path": "outputs",
        "models_path": "outputs/models",
    }

    config_path = base_path / ".dataramprc"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    readme_path = base_path / "README.md"
    with open(readme_path, "w") as readme:
        readme.write("Creates a standard data science project directory structure.")

    # Generate dependency management files
    _generate_requirements_file(base_path)
    _generate_environment_file(base_path)


def _generate_requirements_file(project_path: Path):
    """Generate a default requirements.txt file with core dependencies."""
    requirements = [
        "# Core dependencies",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "pyarrow>=6.0.0  # Required for Parquet support",
        "\n# Optional dependencies",
        "# matplotlib>=3.5.0  # Uncomment for visualization",
        "# seaborn>=0.11.2   # Uncomment for advanced plotting",
    ]

    requirements_path = project_path / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("\n".join(requirements))
    logger.info(f"Created requirements file at {requirements_path}")


def _generate_environment_file(project_path: Path):
    """Generate a default environment.yml file for Conda users."""
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
    with open(environment_path, "w") as f:
        f.write(environment_content)
    logger.info(f"Created environment file at {environment_path}")


def update_dependencies(requirements_file: str = "requirements.txt"):
    """Update core dependencies in requirements.txt while preserving structure."""
    try:
        # Get installed versions of core packages
        core_packages = {
            "pandas": None,
            "numpy": None,
            "scikit-learn": None,
            "joblib": None,
            "pyarrow": None,
        }

        installed = {pkg.key: pkg for pkg in pkg_resources.working_set}
        for pkg in core_packages:
            if pkg in installed:
                core_packages[pkg] = installed[pkg].version

        # Read existing requirements
        with open(requirements_file, "r") as f:
            lines = f.readlines()

        # Update core package versions
        new_lines = []
        for line in lines:
            # Split on any whitespace or comment
            parts = line.strip().split(maxsplit=1)
            if not parts:
                new_lines.append(line)
                continue

            pkg_name = parts[0].split("=")[0].split("<")[0].split(">")[0].lower()
            if pkg_name in core_packages and core_packages[pkg_name]:
                new_line = f"{pkg_name}>={core_packages[pkg_name]}"
                if len(parts) > 1 and parts[1].startswith("#"):
                    new_line += f"  # {parts[1][1:].strip()}"
                new_lines.append(new_line + "\n")
            else:
                new_lines.append(line)

        # Write updated requirements
        with open(requirements_file, "w") as f:
            f.writelines(new_lines)

        logger.info(f"Updated core dependencies in {requirements_file}")

    except Exception as e:
        logger.error(f"Failed to update dependencies: {str(e)}")
        raise


def model_save(
    model: object,
    name: str = "model",
    method: str = "joblib",
    version: Union[str, int, None] = None,
    version_format: str = "timestamp",  # or "increment"
    models_dir: Union[str, Path] = None,
    overwrite: bool = False,
    metadata: Optional[dict] = None,
) -> Path:
    """Save a model with versioning and security checks.

    Parameters:
    -----------
    model : object
        The model object to save
    name : str (default: "model")
        Base name for the output file
    method : str (default: "joblib")
        Serialization method (joblib|pickle)
    version : Union[str, int, None] (default: None)
        Custom version identifier or versioning strategy
    version_format : str (default: "timestamp")
        Auto-versioning format (timestamp|increment)
    models_dir : Union[str, Path] (default: get_path("models_path"))
        Directory to save models
    overwrite : bool (default: False)
        Allow overwriting existing models
    metadata : dict (optional)
        Additional metadata to store with the model

    Returns:
    --------
    Path: Path to the saved model file

    Raises:
    -------
    ValueError: For invalid inputs or version conflicts
    IOError: For file system errors

    Example:
    --------
    >>> model_save(model, "rf_classifier", version="prod_v1")
    Path('/models/rf_classifier_prod_v1.joblib')
    """
    # Security check for pickle
    if method == "pickle":
        warnings.warn(
            "Pickle serialization is not secure. Only load models from trusted sources.",
            UserWarning,
        )

    # Validate model object
    if not hasattr(model, "predict") and not hasattr(model, "transform"):
        logger.warning(
            "Model object doesn't appear to have standard scikit-learn methods"
        )

    # Set up paths
    models_dir = Path(models_dir or get_path("models_path"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine versioning
    version = _resolve_version(
        name=name,
        version=version,
        version_format=version_format,
        models_dir=models_dir,
        method=method,
    )

    # Construct filename
    file_ext = SUPPORTED_MODEL_METHODS[method][1]
    filename = f"{name}_{version}.{file_ext}"
    model_path = models_dir / filename

    # Check for existing files
    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Model already exists at {model_path}. Use overwrite=True to replace."
        )

    # Save model
    try:
        save_func = SUPPORTED_MODEL_METHODS[method][0]
        with open(model_path, "wb") as f:
            save_func(model, f)

        # Save metadata
        if metadata:
            metadata_path = model_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial saves
        raise


def _resolve_version(
    name: str,
    version: Union[str, int, None],
    version_format: str,
    models_dir: Path,
    method: str,
) -> str:
    """Determine the appropriate version string."""
    if version:
        return str(version)

    if version_format == "timestamp":
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    if version_format == "increment":
        pattern = f"{name}_v*.{SUPPORTED_MODEL_METHODS[method][1]}"
        existing = list(models_dir.glob(pattern))
        if not existing:
            return "v1"
        versions = [int(f.stem.split("_v")[-1]) for f in existing]
        return f"v{max(versions) + 1}"

    raise ValueError(f"Invalid version format: {version_format}")


def data_save(
    data: Union[pd.DataFrame, pd.Series],
    name: str = "data",
    method: str = "parquet",
    versioning: bool = False,
    **version_kwargs,
) -> Union[Path, DataVersion]:
    """Save data with optional versioning."""
    if data is None:
        raise ValueError("Cannot save None data")

    method = method.lower()
    if method not in SUPPORTED_DATA_METHODS:
        raise ValueError(
            f"Unsupported data format: {method}. Supported: {list(SUPPORTED_DATA_METHODS.keys())}"
        )

    try:
        save_func, ext = SUPPORTED_DATA_METHODS[method]
        data_path = Path(get_path("processed_data_path")) / f"{name}.{ext}"
        create_directory(data_path.parent)

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if method == "csv":
            save_func(data, data_path, index=False)
        else:
            save_func(data, data_path)

        logger.info(f"Data saved to {data_path}")

        # Add versioning after successful save
        if versioning:
            versioner = DataVersioner()
            return versioner.create_version(
                data=data, name=name, method=method, **version_kwargs
            )
        return data_path

    except Exception as e:
        logger.error(f"Data save failed: {str(e)}")
        raise


def register_model(
    model: object,
    name: str,
    version: str,
    metadata: dict,
    registry_file: str = "model_registry.json",
):
    """Register a model in the registry with validation and conflict checking."""
    try:
        registry_path = Path(get_path("models_path")) / registry_file
        registry = {}

        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)

        # Check for existing version
        if name in registry and version in registry[name]:
            raise ValueError(f"Model {name} version {version} already exists")

        # Get correct file extension
        method = metadata.get("serialization_method", "joblib")
        file_ext = SUPPORTED_MODEL_METHODS.get(method, ("", "joblib"))[1]

        # Store model metadata
        model_entry = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "model_path": str(
                Path(get_path("models_path")) / f"{name}_{version}.{file_ext}"
            ),
        }

        if name not in registry:
            registry[name] = {}
        registry[name][version] = model_entry

        # Save registry
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Registered model {name} version {version}")

    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise


def create_pipeline(steps: list) -> Pipeline:
    """Create a validated scikit-learn compatible pipeline."""
    try:
        # Validate pipeline steps
        for i, (name, estimator) in enumerate(steps):
            is_last = i == len(steps) - 1
            required_methods = ["fit"] + (["transform"] if not is_last else [])

            for method in required_methods:
                if not hasattr(estimator, method):
                    raise ValueError(
                        f"Step {i} ({name}) is missing required method: {method}"
                    )

        return Pipeline(steps)

    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}")
        raise


def cached_operation(func):
    """Safe caching decorator with size limits and invalidation."""
    return lru_cache(maxsize=128)(func)


@cached_operation
def expensive_operation(data: pd.DataFrame):
    """Memory-efficient cached operation for DataFrames."""
    try:
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        logger.info(f"Running expensive operation on dataset {data_hash}")

        result = data.mean().mean()
        return result

    except Exception as e:
        logger.error(f"Expensive operation failed: {str(e)}")
        raise
