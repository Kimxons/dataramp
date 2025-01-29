"""Core functionality for managing data science project structures and model persistence.

This module provides utilities for creating standardized data science project directories,
managing file paths, and saving machine learning models using different serialization methods.
"""

import json
import logging
import os
import pickle as pk
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import joblib as jb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    data: Union[pd.DataFrame, pd.Series], name: str = "data", method: str = "parquet"
):
    """Save a DataFrame using the specified format.

    Args:
        data: The DataFrame or Series to save
        name: Base name for the output file (without extension)
        method: File format (parquet|feather|csv)
    """
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

        logging.info(f"Data saved to {data_path}")
    except Exception as e:
        logging.error(f"Data save failed: {str(e)}")
        raise
