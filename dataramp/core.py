"""Core functionality for managing data science project structures and model persistence.

This module provides utilities for creating standardized data science project directories,
managing file paths, and saving machine learning models using different serialization methods.
"""

import json
import logging
import os
import pickle as pk
from pathlib import Path
from typing import Union

import joblib as jb
import pandas as pd

logging.basicConfig(level=logging.INFO)

# Model serialization methods
SUPPORTED_MODEL_METHODS = {
    "joblib": jb.dump,
    "pickle": pk.dump,
}

# Data serialization methods
SUPPORTED_DATA_METHODS = {
    "parquet": (pd.DataFrame.to_parquet, "data.parquet"),
    "feather": (pd.DataFrame.to_feather, "data.feather"),
    "csv": (pd.DataFrame.to_csv, "data.csv"),
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
        paths = [
            "src",
            "src/scripts/ingest",
            "src/scripts/tests",
            "src/notebooks",
            "src/outputs",
            "src/datasets",
        ]
        for path in paths:
            if filepath.endswith(path.replace("/", os.path.sep)):
                return str(Path(filepath).parents[len(path.split(os.path.sep)) - 1])
    except Exception as e:
        raise ValueError(f"Error in get_project_root: {e}") from e
    return filepath


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
        config_path = os.path.join(homedir, ".datahelprc")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No config file found at {config_path}")
        with open(config_path) as configfile:
            try:
                config = json.load(configfile)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Error decoding JSON in config file {config_path}: {e}"
                ) from e
        if dir not in config:
            raise KeyError(f"No key {dir} in config file {config_path}")
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
    """Create a standard data science project directory structure.

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

    config = {
        "description": "Configure the project settings",
        "base_path": str(base_path),
        "data_path": str(data_path),
        "raw_data_path": str(raw_data_path),
        "processed_data_path": str(processed_data_path),
        "output_path": str(output_path),
        "models_path": str(models_path),
    }

    config_path = base_path / ".datahelprc"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    readme_path = base_path / "README.md"
    with open(readme_path, "w") as readme:
        readme.write("Creates a standard data science project directory structure.")


def model_save(model: object, name: str = "model", method: str = "joblib"):
    """Save a model using the specified serialization method.

    Args:
        model: The model object to save
        name: Base name for the output file (without extension)
        method: Serialization method (joblib|pickle)
    """
    if model is None:
        raise ValueError("Cannot save None model")

    method = method.lower()
    if method not in SUPPORTED_MODEL_METHODS:
        raise ValueError(
            f"Unsupported model format: {method}. Supported: {list(SUPPORTED_MODEL_METHODS.keys())}"
        )

    try:
        save_func, ext = SUPPORTED_MODEL_METHODS[method]
        model_path = Path(get_path("models_path")) / f"{name}.{ext.split('.')[-1]}"
        create_directory(model_path.parent)

        with open(model_path, "wb") as f:
            save_func(model, f)

        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Model save failed: {str(e)}")
        raise


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
        data_path = (
            Path(get_path("processed_data_path")) / f"{name}.{ext.split('.')[-1]}"
        )
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


"""Example usage:
# Save model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model_save(model, "rf_model", method="joblib")

# Save data
import pandas as pd
df = pd.DataFrame({"col1": [1,2,3], "col2": ["a","b","c"]})
data_save(df, "processed_data", method="parquet")
"""
