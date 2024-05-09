import json
import os
import pickle
from pathlib import Path

import joblib

SUPPORTED_METHODS = {
    "joblib": joblib.dump,
    "pickle": pickle.dump,
    # "keras": tf.keras.models.save_model,
}


def _get_home_path(filepath: str) -> str:
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
        raise ValueError(f"Error in _get_home_path: {e}") from e

    return filepath


def _get_path(dir=None):
    try:
        homedir = _get_home_path(os.getcwd())
        config_path = os.path.join(homedir, "config.txt")

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

        path = os.path.join(homedir, config[dir].replace("/", os.path.sep))
        return path
    except Exception as e:
        raise ValueError(f"Error in _get_path: {e}") from e


def create_directory(path: Path):
    """Create a directory if it does not exist already."""
    path.mkdir(parents=True, exist_ok=True)


def create_project(project_name: str):
    # Create project directories
    base_path = Path.cwd() / project_name
    data_path = base_path / "datasets"
    output_path = base_path / "outputs"
    models_path = output_path / "models"
    src_path = base_path / "src"
    scripts_path = src_path / "scripts"
    ingest_path = scripts_path / "ingest"
    test_path = scripts_path / "tests"
    notebooks_path = src_path / "notebooks"

    # The project directories
    dirs = [
        base_path,
        data_path,
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

    # Project config settings
    config = {
        "description": "Holds the project config settings",
        "base_path": str(base_path),
        "data_path": str(data_path),
        "output_path": str(output_path),
        "models_path": str(models_path),
    }

    config_path = base_path / ".datahelprc"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    readme_path = base_path / "README.rst"
    with open(readme_path, "w") as readme:
        readme.write("Creates a standard data science project directory structure.")


def model_save(model, name="model", method="joblib"):
    if model is None:
        raise ValueError("Expecting a binary model file, got 'None'")

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method {method} not supported. Supported methods are: {list(SUPPORTED_METHODS.keys())}"
        )

    try:
        model_path = _get_path("model_path")
        file_name = f"{model_path}/{name}.{method}"

        SUPPORTED_METHODS[method](model, file_name)

    except FileNotFoundError:
        print(
            f"Models folder does not exist. Saving model to the {name} folder. "
            f"It is recommended that you start your project using datahelp's start_project function"
        )
        file_name = f"{name}.{method}"

        SUPPORTED_METHODS[method](model, file_name)

    except PermissionError as e:
        print(f"Permission error while saving model. Check file permissions. {e}")
    except Exception as e:
        print(f"Failed to save model due to {e}")
