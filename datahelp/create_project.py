import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import joblib
import tensorflow as tf
from logger import create_rotating_log
from utils import _get_path

__author__ = "Meshack Kitonga"
__email__ = "dev.kitonga@gmail.com"


def create_directory(path: Optional[Path]):
    """Create a directory if it does not exist already"""
    path.mkdir(parents=True, exist_ok=True)


def create_project(project_name: Optional[str]):
    """
    Creates a standard data science project directory structure.

    Parameters:
        project_name (str): Name of directory to contain folders.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Create a new data science project.")
    parser.add_argument(
        "project_name",
        nargs="?",
        default="project_name",
        help="Name of the project directory",
    )
    args = parser.parse_args()
    create_project(args.project_name)

    if not project_name:
        raise ValueError("Project name cannot be empty or None.")

    log_filename = f"{project_name}.log"
    create_rotating_log(log_filename)

    logging.info("Creating project...")

    # project directory structure
    base_path = Path.cwd() / project_name
    data_path = base_path / "datasets"
    processed_path = data_path / "processed"
    raw_path = data_path / "raw"
    output_path = base_path / "outputs"
    models_path = output_path / "models"
    src_path = base_path / "src"
    scripts_path = src_path / "scripts"
    ingest_path = scripts_path / "ingest"
    preparation_path = scripts_path / "preparation"
    modeling_path = scripts_path / "modeling"
    test_path = scripts_path / "test"
    notebooks_path = src_path / "notebooks"

    # the project directories
    dirs = [
        base_path,
        data_path,
        processed_path,
        raw_path,
        output_path,
        models_path,
        src_path,
        scripts_path,
        ingest_path,
        preparation_path,
        modeling_path,
        test_path,
        notebooks_path,
    ]

    for dir in dirs:
        create_directory(dir)

    # project config settings
    config = {
        "description": "This file holds all configuration settings for the current project",
        "basepath": str(base_path),
        "datapath": str(data_path),
        "processedpath": str(processed_path),
        "rawpath": str(raw_path),
        "outputpath": str(output_path),
        "modelspath": str(models_path),
    }

    with open(base_path / "config.json", "w") as configfile:
        json.dump(config, configfile, indent=4)

    with open(base_path / "README.txt", "w") as readme:
        readme.write("Creates a standard data science project directory structure.")

    logging.info(f"Project created successuflly in {base_path}")


def model_save(model, name="model", method="joblib"):
    """
    Save a trained machine learning model in the models folder.
    Folders must be initialized using the datahelp start_project function.
    Creates a folder models if datahelp standard directory is not provided.
    Parameters:
        model: binary file, Python object
            Trained model file to save in the models folder.
        name: str, optional (default='model')
            Name of the model to save it with.
        method: str, optional (default='joblib')
            Format to use in saving the model. It can be one of ['joblib', 'pickle', 'keras'].
    Returns:
        None
    """
    if model is None:
        raise ValueError("Expecting a binary model file, got 'None'")
    SUPPORTED_METHODS = {
        "joblib": joblib.dump,
        "pickle": pickle.dump,
        "keras": tf.keras.models.save_model,
    }

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Method {method} not supported. Supported methods are: {list(SUPPORTED_METHODS.keys())}")
    try:
        model_path = os.path.join(_get_path("modelpath"), name)

        filename = f"{model_path}.{method}"

        SUPPORTED_METHODS[method](model, filename)

        logging.info(f"Model saved successfully to {filename}")
    except FileNotFoundError:
        msg = "models folder does not exist. Saving model to the {} folder. It is recommended that you start your project using datahelp's start_project function".format(
            name
        )
        logging.info(msg)

        filename = f"{name}.{method}"

        SUPPORTED_METHODS[method](model, filename)

        logging.info(f"Model saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Failed to save model due to {e}")
