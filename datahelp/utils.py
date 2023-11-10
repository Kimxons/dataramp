import json
import os
from pathlib import Path

def _get_home_path(filepath):
    """
    Helper function to get the project home path.
    """
    paths = [
        "src",
        "src/scripts/ingest",
        "src/scripts/preparation",
        "src/scripts/modeling",
        "src/notebooks",
    ]
    for path in paths:
        if filepath.endswith(path):
            return str(Path(filepath).parents[len(path.split('/')) - 1])
    return filepath

def _get_path(dir=None):
    """
    Helper function to get a path from the project configuration file.
    """
    homedir = _get_home_path(os.getcwd())
    config_path = os.path.join(homedir, "config.txt")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config file found at {config_path}")

    with open(config_path) as configfile:
        try:
            config = json.load(configfile)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in config file {config_path}: {e}")

    if dir not in config:
        raise KeyError(f"No key {dir} in config file {config_path}")

    path = os.path.join(homedir, config[dir])
    return path
