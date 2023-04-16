import json
import os


def _get_home_path(filepath):
    """
    Helper function to get the project home path.
    """
    paths = ["src", "src/scripts/ingest", "src/scripts/preparation", "src/scripts/modeling", "src/notebooks"]
    for path in paths:
        if filepath.endswith(path):
            return filepath[0: filepath.index(path)]
    return filepath


def _get_path(dir=None):
    """
    Helper function to get a path from the project configuration file.
    """
    homedir = _get_home_path(os.getcwd())
    config_path = os.path.join(homedir, "config.txt")

    with open(config_path) as configfile:
        config = json.load(configfile)

    path = config[dir]
    return path
