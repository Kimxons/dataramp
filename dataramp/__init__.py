"""DataRamp package for data manipulation and transformation."""

import subprocess
from typing import Optional

__version__ = "0.3.4"


def get_git_revision() -> Optional[str]:
    try:
        git_revision = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return git_revision
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def append_git_revision(version: str) -> str:
    git_revision = get_git_revision()
    if git_revision:
        version += f".dev{git_revision}"
    return version


__version__ = append_git_revision(__version__)

__all__ = ["__version__"]
