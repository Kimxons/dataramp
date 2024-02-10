#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path
import re

def get_version(module):
    """
    Return the version of the module without loading the whole module: as lised in the __version__ attribute
    """
    with open(module, "r") as fh:
        content = fh.read()
    return re.search(r'__version__ = ["\']([^"\']+)["\']', content).group(1)

VERSION = get_version("datahelp/__version__.py")

NAME = "datahelp"
DESCRIPTION = "A Data science library for data science / data analysis teams"
AUTHOR = "Meshack Kitonga"
URL = "https://github.com/kimxons/datahelp"
VERSION = None

about = {}
root = Path(__file__).resolve().parent

with open(root / "README.md", "r") as fh:
    about["long_description"] = fh.read()

if not VERSION:
    with open(root / "datahelp" / "__version__.py") as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

requirements = (root / "requirements_dev.txt").read_text().splitlines()

setup(
    name=NAME,
    version=about["__version__"],
    license="MIT",
    description=DESCRIPTION,
    long_description=about["long_description"],
    long_description_content_type="text/markdown",
    author=AUTHOR,
    url=URL,
    keywords=["Datahelp", "Data Science", "Data Analysis"],
    install_requires=requirements,
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
