#!/usr/bin/env python
"""Setup script for the dataramp package."""

from __future__ import print_function

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 7)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.exit(f"Sorry, Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ is required.")


def get_version():
    """Get the package version from __init__.py file.

    Returns:
        str: The package version string.
    """
    with open("dataramp/__init__.py", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip().strip('"')


if not get_version():
    raise RuntimeError("Unable to find version string.")


def read_file(path):
    """Read and return the contents of a file.

    Args:
        path (str): Path to the file to be read.

    Returns:
        str: Contents of the file.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


long_description = read_file("README.md")

setup(
    name="dataramp",
    description="A Data science library for data science / data analysis teams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Meshack Kitonga",
    author_email="kitongameshack9@gmail.com",
    url="https://github.com/Kimxons/dataramp",
    keywords=["data science", "machine learning", "data analysis"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=["dataramp"],
    include_package_data=True,
    zip_safe=False,
)
