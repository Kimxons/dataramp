#!/usr/bin/env python

from __future__ import print_function

import contextlib
import os
import pathlib
import subprocess
import sys

import pypandoc

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 7)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.exit("Sorry, Python {}.{}+ is required".format(*REQUIRED_PYTHON))

def write_version_py():
    with open(os.path.join("dataramp", "version.txt")) as f:
        version = f.read().strip()

    with contextlib.suppress(Exception): # TODO: Change this handle exceptions appropriately not to suppress them
        if num_commits := (
            subprocess.check_output(["git", "rev-list", "--count", "HEAD"])
            .decode("ascii")
            .strip()
        ):
            version += f".dev{num_commits}"
    # To write version info to dataramp/version.py
    with open(os.path.join("dataramp", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version

def read_file(path):
    # if this fails on windows then add the following environment variable (PYTHONUTF8=1)
    with open(path) as contents:
        return contents.read()

version = write_version_py()

# Read the contents of the requirements_dev file
def list_reqs(fname='requirements_dev.txt'):
    with open(fname, encoding='utf-8') as fd:
        return fd.read().splitlines()

# Convert Markdown to RST for PyPI
# Credits: http://stackoverflow.com/a/26737672

try:
    pypandoc_func = (
        pypandoc.convert_file if hasattr(pypandoc, "convert_file") else pypandoc.convert
    )
    long_description = pypandoc_func("README.rst", "rst")
except (IOError, ImportError, OSError):
    long_description = read_file("README.rst")

long_description = pathlib.Path("README.rst").read_text()

setup(
    name="dataramp",
    version=version,
    license="MIT",
    description="A Data science library for data science / data analysis teams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Meshack Kitonga",
    author_email="kitongameshack9@gmail.com",
    url="",
    keywords=["dataramp", "Data Science", "Data Analysis"],
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
)
