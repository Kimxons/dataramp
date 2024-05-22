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
    version = ""
    try:
        with open(os.path.join("dataramp", "version.txt")) as f:
            version = f.read().strip()
    except FileNotFoundError:
        sys.exit("version.txt not found!")

    try:
        if (
            num_commits := subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"]
            )
            .decode("ascii")
            .strip()
        ):
            version += f".dev{num_commits}"
    except subprocess.CalledProcessError:
        print("Git command failed, version without commit count used.")

    with open(os.path.join("dataramp", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version

def read_file(path):
    with open(path, encoding='utf-8') as contents:
        return contents.read()

version = write_version_py()

def list_reqs(fname='requirements_dev.txt'):
    with open(fname, encoding='utf-8') as fd:
        return fd.read().splitlines()

def get_long_description():
    with contextlib.suppress(IOError, ImportError, OSError):
        if pathlib.Path("README.md").exists():
            pypandoc_func = (
                pypandoc.convert_file if hasattr(pypandoc, "convert_file") else pypandoc.convert
            )
            return pypandoc_func("README.md", "rst")
        elif pathlib.Path("README.rst").exists():
            return pathlib.Path("README.rst").read_text(encoding='utf-8')
    return ""

long_description = get_long_description()

setup(
    name="dataramp",
    version=version,
    license="MIT",
    description="A Data science library for data science / data analysis teams",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Meshack Kitonga",
    author_email="kitongameshack9@gmail.com",
    url="",
    keywords=["dataramp", "Data Science", "Data Analysis"],
    packages=find_packages(exclude=("tests",)),
    classifiers=[
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
