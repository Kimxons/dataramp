#!/usr/bin/env python

import sys
import os
import subprocess
from setuptools import find_packages, setup

if sys.version_info < (3, 7): sys.exit("Sorry, Python >= 3.7 is required")

def write__version_py():
    with open(os.path.join("datahelp", "version.txt")) as f:
        version = f.read().strip()

    # append latest commit hash to version string
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except Exception:
        pass

    # write version info to datahelp/version.py
    with open(os.path.join("datahelp", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version

version = write__version_py()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="datahelp",
    version=version,
    license="MIT",
    description="A Data science library for data science / data analysis teams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Meshack Kitonga",
    author_email="kitongameshack9@gmail.com",
    url="",
    keywords=["Datahelp", "Data Science", "Data Analysis"],
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
