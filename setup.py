#!/usr/bin/env python

import os
import subprocess
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.exit("Sorry, Python >= 3.7 is required")


def write_version_py():
    with open(os.path.join("dataramp", "version.txt")) as f:
        version = f.read().strip()

    try:
        num_commits = (
            subprocess.check_output(["git", "rev-list", "--count", "HEAD"])
            .decode("ascii")
            .strip()
        )
        if num_commits:
            version += f".dev{num_commits}"
    except Exception:
        pass

    # Write version info to dataramp/version.py
    with open(os.path.join("dataramp", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()

with open("README.md") as f:
    long_description = f.read()

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
