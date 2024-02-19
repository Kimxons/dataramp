#!/usr/bin/env python

import os
import subprocess
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.exit("Sorry, Python >= 3.7 is required")


def write_version_py():
    with open(os.path.join("datahelp", "version.txt")) as f:
        version = f.read().strip()

    # append latest commit hash to version string
    try:
        num_commits = (
            subprocess.check_output(["git", "rev-list", "--count", "HEAD"])
            # .strip()
            .decode("ascii")
            .strip()
        )
        # version += "+" + num_commits[:7] # this throws an error while uploading on pypi - bad version as per PEP 440
    except Exception:
        # num_commits = 0
        pass

    if num_commits:
        version += f".dev{num_commits}"

    # write version info to datahelp/version.py
    with open(os.path.join("datahelp", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version
    
# def write_version_py():
#     with open(os.path.join("datahelp", "version.txt")) as f:
#         version = f.read().strip()

#     # Get the latest commit hash
#     try:
#         num_commits = (
#             subprocess.check_output(["git", "rev-parse", "HEAD"])
#             .decode("ascii")
#             .strip()
#         )
#         version_with_commit_hash = version + ".dev0"
#     except Exception:
#         version_with_commit_hash = version + ".dev0"

#     # Write version info to datahelp/version.py
#     with open(os.path.join("datahelp", "version.py"), "w") as f:
#         f.write('__version__ = "{}"\n'.format(version_with_commit_hash))

#     return version_with_commit_hash

version = write_version_py()

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
