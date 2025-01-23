#!/usr/bin/env python

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


def read_file(path):
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
)
