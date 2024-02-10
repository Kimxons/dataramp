#!/usr/bin/env python

from setuptools import setup, find_packages
import re

def get_version(module):
    """
    Return the version of the module without loading the whole module: as listed in the __version__ attribute
    """
    version_info = open('{0}.py'.format(module)).read()
    return re.search(r'__version__ = ["\']([^"\']+)["\']', version_info).group(1)

version = get_version('info')

name = "datahelp"
description = "A Data science library for data science / data analysis teams"
author = "Meshack Kitonga"
author_email = "kitongameshack9@gmail.com"

setup(
    name=name,
    version=version,
    license="MIT",
    description=description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    url="",
    keywords=["Datahelp", "Data Science", "Data Analysis"],
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
