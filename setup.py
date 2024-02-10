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

setup(
    name="datahelp",
    version=version,
    license="MIT",
    description="A Data science library for data science / data analysis teams",
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
