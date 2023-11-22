#!/usr/bin/env python

from io import open
from setuptools import find_packages, setup

version_info = {}

DESC = " A Data science library for data science / data analysis teams "

# read __version__.py as bytes, otherwise exec will complain about
# 'coding: utf-8', which we want there for the normal Python 2 import
try:
    with open('datahelp/__version__.py', 'rb') as fp:
        exec(fp.read(), version_info)
except FileNotFoundError:
    pass

# defines __version__
exec(open("datahelp/__version__.py").read())

LONG_DESC = open("README.md").read()

setup(
    name="datahelp",
    version=__version__,
    description=DESC,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://datahelp.readthedocs.io/en/latest/",
    },
    license="MIT",
    packages=find_packages(),
    url="https://github.com/datahelp/datahelp",
    install_requires=[
        "numpy >= 1.26.2",
        "pandas >= 2.1.3",
        "scikit-learn >= 1.3.2",
        "joblib >= 1.3.2",
        "matplotlib >= 3.8.2",
        "seaborn >= 0.13.0",
        "nltk >= 3.8.1",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov", "scipy"],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)
