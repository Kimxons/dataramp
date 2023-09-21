#!/usr/bin/env python3

import os
from setuptools import find_packages, setup
from datahelp.__version__ import __version__

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="datahelp",
    version=__version__,
    license="MIT",
    description="Data science library for data science / data analysis teams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Meshack Kitonga",
    author_email="dev.kitonga@gmail.com",
    url="https://github.com/kimxons/datahelp",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "seaborn",
        "numpy",
        "jupyter",
        "matplotlib",
        "nltk",
        "joblib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
)
