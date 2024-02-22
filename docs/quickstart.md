# Dataramp Quick Start Guide

Welcome to Dataramp – your go-to library for data science projects! This guide will walk you through the steps to get started with Dataramp in your data science workflows.

## Installation

```bash
pip install dataramp 
```
To upgrade an existing installation of Dataramp, use:

```bash 
pip install --upgrade dataramp
``` 

## Getting Started
Once installed, you can import the library and explore its functionality:

```bash
import dataramp as dr
```
## Creating a New Project

To create a new project using Dataramp, run:

```bash
dr.core.create_project("project-name")
```
This will create a project with a structured directory layout to kickstart your project.

## Project Directory Structure

```bash 
project-name/
├── datasets
│   └── dataset.csv
├── outputs
│   └── models
├── README.md
└── src
    ├── notebooks
    │   └── notebook.ipynb
    └── scripts
        ├── ingest
        └── tests
```

## Sample Usage
```python
import dataramp as dr  # import the dataramp library
import pandas as pd

from dataramp.utils import (
    describe_df,
    get_cat_vars,
    feature_summary,
    display_missing,
    get_unique_counts,
)

df = pd.read_csv("data/iris.csv")  # load iris dataset

df.head() #  Snapshot of your df

missing = display_missing(df)
print(missing)
```