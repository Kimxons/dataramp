# Dataramp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pylint](https://img.shields.io/badge/pylint-enabled-brightgreen.svg)](https://github.com/PyCQA/pylint)
[![Flake8](https://img.shields.io/badge/flake8-enabled-blue.svg)](https://flake8.pycqa.org/en/latest/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v0.24.2-blue)](https://scikit-learn.org/stable/)


**Dataramp** is a Python library designed to streamline data science and data analysis workflows. It offers a collection of utility functions and tools tailored to assist data science teams in various aspects of their projects.

## Key Features

### 1. Project Management
   - Simplify project setup with a single function call to generate a standardized project directory structure.
   - Organize datasets, model outputs, scripts, notebooks, and more in predefined folders for better project management.

### 2. Model Saving and Loading
   - Save and load trained machine learning models effortlessly.
   - Supports multiple formats including joblib, pickle, and keras for compatibility with diverse model types.

### 3. Data Exploration and Visualization
   - Explore datasets and generate summary statistics with ease.
   - Visualize feature distributions and missing data patterns to gain insights into your data.

### 4. Feature Engineering
   - Handle missing data and outliers effectively.
   - Drop missing columns based on user-defined thresholds and detect outliers using Tukey's Interquartile Range (IQR) method.

### 5. Model Evaluation and Cross-Validation
   - Evaluate model performance with comprehensive metrics such as accuracy, F1-score, precision, and recall.
   - Generate classification reports and support cross-validation for robust model evaluation.

### 6. Scaling and Normalization
   - Scale and normalize data using min-max scaling and z-score normalization techniques.
   - Bring features to a common scale for improved model performance.

By providing a range of functionalities, Dataramp aims to enhance productivity and efficiency in data science projects, empowering teams to focus on deriving meaningful insights from their data.


## Quickstart
To get started with Dataramp in your data science projects, follow these simple steps:

You can install Dataramp via pip:

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
## Project Links
- GitHub Repository: [dataramp](https://github.com/kimxons/dataramp)
- PyPI Package: [dataramp](https://pypi.org/project/dataramp/)
