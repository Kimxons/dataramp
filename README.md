# Dataramp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pylint](https://img.shields.io/badge/pylint-enabled-brightgreen.svg)](https://github.com/PyCQA/pylint)
[![Flake8](https://img.shields.io/badge/flake8-enabled-blue.svg)](https://flake8.pycqa.org/en/latest/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-v0.24.2-blue)](https://scikit-learn.org/stable/)

Welcome to the Dataramp documentation! Here you will find information about Dataramp, including some examples to get you started.

## Dataramp

Dataramp is a Python library designed to streamline data science and data analysis workflows. It offers a collection of utility functions and tools tailored to assist data science teams in various aspects of their projects.

By providing a range of functionalities, Dataramp aims to enhance productivity and efficiency in data science projects, empowering teams to focus on deriving meaningful insights from their data.

## Getting Started

Read the quick start guide [here](docs/quickstart.md).

If you want to see some examples, you can look at the examples in the [examples](examples/) directory.

You can install Dataramp and learn more from [PyPi](https://pypi.org/project/dataramp/).


# Example
```python
# Create and register a model pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer())
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])

model_save(pipeline, "classifier", method="joblib", metadata={"dataset": "2023_sales"})
register_model(
    pipeline,
    name="sales_classifier",
    version="v1.0",
    metadata={
        "metrics": {"accuracy": 0.89},
        "serialization_method": "joblib"
    }
)

# Create versioned dataset
df = pd.read_csv("data.csv")
data_save(df, "processed_data", versioning=True, description="Initial cleaned version")
```

# Potential Use Cases
- Data Science Projects : Initialize projects with a standardized structure and manage datasets and models effectively.
- Team Collaboration : Facilitate collaboration by providing clear project organization and versioning.
- Reproducibility : Ensure reproducibility by tracking dataset versions, model metadata, and dependencies.
- Automation : Integrate into CI/CD pipelines for automated testing, deployment, and dependency updates.
