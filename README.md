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

To use dataramp in your data science projects, you can install it via pip:

```bash
pip install dataramp
```

Once installed, you can import the library and explore its functionality:

```python
import dataramp as dr  # import the dataramp library
import pandas as pd

# To create your project with help of dataramp
from dataramp.create_project import create_project



df = pd.read_csv("data/iris.csv")  # load iris dataset

df.head() #  Snapshot of your df


cats = dh.eda.get_cat_vars(df)
print(cats)

num_var = dh.eda.get_num_vars(df)
print(num_var)

cat_count = dh.eda.get_cat_counts(df)
cat_count

missing = dh.eda.display_missing(df)
missing
```
## Lins
Project: https://github.com/kimxons/dataramp
PyPi: https://pypi.org/project/dataramp/

## Documentation

For detailed usage instructions and API reference, please refer to the official documentation at [https://dataramp-docs.example.com](https://dataramp-docs.example.com)

We use SemVer for versioning

## Contribution

dataramp is an open-source project, and we welcome contributions from the data science community. If you find a bug, have a feature request, or want to contribute improvements, please open an issue or submit a pull request on our GitHub repository at [https://github.com/kimxons/dataramp](https://github.com/kimxons/dataramp).

## License

dataramp is licensed under the MIT License. See the [LICENSE](https://github.com/dataramp/dataramp/blob/main/LICENSE) file for more details.

## Contact

If you have any questions or feedback, feel free to reach out to our support team at dev.kitonga@gmail.com or join our community forum at [https://community.dataramp.com](https://community.dataramp.com). We are here to assist you in making your data science journey smooth and successful!
