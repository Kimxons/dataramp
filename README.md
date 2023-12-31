# datahelp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**datahelp** is a Python library designed to assist data science and data analysis teams in their workflow. It provides various utility functions and tools to streamline common data science tasks.

## Features

datahelp offers the following key features:

1. **Project Management:** Simplifies the creation of standard data science project structures. With a single function call, you can generate a well-organized project directory with predefined folders for datasets, processed data, raw data, outputs, models, scripts, notebooks, and more.

2. **Model Saving and Loading:** Provides easy-to-use functions for saving and loading trained machine learning models. It supports various formats such as joblib, pickle, and keras, enabling seamless integration with different model types.

3. **Data Exploration and Visualization:** Includes functions for data exploration, summary statistics, and visualization. Quickly generate feature vi plots and visualize missing data to gain insights into your datasets.

4. **Feature Engineering:** Methods for handling missing data and noise in your datasets. Offers functions for dropping missing columns based on a specified threshold and detecting outliers using Tukey's Interquartile Range (IQR) method.

5. **Model Evaluation and Cross-Validation:** Provides tools to evaluate model performance, including functions to calculate accuracy, F1-score, precision, recall, and generate classification reports. Also supports cross-validation for model evaluation.

6. **Scaling and Normalization:** Offers functions for min-max scaling and z-score normalization of data to bring features to a common scale.

## Quickstart

To use datahelp in your data science projects, you can install it via pip:

```bash
pip install datahelp
```

Once installed, you can import the library and explore its functionality:

```python
import datahelp as dh  # import the datahelp library

df = pd.read_csv("data/iris.csv")  # load iris dataset

df.head()

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
Project: https://github.com/kimxons/datahelp
PyPi: https://pypi.org/project/datahelp/

## Documentation

For detailed usage instructions and API reference, please refer to the official documentation at [https://datahelp-docs.example.com](https://datahelp-docs.example.com)

We use SemVer for versioning

## Contribution

datahelp is an open-source project, and we welcome contributions from the data science community. If you find a bug, have a feature request, or want to contribute improvements, please open an issue or submit a pull request on our GitHub repository at [https://github.com/kimxons/datahelp](https://github.com/kimxons/datahelp).

## License

datahelp is licensed under the MIT License. See the [LICENSE](https://github.com/datahelp/datahelp/blob/main/LICENSE) file for more details.

## Contact

If you have any questions or feedback, feel free to reach out to our support team at dev.kitonga@gmail.com or join our community forum at [https://community.datahelp.com](https://community.datahelp.com). We are here to assist you in making your data science journey smooth and successful!
