Dataramp Quick Start Guide
===========================

Welcome to Dataramp - Your go-to library for data science projects! This guide will walk you through the steps to get started with Dataramp in your data science workflows.

Installation
------------

.. code-block:: bash

    pip install dataramp

To upgrade an existing installation of Dataramp, use:

.. code-block:: bash

    pip install --upgrade dataramp

Getting Started
---------------

Once installed, you can import the library and explore its functionality:

.. code-block:: python

    import dataramp as dr

Creating a New Project
----------------------

To create a new project using Dataramp, run:

.. code-block:: bash

    dr.core.create_project("project-name")

This will create a project with a structured directory layout to kickstart your project.

Project Directory Structure
---------------------------

.. code-block:: bash

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

Sample Usage
------------

.. code-block:: python

    import pandas as pd

    import dataramp as dr  # import the dataramp library
    from dataramp.utils import describe_df, display_missing, feature_summary, get_cat_vars, get_unique_counts

    df = pd.read_csv("data/iris.csv")  # load iris dataset

    df.head()  # Snapshot of your df

    missing = display_missing(df)
    print(missing)
