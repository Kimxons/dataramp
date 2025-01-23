import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "dataramp"
author = "Meshack Kitonga"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
