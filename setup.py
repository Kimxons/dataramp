import io
from os.path import dirname, join

from setuptools import find_packages, setup


def read(*names, **kwargs):
    try:
        with io.open(join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fr:
            return fr.read()
    except (IOError, UnicodeError) as e:
        print(f"Error reading the file: {e}")
        return None


setup(
    name="data-help",
    version="0.1.2",
    license="MIT",
    description="A python library for easy modelling",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Meshack Kitonga",
    author_email="dev.kitonga@gmail.com",
    url="https://github.com/data_help",
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
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
