#!/bin/bash -e

python3 -m venv venv
source venv/bin/activate

pip install wheel setuptools twine
python3 setup.py clean

rm -rf dist

python3 setup.py sdist bdist_wheel

twine upload dist/*

deactivate

rm -rf venv
