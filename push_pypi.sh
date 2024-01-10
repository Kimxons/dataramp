#!/bin/bash 

set -e

pip install wheel twine

PACKAGE_NAME="datahelp"
VERSION="0.1.0"

rm -rf dist

python -m build

# python setup.py sdist bdist_wheel

twine upload dist/*

echo "Execution completed!"
