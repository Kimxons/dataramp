#!/bin/bash -e

rm -rf dist build *.egg-info

python3 setup.py sdist bdist_wheel

if [ -z "$PYPI_API_TOKEN" ]; then
  echo "Error: PYPI_API_TOKEN environment variable is not set."
  exit 1
fi

twine upload -u __token__ -p "$PYPI_API_TOKEN" dist/*
