#!/bin/bash -e

poetry install

poetry build

poetry publish --build

poetry env remove
