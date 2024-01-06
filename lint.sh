#!/bin/bash

set -ex # exit on first error & display each command as it executed

black --version  # 22.6.0 (on Python 3.9) - code formatting
usort --version  # 1.0.4 - sort imports
flake8 --version  # 5.0.4 - code linting

usort format datahelp
usort format tests
usort format examples
usort format setup.py

black datahelp
black tests
black examples
black setup.py

flake8 --max-line-length 88 --ignore E203,E402,W503,W504,F821,E501 datahelp
flake8 --max-line-length 88 --ignore E203,E402,W503,W504,F821,E501 tests
flake8 --max-line-length 88 --ignore E203,E402,W503,W504,F821,E501 examples
flake8 --max-line-length 88 --ignore E203,E402,W503,W504,F821,E501 setup.py

function lintAllFiles () {
  echo "Running linter on module $1"
  pylint --disable=$2 $1
}

function lintChangedFiles () {
  files=`git status -s $1 | (grep -v "^D") | awk '{print $NF}' | (grep .py$ || true)`
  for f in $files
  do
    echo "Running linter on $f"
    pylint --disable=$2 $f
  done
}

set -o errexit
set -o nounset

SKIP_FOR_TESTS="redefined-outer-name,protected-access,missing-docstring,too-many-lines,len-as-condition"
SKIP_FOR_SNIPPETS="${SKIP_FOR_TESTS},reimported,unused-variable,unused-import,import-outside-toplevel"

if [[ "$#" -eq 1 && "$1" = "all" ]]
then
  CHECK_ALL=true
elif [[ "$#" -eq  0 ]]
then
  CHECK_ALL=false
else
  # echo "Usage: "
  # echo "   chmod +x lint.sh"
  echo "   ./lint.sh [all]"
  exit 1
fi

if [[ "$CHECK_ALL" = true ]]
then
  lintAllFiles "datahelp" ""
  lintAllFiles "tests" "$SKIP_FOR_TESTS"
  lintAllFiles "integration" "$SKIP_FOR_TESTS"
  lintAllFiles "snippets" "$SKIP_FOR_SNIPPETS"
else
  lintChangedFiles "datahelp" ""
  lintChangedFiles "tests" "$SKIP_FOR_TESTS"
  lintChangedFiles "integration" "$SKIP_FOR_TESTS"
  lintChangedFiles "snippets" "$SKIP_FOR_SNIPPETS"
fi
