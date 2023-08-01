#!/bin/bash

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
  echo "Usage: ./lint.sh [all]"
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