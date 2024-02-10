#!/bin/bash

set -euo pipefail 

function print_header() {
  local header_text=$1
  echo -e "\n\e[1;36m=== $header_text ===\e[0m\n"
}

function command_exists() {
  command -v "$1" >/dev/null 2>&1
}

for cmd in black usort flake8 pylint; do
  if ! command_exists "$cmd"; then
    echo -e "\e[1;31mError: $cmd not found. Please install it before running the script.\e[0m" >&2
    exit 1
  fi
done

BLACK_VERSION=$(black --version)
USORT_VERSION=$(usort --version)
FLAKE8_VERSION=$(flake8 --version)

print_header "Code Formatting"
echo "BLACK_VERSION: $BLACK_VERSION"
echo "USORT_VERSION: $USORT_VERSION"

print_header "Code Linting"
echo "FLAKE8_VERSION: $FLAKE8_VERSION"

DIRECTORIES_TO_LINT=("examples" "datahelp")

function format_directories() {
  for directory in "${DIRECTORIES_TO_LINT[@]}"; do
    echo -e "\n\e[1;34mFormatting directory: $directory\e[0m"
    if [ -n "$(find "$directory" -maxdepth 1 -name '*.py' -print -quit)" ]; then
      usort format "$directory"
      black "$directory"
    else
      echo "No Python files are present to be formatted. Nothing to do 😴"
    fi
  done
}

function lint_files() {
  local module=$1
  local skip_flags=${2:-""} 
  print_header "Linting Module: $module"
  if [ -n "$(find "$module" -name '*.py' -print -quit)" ]; then
    pylint --disable="$skip_flags" "$module" || true
  else
    echo -e "\e[1;32mNo Python files to lint in module $module.\e[0m"
  fi
}

function lint_changed_files() {
  local module=$1
  local skip_flags=${2:-""} 
  local files
  files=$(git status -s "$module" | grep -v "^D" | awk '{print $NF}' | grep .py$ || true)
  if [ -n "$files" ]; then
    print_header "Linting Changed Files in Module: $module"
    for file in $files; do
      echo -e "\e[1;33m$file\e[0m"
      pylint --disable="$skip_flags" "$file" || true
    done
  else
    echo -e "\e[1;32mNo changed files in module $module.\e[0m"
  fi
}

function main() {
  format_directories

  for module in "${DIRECTORIES_TO_LINT[@]}"; do
    lint_files "$module"
    lint_changed_files "$module"
  done
}

CHECK_ALL=false

main
