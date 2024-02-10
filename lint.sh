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

function format_and_lint() {
  local current_dir=$(pwd)
  print_header "Formatting and Linting Current Directory: $current_dir"

  usort format "$current_dir"
  black "$current_dir"

  for file in "$current_dir"/*.py; do
    [ -e "$file" ] || continue
    pylint "$file" || true
  done

  local changed_files
  changed_files=$(git status -s "$current_dir" | grep -v "^D" | awk '{print $NF}' | grep .py$ || true)
  if [ -n "$changed_files" ]; then
    print_header "Linting Changed Files in Current Directory: $current_dir"
    for file in $changed_files; do
      [ -e "$file" ] || continue
      echo -e "\e[1;33m$file\e[0m"
      pylint "$file" || true
    done
  else
    echo -e "\e[1;32mNo changed files in the current directory.\e[0m"
  fi
}

function main() {
  format_and_lint
}

main
