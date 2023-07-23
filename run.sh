#!/bin/bash

# Exit on any error
set -e

# Check if all required packages are installed
python scripts/check_requirements.py requirements.txt

# If any package is missing, install it using pip
if [[ $? -eq 1 ]]; then
    echo "Installing missing packages"
    pip install -r requirements.txt
fi

# Run the datahelp module with any command-line arguments passed to the script
python -m datahelp "$@"

# Prompt the user to press Enter to continue
read -p "Press Enter or any key to continue..."
