#!/usr/bin/env python

import subprocess
import sys


def check_requirements(requirements_file):
    try:
        with open(requirements_file, "r") as file:
            requirements = file.read().splitlines()

        missing_packages = []
        for requirement in requirements:
            try:
                subprocess.check_output([sys.executable, "-m", "pip", "show", requirement])
            except subprocess.CalledProcessError:
                missing_packages.append(requirement)

        return missing_packages

    except FileNotFoundError:
        print(f"Error: {requirements_file} not found.")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_requirements.py requirements.txt")
    else:
        requirements_file = sys.argv[1]
        missing_packages = check_requirements(requirements_file)
        if missing_packages:
            print("The following packages are missing:")
            for package in missing_packages:
                print(f"- {package}")
            sys.exit(1)
        else:
            print("All required packages are installed.")
