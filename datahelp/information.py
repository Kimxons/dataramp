"""
Datahelp: A Python library for Data Analysts/Scientists.

Author: {__author__}
Contact: {__author_email__}
"""

from __version__ import __author__, __author_email__, __version__


def print_header():
    """
    Print the header information including the version and copyright details.
    """
    header = (
        f"Datahelp Version {__version__}\n"
        f"Copyright (c) 2023 {__author__}, {__author_email__}.\nLicensed under License 1.0."
        "\n"
    )

    print(header)


print_header()
