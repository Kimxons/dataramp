"""Library information."""

__version__ = "0.1.7"
__author__ = "Meshack Kitonga"
__author_email__ = "kitogameshack@gmail.com"


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
