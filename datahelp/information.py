from __version__ import __version__


def print_header():
    """
    Print the header information including the version and copyright details.
    """
    header = (
        f"datahelp (Version {__version__})\n"
        "Copyright (c) 2023 Meshack Kitonga. Licensed under License 1.0."
        "\n"
    )

    print(header)


print_header()
