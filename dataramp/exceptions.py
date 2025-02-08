"""Exception classes for handling data loading errors in the dataramp package."""


class DataLoadError(Exception):
    """Base exception for data loading errors."""


class SecurityValidationError(DataLoadError):
    """Security validation failure."""


class EmptyDataError(DataLoadError):
    """Empty file or query result."""
