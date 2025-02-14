"""Exception classes for handling data loading errors in the dataramp package."""


class DataLoadError(Exception):
    """Base exception for data loading errors."""

    pass


class SecurityValidationError(DataLoadError):
    """Security validation failure."""

    pass


class EmptyDataError(DataLoadError):
    """Empty file or query result."""

    pass


class VisualizationError(Exception):
    """Custom exception for visualization-related errors."""

    pass
