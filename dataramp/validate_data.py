"""Module for validating data using Pandera schemas with detailed error reporting.

This module provides functionality to validate pandas DataFrames against schema
definitions, with support for sampling and error reporting.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DataSchema(pa.SchemaModel):
    """Base schema for data validation. Extend this for custom datasets."""

    class Config:
        """Configuration for data validation schema.

        Attributes:
            coerce: Automatically coerce data types
            strict: Allow extra columns in the data
        """

        coerce = True  # Automatically coerce dtypes
        strict = False  # Allow extra columns


def validate_data(
    df: pd.DataFrame,
    schema: pa.SchemaModel = DataSchema,
    sample_size: Optional[int] = 1000,
    report_path: Optional[Path] = None,
) -> bool:
    """Validate DataFrame against a schema with detailed error reporting.

    Args:
        df: DataFrame to validate
        schema: Pandera schema (default: basic schema)
        sample_size: Number of rows to sample for validation (None=all)
        report_path: Path to save validation report

    Returns:
        bool: True if validation passes

    Example:
        class MySchema(DataSchema):
            age = pa.Column(int, pa.Check.ge(0))
            income = pa.Column(float, pa.Check.ge(0))

        validate_data(df, MySchema)
    """
    try:
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(min(sample_size, len(df)))
        else:
            df_sample = df

        # Validate
        validated_df = schema.validate(df_sample, lazy=True)

        # Generate data fingerprint
        fingerprint = hashlib.md5(
            pd.util.hash_pandas_object(validated_df).values
        ).hexdigest()

        logger.info(f"Data validation passed. Fingerprint: {fingerprint}")
        return True

    except pa.errors.SchemaErrors as err:
        logger.error(f"Data validation failed: {err.failure_cases}")

        # Save error report
        if report_path:
            error_report = {
                "failure_cases": err.failure_cases.to_dict(),
                "schema": schema.to_schema().to_json(),
                "data_sample": df_sample.head(100).to_dict(),
            }
            with open(report_path, "w") as f:
                json.dump(error_report, f, indent=2)

        raise ValueError("Data validation failed") from err
