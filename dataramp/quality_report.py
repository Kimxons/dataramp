"""Module for generating automated data quality reports with statistics and visualizations.

This module provides functionality to analyze pandas DataFrames and generate comprehensive
HTML reports containing data quality metrics, statistical analysis, and visualizations.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_data_quality_report(
    data: Union[pd.DataFrame, pd.Series],
    report_path: Optional[Union[str, Path]] = None,
    report_title: str = "Data Quality Report",
    sample_size: Optional[int] = 10_000,
    correlation_threshold: float = 0.9,
    image_format: str = "png",
) -> Path:
    """Generate an automated data quality report with statistics and visualizations.

    Args:
        data: Input DataFrame or Series to analyze
        report_path: Output path for the report (default: outputs/quality_reports)
        report_title: Title for the report
        sample_size: Number of rows to sample for visualizations
        correlation_threshold: Threshold for highlighting high correlations
        image_format: Format for embedded images (png|svg|jpg)

    Returns:
        Path to generated HTML report

    Example:
        generate_data_quality_report(df, "reports/data_quality.html")
    """
    # Set up paths and initialize report
    report_path = Path(report_path or get_path("output_path")) / "quality_reports"
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = (
        report_path / f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Sample data for visualizations
    analysis_data = data.sample(min(sample_size, len(data))) if sample_size else data

    # Generate report sections
    sections = {
        "overview": _generate_overview_section(data),
        "missing_values": _generate_missing_values_section(data),
        "data_types": _generate_datatype_section(data),
        "statistics": _generate_statistical_section(data),
        "correlations": _generate_correlation_section(data, correlation_threshold),
        "duplicates": _generate_duplicate_section(data),
        "distributions": _generate_distribution_visualizations(
            analysis_data, image_format
        ),
    }

    # Create HTML report
    html_content = _create_html_report(report_title, sections)

    with open(report_file, "w") as f:
        f.write(html_content)

    logger.info(f"Generated data quality report at {report_file}")
    return report_file


def _generate_overview_section(df: pd.DataFrame) -> dict:
    """Generate dataset overview statistics."""
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "total_missing": df.isnull().sum().sum(),
        "total_duplicates": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1e6,  # MB
    }


def _generate_missing_values_section(df: pd.DataFrame) -> dict:
    """Analyze missing values patterns."""
    missing = df.isnull().sum()
    return {
        "columns": missing[missing > 0].to_dict(),
        "total": missing.sum(),
        "patterns": {
            "complete_rows": len(df.dropna()),
            "complete_cols": missing[missing == 0].index.tolist(),
        },
    }


def _generate_datatype_section(df: pd.DataFrame) -> dict:
    """Analyze data types and potential type mismatches."""
    type_counts = df.dtypes.value_counts().to_dict()
    return {"dtype_distribution": type_counts, "type_issues": _detect_type_issues(df)}


def _detect_type_issues(df: pd.DataFrame) -> dict:
    """Detect potential data type issues."""
    issues = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == "object":
            try:
                pd.to_numeric(df[col])
                issues[col] = "Numeric values stored as text"
            except ValueError:
                pass
        elif np.issubdtype(col_dtype, np.number):
            if (df[col] % 1 != 0).any():
                issues[col] = (
                    "Floats stored as integers"
                    if np.issubdtype(col_dtype, np.integer)
                    else None
                )
    return issues


def _generate_statistical_section(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics with anomaly detection."""
    stats = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "zeros": (df[col] == 0).sum(),
                "negatives": (
                    (df[col] < 0).sum()
                    if np.issubdtype(df[col].dtype, np.signedinteger)
                    else 0
                ),
                "outliers": _detect_outliers(df[col]),
            }
    return stats


def _detect_outliers(series: pd.Series) -> dict:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        "lower": (series < (q1 - 1.5 * iqr)).sum(),
        "upper": (series > (q3 + 1.5 * iqr)).sum(),
    }


def _generate_correlation_section(df: pd.DataFrame, threshold: float) -> dict:
    """Analyze column correlations."""
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        return {}

    corr_matrix = numeric_df.corr().abs()
    high_corr = (corr_matrix > threshold) & (corr_matrix < 1.0)
    return {
        "high_correlations": corr_matrix[high_corr].stack().to_dict(),
        "correlation_matrix": _plot_to_base64(
            corr_matrix, "correlation_heatmap", "png"
        ),
    }


def _generate_duplicate_section(df: pd.DataFrame) -> dict:
    """Analyze duplicate rows and columns."""
    return {
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_cols": _find_duplicate_columns(df),
    }


def _find_duplicate_columns(df: pd.DataFrame) -> list:
    """Identify duplicated columns."""
    seen = {}
    duplicates = []
    for col in df.columns:
        col_hash = hash(tuple(df[col]))
        if col_hash in seen:
            duplicates.append((seen[col_hash], col))
        else:
            seen[col_hash] = col
    return duplicates


def _generate_distribution_visualizations(df: pd.DataFrame, fmt: str) -> dict:
    """Generate distribution plots for each column."""
    plots = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            fig, ax = plt.subplots()
            df[col].plot(kind="hist", ax=ax)
            plots[col] = _plot_to_base64(fig, col, fmt)
            plt.close(fig)
        elif df[col].nunique() < 50:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax)
            plots[col] = _plot_to_base64(fig, col, fmt)
            plt.close(fig)
    return plots


def _plot_to_base64(fig: plt.Figure, title: str, fmt: str) -> str:
    """Convert matplotlib figure to base64 encoded image."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return f"data:image/{fmt};base64,{base64.b64encode(buf.read()).decode()}"


def _create_html_report(title: str, sections: dict) -> str:
    """Generate HTML report from analysis sections."""
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; }}
                .section {{ margin-bottom: 3em; border-bottom: 1px solid #ccc; padding-bottom: 2em; }}
                h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                img {{ max-width: 600px; margin: 1em 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="section">
                <h2>Dataset Overview</h2>
                <ul>
                    <li>Number of rows: {sections['overview']['num_rows']:,}</li>
                    <li>Number of columns: {sections['overview']['num_columns']}</li>
                    <li>Total missing values: {sections['overview']['total_missing']:,}</li>
                    <li>Duplicate rows: {sections['overview']['total_duplicates']:,}</li>
                    <li>Memory usage: {sections['overview']['memory_usage']:.2f} MB</li>
                </ul>
            </div>

            <div class="section">
                <h2>Data Types</h2>
                <table>
                    <tr><th>Data Type</th><th>Count</th></tr>
                    {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                    for k, v in sections['data_types']['dtype_distribution'].items())}
                </table>
            </div>

            <!-- Add other sections following the same pattern -->

        </body>
    </html>
    """


from pathlib import Path
from typing import Optional, Union


def assess_data_quality(
    data: Union[pd.DataFrame, pd.Series],
    report_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = 10_000,
    correlation_threshold: float = 0.9,
    image_format: str = "png",
) -> dict:
    """
    Perform a comprehensive data quality assessment and generate a report.

    Args:
        data: Input DataFrame or Series to analyze
        report_path: Output path for the report (default: outputs/quality_reports)
        sample_size: Number of rows to sample for visualizations
        correlation_threshold: Threshold for highlighting high correlations
        image_format: Format for embedded images (png|svg|jpg)

    Returns:
        dict: A dictionary containing the quality assessment summary and recommended actions

    Example:
        quality_report = assess_data_quality(df)
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Initialize quality assessment summary
    quality_summary = {
        "missing_values": _analyze_missing_values(data),
        "duplicates": _analyze_duplicates(data),
        "data_types": _analyze_data_types(data),
        "statistical_issues": _analyze_statistical_issues(data),
        "correlations": _analyze_correlations(data, correlation_threshold),
        "recommended_actions": [],
    }

    # Generate recommended actions
    if quality_summary["missing_values"]["total_missing"] > 0:
        quality_summary["recommended_actions"].append(
            "Handle missing values using imputation or removal."
        )

    if quality_summary["duplicates"]["total_duplicates"] > 0:
        quality_summary["recommended_actions"].append(
            "Remove duplicate rows or investigate their source."
        )

    if quality_summary["data_types"]["type_issues"]:
        quality_summary["recommended_actions"].append(
            "Fix data type inconsistencies (e.g., convert text to numeric)."
        )

    if quality_summary["statistical_issues"]["outliers"]:
        quality_summary["recommended_actions"].append(
            "Investigate and handle outliers using winsorization or removal."
        )

    if quality_summary["correlations"]["high_correlations"]:
        quality_summary["recommended_actions"].append(
            "Address high correlations by removing redundant features."
        )

    # Generate a report if a path is provided
    if report_path:
        report_path = Path(report_path)
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = (
            report_path
            / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        _generate_html_report(quality_summary, report_file, image_format)

    return quality_summary


def _analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values in the dataset."""
    missing = df.isnull().sum()
    return {
        "columns_with_missing": missing[missing > 0].to_dict(),
        "total_missing": missing.sum(),
        "complete_rows": len(df.dropna()),
        "complete_cols": missing[missing == 0].index.tolist(),
    }


def _analyze_duplicates(df: pd.DataFrame) -> dict:
    """Analyze duplicate rows and columns."""
    return {
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_cols": _find_duplicate_columns(df),
    }


def _find_duplicate_columns(df: pd.DataFrame) -> list:
    """Identify duplicated columns."""
    seen = {}
    duplicates = []
    for col in df.columns:
        col_hash = hash(tuple(df[col]))
        if col_hash in seen:
            duplicates.append((seen[col_hash], col))
        else:
            seen[col_hash] = col
    return duplicates


def _analyze_data_types(df: pd.DataFrame) -> dict:
    """Analyze data types and potential type mismatches."""
    type_counts = df.dtypes.value_counts().to_dict()
    return {"dtype_distribution": type_counts, "type_issues": _detect_type_issues(df)}


def _detect_type_issues(df: pd.DataFrame) -> dict:
    """Detect potential data type issues."""
    issues = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == "object":
            try:
                pd.to_numeric(df[col])
                issues[col] = "Numeric values stored as text"
            except:
                pass
        elif np.issubdtype(col_dtype, np.number):
            if (df[col] % 1 != 0).any():
                issues[col] = (
                    "Floats stored as integers"
                    if np.issubdtype(col_dtype, np.integer)
                    else None
                )
    return issues


def _analyze_statistical_issues(df: pd.DataFrame) -> dict:
    """Analyze statistical issues like outliers and invalid values."""
    stats = {}
    for col in df.select_dtypes(include=np.number).columns:
        stats[col] = {
            "outliers": _detect_outliers(df[col]),
            "zeros": (df[col] == 0).sum(),
            "negatives": (
                (df[col] < 0).sum()
                if np.issubdtype(df[col].dtype, np.signedinteger)
                else 0
            ),
        }
    return stats


def _detect_outliers(series: pd.Series) -> dict:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {
        "lower": (series < (q1 - 1.5 * iqr)).sum(),
        "upper": (series > (q3 + 1.5 * iqr)).sum(),
    }


def _analyze_correlations(df: pd.DataFrame, threshold: float) -> dict:
    """Analyze column correlations."""
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        return {}

    corr_matrix = numeric_df.corr().abs()
    high_corr = (corr_matrix > threshold) & (corr_matrix < 1.0)
    return {
        "high_correlations": corr_matrix[high_corr].stack().to_dict(),
        "correlation_matrix": _plot_to_base64(
            corr_matrix, "correlation_heatmap", "png"
        ),
    }


def _plot_to_base64(fig: plt.Figure, title: str, fmt: str) -> str:
    """Convert matplotlib figure to base64 encoded image."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return f"data:image/{fmt};base64,{base64.b64encode(buf.read()).decode()}"


def _generate_html_report(summary: dict, report_file: Path, image_format: str):
    """Generate an HTML report from the quality assessment summary."""
    html_content = f"""
    <html>
        <head>
            <title>Data Quality Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2em; }}
                .section {{ margin-bottom: 3em; border-bottom: 1px solid #ccc; padding-bottom: 2em; }}
                h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                img {{ max-width: 600px; margin: 1em 0; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Assessment Report</h1>
            <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="section">
                <h2>Missing Values</h2>
                <p>Total missing values: {summary['missing_values']['total_missing']}</p>
                <table>
                    <tr><th>Column</th><th>Missing Count</th></tr>
                    {"".join(f"<tr><td>{col}</td><td>{count}</td></tr>"
                    for col, count in summary['missing_values']['columns_with_missing'].items())}
                </table>
            </div>

            <div class="section">
                <h2>Duplicates</h2>
                <p>Duplicate rows: {summary['duplicates']['duplicate_rows']}</p>
                <p>Duplicate columns: {summary['duplicates']['duplicate_cols']}</p>
            </div>

            <div class="section">
                <h2>Data Types</h2>
                <p>Data type distribution:</p>
                <table>
                    <tr><th>Data Type</th><th>Count</th></tr>
                    {"".join(f"<tr><td>{dtype}</td><td>{count}</td></tr>"
                    for dtype, count in summary['data_types']['dtype_distribution'].items())}
                </table>
                <p>Type issues: {summary['data_types']['type_issues']}</p>
            </div>

            <div class="section">
                <h2>Statistical Issues</h2>
                <p>Outliers and invalid values:</p>
                <pre>{json.dumps(summary['statistical_issues'], indent=2)}</pre>
            </div>

            <div class="section">
                <h2>Correlations</h2>
                <p>High correlations (>{summary['correlations']['threshold']}):</p>
                <pre>{json.dumps(summary['correlations']['high_correlations'], indent=2)}</pre>
                <img src="{summary['correlations']['correlation_matrix']}" alt="Correlation Heatmap">
            </div>

            <div class="section">
                <h2>Recommended Actions</h2>
                <ul>
                    {"".join(f"<li>{action}</li>" for action in summary['recommended_actions'])}
                </ul>
            </div>
        </body>
    </html>
    """

    with open(report_file, "w") as f:
        f.write(html_content)
