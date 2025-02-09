"""Command-line interface for DataRamp project management."""

from datetime import datetime
from pathlib import Path
from typing import List

import click
import joblib
import pandas as pd

from dataramp.core import create_project as core_create_project
from dataramp.core import data_save, model_save, update_dependencies


@click.group()
def cli():
    """DataRamp CLI for data science project management."""
    pass


@cli.command("create")
@click.argument("project_name")
@click.option("--description", help="Project description for README")
@click.option("--python-version", default="3.9", help="Base Python version")
@click.option("--with-git/--no-git", default=True, help="Initialize Git repository")
@click.option("--add-dir", multiple=True, help="Additional directories to create")
@click.option("--package", multiple=True, help="Additional Python packages")
def create_project(
    project_name: str,
    description: str,
    python_version: str,
    with_git: bool,
    add_dir: List[str],
    package: List[str],
):
    """Create a new DataRamp project structure."""
    try:
        core_create_project(
            project_name=project_name,
            description=description or f"{project_name} Data Science Project",
            python_version=python_version,
            extra_dirs=list(add_dir),
            packages=list(package),
            init_git=with_git,
        )

        project_root = Path(project_name)
        click.echo(f"\nCreated project structure at {project_root.resolve()}")
        click.echo("├── .dataramprc (configuration)")
        click.echo("├── requirements.txt (dependencies)")
        click.echo("├── environment.yml (conda environment)")
        click.echo("├── datasets/")
        click.echo("│   ├── raw/")
        click.echo("│   └── processed/")
        click.echo("├── outputs/")
        click.echo("│   └── models/")
        click.echo("└── src/")
        click.echo("    ├── notebooks/")
        click.echo("    └── scripts/")

        if with_git:
            click.echo("\nInitialized Git repository")

    except Exception as e:
        click.secho(f"Error creating project: {str(e)}", fg="red")
        raise click.Abort()


@cli.command("save-data")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--name", default="dataset", help="Name for the dataset")
@click.option(
    "--format",
    default="parquet",
    type=click.Choice(["parquet", "csv", "feather"]),
    help="File format for saving",
)
@click.option("--versioned/--no-versioned", default=True, help="Enable data versioning")
def save_data(input_file: str, name: str, format: str, versioned: bool):
    """Save a dataset with optional versioning."""
    try:
        df = (
            pd.read_csv(input_file)
            if input_file.endswith(".csv")
            else pd.read_parquet(input_file)
        )

        result = data_save(
            df, name=name, method=format, versioning=versioned, compression="snappy"
        )

        if versioned:
            click.echo(f"Created data version: {result.version_id}")
            click.echo(f"Data hash: {result.data_hash}")
        else:
            click.echo(f"Saved dataset to: {result}")

    except Exception as e:
        click.secho(f"Data save failed: {str(e)}", fg="red")


@cli.command("models")
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--name", required=True, help="Model name")
@click.option("--version", help="Model version (default: autogenerate)")
@click.option(
    "--format",
    default="joblib",
    type=click.Choice(["joblib", "pickle"]),
    help="Serialization format",
)
def save_model(model_file: str, name: str, version: str, format: str):
    """Save a trained ML model with versioning."""
    try:
        model = joblib.load(model_file)
        model_path = model_save(
            model,
            name=name,
            method=format,
            version=version,
            metadata={
                "training_date": datetime.now().isoformat(),
                "source_file": model_file,
            },
        )
        click.echo(f"Saved model version to: {model_path}")
    except Exception as e:
        click.secho(f"Model save failed: {str(e)}", fg="red")


@cli.command("deps")
@click.option(
    "--update/--freeze",
    default=True,
    help="Update dependencies or freeze current versions",
)
def manage_dependencies(update: bool):
    """Manage project dependencies."""
    try:
        if update:
            update_dependencies()
            click.echo("Updated dependency versions in requirements.txt")
        else:
            click.echo("Frozen current dependency versions")
    except Exception as e:
        click.secho(f"Dependency management failed: {str(e)}", fg="red")


if __name__ == "__main__":
    cli()
