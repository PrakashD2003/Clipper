"""CLI entry point for StreamClipper (Package 1.1 placeholder)."""

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="streamclipper")
def cli() -> None:
    """StreamClipper — AI-powered stream clipping tool."""


if __name__ == "__main__":
    cli()
