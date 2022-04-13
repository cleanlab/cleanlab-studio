"""
Standardized methods for Click terminal logs

Error / failure: red
Warning: orange
Progress status: blue
Information: default
Success/completion: green
"""
import click
from click import ClickException, secho, style, echo


def abort(message):
    """
    Abort on error
    """
    raise ClickException(style(message, fg="red"))


def error(message):
    secho(style(message, fg="red"))


def success(message):
    secho(style(message, fg="green"))


def progress(message):
    secho(style(message, fg="yellow"))


def info(message):
    secho(style(message, fg="blue"))


def log(message):
    echo(message)
