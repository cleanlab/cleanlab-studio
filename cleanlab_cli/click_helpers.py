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


def warn(message):
    """
    An issue was found
    """
    secho(style(message, fg="orange"))


def error(message):
    secho(style(message, fg="red"))


def success(message):
    secho(style(message, fg="green"))


def info(message):
    secho(style(message, fg="yellow"))


def log(message):
    echo(message)


def progress(message):
    secho(style(message, fg="blue"))
