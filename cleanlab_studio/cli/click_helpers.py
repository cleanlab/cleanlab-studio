"""
Standardized methods for Click terminal logs

Error / failure: red
Warning: orange
Progress status: blue
Information: default
Success/completion: green
"""

import os

import click
from click import ClickException, secho, style, echo
from typing import Optional, Any


def abort(message: str) -> None:
    """
    Abort on error
    """
    raise ClickException(style(message, fg="red"))


def error(message: str) -> None:
    secho(style(message, fg="red"))


def success(message: str) -> None:
    secho(style(message, fg="green"))


def progress(message: str) -> None:
    secho(style(message, fg="yellow"))


def info(message: str) -> None:
    secho(style(message, fg="blue"))


def warn(message: str) -> None:
    secho(style(message, fg="magenta"))


def log(message: str) -> None:
    echo(message)


def prompt_for_filepath(prompt_message: str, default: Optional[Any] = None) -> str:
    filepath = str(click.prompt(prompt_message, default=default))
    while not os.path.exists(filepath):
        error(f"No file exists at: {filepath}")
        filepath = str(click.prompt(prompt_message))
    return filepath


def confirm_open_file(message: str, filepath: str) -> None:
    edit = click.confirm(message, default=None)
    if edit:
        click.edit(filename=filepath)


def confirm_save_data(message: str, default: Optional[Any] = None) -> bool:
    save = click.confirm(message, default=default)
    return save


def confirm_save_prompt_filepath(
    save_message: str,
    save_default: Optional[str],
    prompt_message: str,
    prompt_default: str,
    no_save_message: str,
) -> Optional[str]:
    save = confirm_save_data(save_message, default=save_default)
    if save:
        output: str = click.prompt(prompt_message, default=prompt_default)
        return output
    else:
        info(no_save_message)
        return None
