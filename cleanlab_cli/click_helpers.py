"""
Standardized methods for Click terminal logs

Error / failure: red
Warning: orange
Progress status: blue
Information: default
Success/completion: green
"""
import click
import os
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


def prompt_for_filepath(prompt_message, default=None):
    filepath = click.prompt(prompt_message, default=default)
    while not os.path.exists(filepath):
        error(f"No file exists at: {filepath}")
        filepath = click.prompt(prompt_message)
    return filepath


def prompt_for_save_filepath(prompt_message, default=None):
    filepath = click.prompt(prompt_message, default=default)
    return filepath


def confirm_open_file(message, filepath):
    edit = click.confirm(message, default=None)
    if edit:
        try:
            # TODO expand list of recommended editors
            click.edit(filename=filepath, editor="atom")
        except click.UsageError:
            click.edit(filename=filepath)


def confirm_save_data(message, default=None):
    save = click.confirm(message, default=default)
    return save


def confirm_save_prompt_filepath(
    save_message, save_default, prompt_message, prompt_default, no_save_message
):
    save = confirm_save_data(save_message, default=save_default)
    if save:
        output = prompt_for_save_filepath(prompt_message, default=prompt_default)
    else:
        info(no_save_message)
        return
    return output
