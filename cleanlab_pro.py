import click


class AuthConfig:
    def __init__(self):
        self.username = None
        self.password = None
        self.bearer = None

    def status(self):
        if self.bearer:
            click.echo("Currently logged in!")
        else:
            click.echo("Not logged in.")


auth_config = click.make_pass_decorator(AuthConfig, ensure=True)


@click.group()
@auth_config
def main(config):
    pass


@main.command()
@auth_config
def login(config):
    click.echo("I am logged in!")


@main.group()
@auth_config
def dataset(config):
    pass

@dataset.command()
@click.option('--file', type=click.Path(), help='Filepath to dataset', required=True)
@click.option('--modality', type=str, help="Data modality: text, tabular, or image")
@click.option('--name', type=str, help='Name of dataset')
@click.option('--id', type=str, help="If resuming upload or appending to an existing dataset, specify the dataset ID")
@auth_config
def upload(config, file, modality, name, id):
    click.echo(config.status())
    click.echo(f"Uploading {file} with {modality} modality named {name} of ID {id}")

