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


@click.command()
def login(config):
    click.echo("I am logged in!")
