import click

from simms import __version__

from . import skysim, telsim


@click.group()
@click.version_option(str(__version__))
@click.option(
    "--log-level", "-ll", help="Log level", type=click.Choice(["INFO", "WARNING", "CRITICAL", "ERROR"]), default="INFO"
)
@click.pass_context
def cli(ctx, log_level):
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


cli.add_command(skysim.runit)
cli.add_command(telsim.runit)
