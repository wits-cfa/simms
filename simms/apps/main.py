import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

from simms import BIN, __version__
from simms.apps import skysim, telsim

thisdir = os.path.dirname(__file__)

telsim_parserfile = File(f"{thisdir}/{BIN.telsim}.yaml")
skysim_parserfile = File(f"{thisdir}/{BIN.skysim}.yaml")

telsim_config = paramfile_loader(telsim_parserfile, sources=[])[BIN.telsim]
skysim_config = paramfile_loader(skysim_parserfile, sources=[])[BIN.skysim]


class RemoveMSIfChained(click.Command):
    """
    A custom command class that removes an option based on a parent flag.
    """
    def make_parser(self, ctx):
        """
        This method is called before parsing. It checks the parent context
        for a '--chain' flag and removes the 'ms' argument if it's found.
        """
        # Check if the parent context and its params exist
        if ctx.parent and ctx.parent.params.get('chain', False):
            # The 'params' attribute is a list of all options/arguments.
            # We rebuild the list, excluding the one named 'ms'.
            self.params = [p for p in self.params if p.name != 'ms']
        
        # IMPORTANT: Call the superclass method to continue the parsing process
        return super().make_parser(ctx)


@click.group(chain=True, no_args_is_help=True)
@click.version_option(str(__version__))
@click.option("--ms", "-ms")
@click.option(
    "--log-level", "-ll", help="Log level", type=click.Choice(["INFO", "WARNING", "CRITICAL", "ERROR"]), default="INFO"
)
@click.option("--chain", is_flag=True, help="Chain telsim and skysim for an End-to-End simulation.")
@click.pass_context
def cli(ctx, ms, log_level, chain):
    
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["chain"] = chain
    if chain:
        ctx.obj["ms"] = ms
        del skysim_config.inputs.ms
        del telsim_config.inputs.ms


@cli.command(BIN.telsim, no_args_is_help=True, cls=RemoveMSIfChained)
@click.option(
    "--list",
    "-ls",
    is_flag=True,
    callback=telsim.print_data_database,
    expose_value=False,
    help="Displays a message and then exits the program.",
)
@clickify_parameters(telsim_config)
@click.pass_context
def run_telsim(ctx, **kwargs):
    opts = OmegaConf.create(kwargs)
    opts["log_level"] = ctx.obj["log_level"]
    if ctx.obj["chain"]:
        opts["ms"] = ctx.obj["ms"]
    
    telsim.runit(opts)


@cli.command(BIN.skysim, no_args_is_help=True, cls=RemoveMSIfChained)
@clickify_parameters(skysim_config)
@click.pass_context
def run_skysim(ctx, **kwargs):
    opts = OmegaConf.create(kwargs)
    opts["log_level"] = ctx.obj["log_level"]
    if ctx.obj["chain"]:
        opts["ms"] = ctx.obj["ms"]
    skysim.runit(opts)
