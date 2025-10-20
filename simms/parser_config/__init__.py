import glob
import os

import click
from scabha.basetypes import File


@click.group()
def cli():
    pass


def add_commands():
    from .skysim import skysim_runit
    from .telsim import telsim_runit


add_commands()
