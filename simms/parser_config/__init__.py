import click
from scabha.basetypes import File
import glob
import os

@click.group()
def cli():
    pass

def add_commands():
    from .telsim import telsim_runit
    from .skysim import skysim_runit
    
add_commands()