import click
from omegaconf import OmegaConf
from scabha.schema_utils import clickify_parameters

import simms
from simms.parser_config.utils import load, load_sources

command = "observe"
sources = load_sources(["skysim", "telescope"])
config = load(command, use_sources=sources)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)


"""import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

import simms
from simms.parser_config.utils import load

command = "observe"
config = load(command)

thisdir = os.path.dirname(__file__)
observe_params = glob.glob(f"{thisdir}/library/*.yaml")
observe_files = [File(item) for item in observe_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, observe_files)[command]


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)"""
