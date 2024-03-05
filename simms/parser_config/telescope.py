import glob
import os

import click
from omegaconf import OmegaConf
from scabha.schema_utils import clickify_parameters, paramfile_loader

import simms
from simms.parser_config.utils import load

# from scabha.basetypes import File
from simms.utilities import File

command = "telescope"
config = load(command)

thisdir = os.path.dirname(__file__)
telescope_files = glob.glob(f"{thisdir}/library/*.yaml")
telefiles = [File(item) for item in telescope_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, telefiles)[command]


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)


runit()
