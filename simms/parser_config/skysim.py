import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

import simms
from simms import BIN, get_logger
from simms.parser_config.utils import load, load_sources
from simms.skymodel import catalogue
from simms.skymodel import thisdir as skysimdir

log = get_logger(BIN.skysim)

command = BIN.skysim
sources = load_sources(["library/sys_noise"])
thisdir = os.path.dirname(__file__)
config = load(command, use_sources=sources)

source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]


skyspec = paramfile_loader(os.path.join(skysimdir, "skyspec.yaml"))
print(skyspec)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    cat = opts.catalogue
    ms = opts.ms
    map_path = opts.mapping
    mapdata = OmegaConf.load(map_path)
    catcols = []
    cattypes = []
    for key in mapdata:
        value = mapdata.get(key)
        if value != None:
            key = value
        catcols.append(key)
