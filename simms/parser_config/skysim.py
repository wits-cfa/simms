import simms
from simms.parser_config.utils import load, load_sources
from simms.skymodel import thisdir as skysimdir
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from omegaconf import OmegaConf
import os

command = "skysim"
sources = load_sources(["library/sys_noise"])
config = load(command, use_sources=sources)


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

        
