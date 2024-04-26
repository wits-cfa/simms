import os
import simms
from simms.parser_config.utils import load, load_sources
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
from simms.skymodel import catalogue

log = get_logger(BIN.skysim)

command = BIN.skysim
sources = load_sources(["library/sys_noise"])
thisdir  = os.path.dirname(__file__)
config = load(command, use_sources=sources)

source_files = glob.glob(f"{thisdir}/library/*.yaml") 
sources = [File(item) for item in source_files] 
parserfile = File(f"{thisdir}/{command}.txt") 
config = paramfile_loader(parserfile, sources)[command] 


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)

    inpms = File(opts.ms)
    sourcecat = File(opts.source_catalogue)
    sourcecat = catalogue(sourcecat)
    die = File(opts.die)
    dde = File(opts.dde) 
    if opts.die.EXISTS:
        die = File(opts.die)
    sourcetype = opts.source_type  
    spectype = opts.spectrum
    
runit()