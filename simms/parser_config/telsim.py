import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

import simms
from simms import BIN, get_logger
from simms.telescope import generate_ms

log = get_logger(BIN.telsim)

command = BIN.telsim

thisdir = os.path.dirname(__file__)
telescope_params = glob.glob(f"{thisdir}/library/*.yaml")
telescope_files = [File(item) for item in telescope_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, telescope_files)[command]


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    msname = opts.ms
    telescope = opts.telescope
    direction = opts.direction.split(",")
    starttime = opts.starttime
    dtime = opts.dtime
    ntimes = opts.ntimes
    startfreq = opts.startfreq
    dfreq = opts.dfreq
    nchan = opts.nchan
    correlations = opts.correlations
    rowchunks = opts.rowchunks
    addnoise = opts.addnoise
    sefd = opts.sefd
    column = opts.column
    generate_ms.create_ms(msname, telescope, direction, dtime,
                          ntimes, startfreq, dfreq, nchan,
                          correlations, rowchunks, addnoise,
                          sefd, column, starttime)
