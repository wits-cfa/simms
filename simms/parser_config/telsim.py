import glob
import os

import click
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.schema_utils import clickify_parameters, paramfile_loader

from simms import BIN, __version__, set_logger
from simms.telescope import generate_ms, layouts

command = BIN.telsim

thisdir = os.path.dirname(__file__)
telescope_params = glob.glob(f"{thisdir}/library/*.yaml")
telescope_files = [File(item) for item in telescope_params]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, telescope_files)[command]


def print_data_database(ctx, param, value):
    """
    Display telescope array database
    """
    if value is False:
        return

    for key, val in layouts.SIMMS_TELESCOPES.items():
        info = getattr(val, "info", " --- ")
        if not getattr(val, "issubarray", False):
            print(f"{key}: {info.strip()}")
            subarrays = getattr(val, "subarray", [])
            if subarrays:
                subarray_string = ", ".join(subarrays)
                print(f"  Subarrays: {subarray_string}")
    raise SystemExit()


@click.command(command)
@click.option(
    "--list",
    "-ls",
    is_flag=True,
    callback=print_data_database,
    expose_value=False,
    help="Displays a message and then exits the program.",
)
@clickify_parameters(config)
@click.pass_context
def runit(ctx, **kwargs):
    opts = OmegaConf.create(kwargs)
    
    log = set_logger(BIB.telsim, ctx.obj["log_level"])

    msname = opts.ms
    telescope = opts.telescope
    direction = opts.direction.split(",")
    starttime = opts.starttime
    dtime = opts.dtime
    ntimes = opts.ntime
    startfreq = opts.startfreq
    dfreq = opts.dfreq
    nchan = opts.nchan
    correlations = opts.correlations.split(",")
    rowchunks = opts.rowchunks
    sefd = opts.sefd
    tsys_over_eta = opts.tsys_over_eta
    column = opts.column
    startha = opts.startha
    l_src_limit = opts.low_source_limit
    h_src_limit = opts.high_source_limit
    freq_range = opts.freq_range
    sfile = opts.sensitivity_file
    if freq_range is not None:
        freq_range = freq_range.split(",")
    subarray_list = opts.subarray_list
    subarray_range = opts.subarray_range
    subarray_file = opts.subarray_file
    smooth = opts.smooth
    fit_order = opts.fit_order

    generate_ms.create_ms(
        ms=msname,
        telescope_name=telescope,
        pointing_direction=direction,
        dtime=dtime,
        ntimes=ntimes,
        start_freq=startfreq,
        dfreq=dfreq,
        nchan=nchan,
        correlations=correlations,
        row_chunks=rowchunks,
        sefd=sefd,
        column=column,
        smooth=smooth,
        fit_order=fit_order,
        start_time=starttime,
        start_ha=startha,
        freq_range=freq_range,
        sfile=sfile,
        tsys_over_eta=tsys_over_eta,
        subarray_list=subarray_list,
        subarray_range=subarray_range,
        subarray_file=subarray_file,
        low_source_limit=l_src_limit,
        high_source_limit=h_src_limit,
    )
