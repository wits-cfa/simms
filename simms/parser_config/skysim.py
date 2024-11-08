import os
import simms
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
import glob
from simms.utilities import CatalogueError, isnummber, ParameterError
from simms.skymodel.skymods import makesources, computevis
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm.dask import TqdmCallback
import dask.array as da

log = get_logger(BIN.skysim)

command = BIN.skysim

thisdir = os.path.dirname(__file__)

source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")

config = paramfile_loader(parserfile, sources)[command]


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    cat = opts.catalogue
    ms = opts.ms
    map_path = opts.mapping

    if opts.mapping and opts.cat_species:
        raise ParameterError("Cannot use custom map and built-in map simultaneously")
    elif opts.mapping:
        map_path = opts.mapping
    elif opts.cat_species:
        map_path = f"{thisdir}/library/{opts.cat_species}.yaml"
    else:
        map_path = f"{thisdir}/library/catalogue_template.yaml"

    mapdata = OmegaConf.load(map_path)
    mapcols = OmegaConf.create({})
    column = opts.column
    delimiter = opts.cat_delim

    for key in mapdata:
        keymap = mapdata.get(key)
        if keymap:   
            mapcols[key] = (keymap.name or key, [], keymap.get("unit"))
        else:
            mapcols[key] = (key, [], None)
            
    with open(cat) as stdr:
        header = stdr.readline().strip()

        if not header.startswith("#format:"):
            raise CatalogueError("Catalogue needs to have a header starting with the string #format:")
        
        header = header.strip().replace("#format:", "").strip()
        header = header.split(delimiter)

        for line in stdr.readlines():
            if line.startswith("#"):
                continue
            rowdata = line.strip().split(delimiter)
            if len(rowdata) != len(header):
                raise CatalogueError("The number of elements in one or more rows does not equal the\
                                    number of expected elements based on the number of elements in the\
                                    header")
            for key in mapcols:
                catkey = mapcols[key][0]
                if catkey not in header:
                    continue
                
                index = header.index(catkey)
                value = rowdata[index]
                if isnummber(value) and mapcols[key][2]:
                    value += mapcols[key][2]
                
                mapcols[key][1].append(value)
                
                
    ms_dsl = xds_from_ms(ms, index_cols=["TIME", "ANTENNA1", "ANTENNA2"], chunks={"row":10000})               
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field_ds  = xds_from_table(f"{ms}::FIELD")[0]
    ms_ds0 = ms_dsl[0]
    
    radec0 = field_ds.PHASE_DIR.data[opts.field_id].compute()
    ra0 = radec0[0][0] 
    dec0 = radec0[0][1]
    nrows, nchan, ncorr = ms_ds0.DATA.data.shape
    freqs = spw_ds.CHAN_FREQ.data[opts.spwid].compute()
    noise = 0
    if opts.sefd:
        df = spw_ds.CHAN_WIDTH.data[opts.spwid][0].compute()
        dt = ms_ds0.EXPOSURE.data[0].compute() 
        noise = opts.sefd / np.sqrt(2*dt*df)
    
    sources = makesources(mapcols,freqs, ra0, dec0)

    
    allvis = []
    
    if isinstance(opts.input_column, str):
        incol = getattr(ms_ds0, opts.input_column).data
        incol_dims = ("row", "chan", "corr")
    else:
        incol = None
        incol_dims = None

    for ds in ms_dsl:
        simvis = da.blockwise(computevis, ("row", "chan", "corr"),
                            sources, ("source",),
                            ds.UVW.data, ("row", "uvw"),
                            freqs, ("chan",),
                            ncorr, None,
                            incol, None,
                            noise, None,
                            opts.mode == "subtract", None,
                            new_axes={"corr": ncorr},
                            dtype=ds.DATA.data.dtype,
                            concatenate=True,
                            )
        
        allvis.append(simvis)
        
    writes = []
    for i, ds in enumerate(ms_dsl):
        ms_dsl[i] = ds.assign(**{
                opts.column: ( ("row", "chan", "corr"), 
                    allvis[i]),
        })
    
        writes.append(xds_to_table(ms_dsl, ms, [opts.column]))
        
    with TqdmCallback(desc="compute"):
        da.compute(writes)
