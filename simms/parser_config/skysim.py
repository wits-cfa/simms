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
from casacore.tables import table
from tqdm import tqdm

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
    else:
        map_path = f'{thisdir}/library/{opts.cat_species}.yaml'

    mapdata = OmegaConf.load(map_path)
    mapcols = OmegaConf.create({})
    column = opts.column
    delimiter = opts.cat_delim

    for key in mapdata:
        keymap = mapdata.get(key)
        if keymap:   
            mapcols[key] = (keymap.name, [], keymap.get("unit"))
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
                
    tab = table(ms, readonly=False, ack=False) 
    data = tab.getcol(column)
    uvw = tab.getcol("UVW")
    fldtab = table(f"{ms}::FIELD", ack=False) 
    radec0 = fldtab.getcol("PHASE_DIR")
    ra0 = radec0[0,0][0] 
    dec0 = radec0[0,0][1]
    nrow = tab.nrows()
    spw_tab = table(f"{ms}::SPECTRAL_WINDOW", ack=False)
    freqs = spw_tab.getcol("CHAN_FREQ")[0]
    nrows, nchan, ncorr = data.shape
    
    if ncorr == 2:
        xx,yy = 0,1
    elif ncorr == 4:
        xx,yy = 0,4
    else:
        raise RuntimeError("The input MS must have 2 or 4 correlations")

    sources = makesources(mapcols,freqs, ra0, dec0) 

    with tqdm(total=nrow, desc='computing visibilities', unit='rows') as pbar:
        print(f'computing visibilities for {nrows} rows ')
        vischan = np.zeros_like(data)
        for row in range(nrows):
            # adding the visibilites to the first diagonal
            vischan[row,:,xx] = computevis(sources, uvw[row], nchan, freqs)  
            # adding the visibilities to the fourth diagonal
            vischan[row,:,yy] = vischan[row,:,0] 
            pbar.update(1)

    nrow = tab.nrows()
    with tqdm(total=nrow, desc='simulating', unit='rows') as pbar2:
        print("Starting to add rows to ms file")
        for i in range(nrow):
            if opts.mode == "add":
                datai = tab.getcell(column, i) + vischan[i]
            elif opts.mode == "subtract":
                datai = tab.getcell(column, i) + vischan[i]
            else:
                datai = vischan[i]

            tab.putcell(column, i, datai)
            pbar2.update(1)
        tab.close()
