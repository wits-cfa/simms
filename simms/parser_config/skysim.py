import os
from glob import glob
import simms
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
import glob
from simms.utilities import CatalogueError, isnummber, ParameterError
from simms.skymodel.skymods import (
    read_ms, 
    makesources, 
    computevis,
    process_fits_skymodel,
    augmented_im_to_vis,
    add_to_column,
    add_noise
)
import numpy as np
from tqdm.dask import TqdmCallback
import dask.array as da
from daskms import xds_to_table

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
    ms = opts.ms
    cat = opts.catalogue
    fs = opts.fits_sky
    chunks = {"row": opts.row_chunk_size}
    
    if cat and fs:
        raise ParameterError("Cannot use both a catalogue and a FITS sky model simultaneously")
    elif cat:
        map_path = opts.mapping

        if opts.mapping and opts.cat_species:
            raise ParameterError("Cannot use custom map and built-in map simultaneously")
        elif opts.mapping:
            map_path = opts.mapping
        elif opts.cat_species:
            map_path = f"{thisdir}/library/{opts.cat_species}.yaml"
        else:
            map_path = f"{thisdir}/library/catalogue_template.yaml"
            log.warning(f"No mapping file specified nor built-in map selected. Assuming default column names (see {map_path})")

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
        
        # validate mapcols to ensure that the required columns were read successfully
        for col in ["name", "ra", "dec", "stokes_i"]:
            if not mapcols[col][1]: # if the list storing column's data is empty
                if col == "name": # user might not understand what simply "name" means
                    raise CatalogueError(
                        f"Failed to identify required column corresponding to source name/ID in the catalogue."
                        " Please ensure that catalogue column headers match those in mapping file."
                    )
                else:
                    raise CatalogueError(
                    f"Failed to identify required column corresponding to '{col}' in the catalogue."
                    " Please ensure that catalogue column headers match those in mapping file."
                )      
        
        # read MS (also computes noise)
        ms_dsl, ra0, dec0, freqs, nrow, nchan, _, ncorr, noise, incol, incol_dims = read_ms(ms, 
                                                                                         opts.spwid, 
                                                                                         opts.field_id, 
                                                                                         chunks, 
                                                                                         sefd = opts.sefd, 
                                                                                         input_column = opts.input_column
                                                                                        )
        
        sources = makesources(mapcols, freqs, ra0, dec0)
        
        allvis = []

        # check for polarisation information
        if any(mapcols[col][1] for col in ['stokes_q', 'stokes_u', 'stokes_v']):
            polarisation = True
        else:
            polarisation = False
        # TODO: Consider adding condition that all elements are non-zero and not "null" or None
        
        # warn user if polarisation is detected but only two correlations are requested    
        if polarisation and ncorr == 2:
            log.warning("Q, U and/or V detected but only two correlations requested. U and V will be absent from the output MS.")

        for ds in ms_dsl:
            simvis = da.blockwise(computevis, ("row", "chan", "corr"),
                                sources, ("source",),
                                ds.UVW.data, ("row", "uvw"),
                                freqs, ("chan",),
                                ncorr, None,
                                polarisation, None,
                                new_axes={"corr": ncorr},
                                dtype=ds.DATA.data.dtype,
                                concatenate=True,
                                )
            
            allvis.append(simvis)
        
    elif fs:
        if os.path.exists(fs):
            if os.path.isdir(fs):
                fs = []
                try:
                    fs.append(glob(f"{fs}/*I.fits")[0])
                    fs.append(glob(f"{fs}/*Q.fits")[0])
                    fs.append(glob(f"{fs}/*U.fits")[0])
                    fs.append(glob(f"{fs}/*V.fits")[0])
                except IndexError:
                    raise ParameterError("Could not find all required FITS files in the specified directory")
                
                if len(fs) > 4:
                    raise ParameterError("Too many FITS files found in the specified directory")
                
            elif not fs.endswith(".fits"):
                raise ParameterError("Invalid FITS file specified")
        else:
            raise ParameterError("FITS file/directory does not exist")
        
        # read MS (also computes noise)
        ms_dsl, ra0, dec0, freqs, nrow, nchan, df, ncorr, noise, incol, incol_dims = read_ms(ms, 
                                                                                         opts.spwid, 
                                                                                         opts.field_id, 
                                                                                         chunks, 
                                                                                         sefd = opts.sefd, 
                                                                                         input_column = opts.input_column
                                                                                        )
        
        # process FITS sky model
        image, lm, sparsity, n_pix_l, n_pix_m, delta_l, delta_m = process_fits_skymodel(fs, ra0, dec0, freqs, df, ncorr, opts.pol_basis, tol=float(opts.pixel_tol))
        
        allvis = []
    
        for ds in ms_dsl:
            simvis = da.blockwise(
                augmented_im_to_vis, ("row", "chan", "corr"),
                image, ("npix", "chan", "corr"),
                ds.UVW.data, ("row", "uvw"),
                lm, ("npix", "lm"),
                freqs, ("chan",), 
                sparsity, None,
                n_pix_l = n_pix_l,
                n_pix_m = n_pix_m, 
                delta_l = delta_l,
                delta_m = delta_m,
                tol=float(opts.pixel_tol),
                dtype=ds.DATA.data.dtype,
                concatenate=True
            )
            
            allvis.append(simvis)
        
      
    else:
        raise ParameterError("No sky model specified. Please provide either a catalogue or a FITS sky model.")

    if opts.noise or opts.mode:
        with TqdmCallback(desc="compute visibilities"):
            allvis = da.compute(*allvis)
            allvis = np.concatenate(allvis, axis=0)

        if opts.noise:
            allvis = add_noise(allvis, opts.noise)
        
        if opts.mode:
            allvis = add_to_column(allvis, incol, opts.mode)

    writes = []
    for i, ds in enumerate(ms_dsl):
        ms_dsl[i] = ds.assign(**{
                opts.column: ( ("row", "chan", "corr"), 
                    allvis[i]),
        })
    
        writes.append(xds_to_table(ms_dsl, ms, [opts.column]))
        
    with TqdmCallback(desc="compute"):
        da.compute(writes)
