import os
from glob import glob
import simms
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
import glob
from simms.utilities import ParameterError
from simms.skymodel.skymods import (
    skymodel_from_catalogue,
    skymodel_from_fits,
)

from simms.skymodel.mstools import (
    compute_vis,
    augmented_im_to_vis,
    vis_noise_from_sefd_and_ms,
    sim_noise,
)
from tqdm.dask import TqdmCallback
import dask.array as da
from daskms import xds_to_table, xds_from_ms, xds_from_table

log = get_logger(BIN.skysim, level="ERROR")

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
    
    if cat and fs:
        raise ParameterError("Cannot use both a catalogue and a FITS sky model simultaneously")
    
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"],
                taql_where=f"FIELD_ID=={opts.field_id}",
                chunks=dict(row=opts.row_chunks),
                )[opts.spw_id]
    
    if opts.sefd:
        vis_noise = vis_noise_from_sefd_and_ms(ms, opts.sefd, opts.spw_id, opts.field_id)
    else:
        vis_noise = 0
    
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    
    radec0 = field_ds.PHASE_DIR.data[opts.field_id].compute()
    ra0, dec0 = radec0[0][0], radec0[0][1]
    freqs = spw_ds.CHAN_FREQ.data[opts.spw_id].compute()
    dfreq = spw_ds.CHAN_WIDTH.data[opts.spw_id][0].compute()
    ncorr = msds.DATA.data.shape[-1]
    
    if cat:
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

        sources = skymodel_from_catalogue(cat, map_path=map_path, delimiter=opts.cat_delim,
                                    chan_freqs=freqs, full_stokes=opts.polarisation)
        
## These checks should be done in skymods/source factory 
#       # check for polarisation information
#       if any(catsky[col][1] for col in ['stokes_q', 'stokes_u', 'stokes_v']):
#           polarisation = True
#       else:
#           polarisation = False
#       # TODO: Consider adding condition that all elements are non-zero and not "null" or None
#       
#       # warn user if polarisation is detected but only two correlations are requested    
#       if polarisation and ncorr == 2:
#           log.warning("Q, U and/or V detected but only two correlations requested. U and V will be absent from the output MS.")

        
        simvis = da.blockwise(
            compute_vis, ("row", "chan", "corr"),
            sources, ("source",),
            msds.UVW.data, ("row", "uvw"),
            freqs, ("chan",),
            ncorr=ncorr,
            polarisation=opts.polarisation,
            pol_basis=opts.pol_basis,
            noise_vis=vis_noise,
            ra0=ra0,
            dec0=dec0,
            new_axes={"corr": ncorr},
            dtype=msds.DATA.data.dtype,
            concatenate=True,
        )
        
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
        
        # process FITS sky model
        image, lm, polarisation, use_dft, delta_ra, delta_dec = skymodel_from_fits(
            fs, 
            ra0, 
            dec0, 
            freqs, 
            dfreq, 
            ncorr, 
            opts.pol_basis, 
            tol=float(opts.pixel_tol)
        )
        
        epsilon = 1e-7 if opts.fft_precision == "double" else 1e-5
        
        simvis = da.blockwise(
            augmented_im_to_vis, ("row", "chan", "corr"),
            image, ("npix", "chan", "corr") if use_dft else ("l", "m", "chan", "corr"),
            msds.UVW.data, ("row", "uvw"),
            lm, ("npix", "lm") if use_dft else None,
            freqs, ("chan",),
            polarisation = polarisation,
            use_dft = use_dft,
            ncorr = ncorr,
            delta_ra = delta_ra,
            delta_dec = delta_dec,
            epsilon = epsilon,
            do_wstacking = opts.do_wstacking,
            noise = vis_noise,
            dtype = msds.DATA.data.dtype,
            concatenate=True,
        )
        
    elif opts.sefd:
        # Simulate noise if no skymodel is given
        simvis = da.blockwise(
            sim_noise, ("nrow", "nchan", "ncorr"),
            dshape=msds.DATA.data.shape,
            noise = vis_noise,
            dtype = msds.DATA.data.dtype,
            concatenate=True,
        )
    
    if opts.input_column:
        if not hasattr(msds, opts.input_column):
            raise RuntimeError(f"Specified input-column '{opts.input_column}' does not exist in the MS")
        incol = getattr(msds, opts.input_column).data 
        if opts.mode == "add":
            outvis = incol + simvis
        elif opts.mode == "subtract":
            outvis = incol - simvis
        elif opts.mode == "sim":
            outvis = simvis
    else:
        outvis = simvis 
    
    ms_dsl = [                      
    msds.assign(**{
                opts.column: (("row", "chan", "corr"), outvis),
            },
        )]
    
    writes = xds_to_table(ms_dsl, ms, columns=[opts.column])

    with TqdmCallback(desc="Computing and writing visibilities"):
        da.compute(writes)
