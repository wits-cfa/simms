import os
from glob import glob
import simms
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from . import cli
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

@cli.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def skysim_runit(**kwargs):
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
    # TODO(Sphe,Senkhosi) This is too taylored to a specific use case and input file schema.
    # We need more generic API for handling multiple FITS images as input.
    # Something like we do in skymods.skymodel_from_fits(stack_axis)
    elif fs:
        if os.path.exists(fs):
            if os.path.isdir(fs):
                fs = []
                try:
                    fs.append(glob(f"{fs}/*I.fits")[0]) # the wildcard pattern must be a user input
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
        predict = skymodel_from_fits(
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

        # Build blockwise arguments conditionally
        blockwise_args = [
            augmented_im_to_vis, ("row", "chan", "corr"),
            predict.image, ("npix", "chan", "corr") if predict.use_dft else ("l", "m", "chan", "corr"),
            msds.UVW.data, ("row", "uvw"),
        ]

        # Only add lm for DFT mode
        if predict.use_dft:
            blockwise_args.extend([predict.lm, ("npix", "lm")])
        else:
            blockwise_args.extend([None, None])

        blockwise_args.extend([freqs, ("chan",)])

        simvis = da.blockwise(
            *blockwise_args,
            polarisation = predict.is_polarised,
            expand_freq_dim = predict.expand_freq_dim,
            use_dft = predict.use_dft,
            ncorr = ncorr,
            delta_ra = predict.ra_pixel_size,
            delta_dec = predict.dec_pixel_size,
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
