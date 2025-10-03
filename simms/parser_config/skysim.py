import os
from glob import glob
import simms
import numpy as np
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
        
        simvis = da.blockwise(
            augmented_im_to_vis, ("row", "chan", "corr"),
            predict.image, ("npix", "chan", "corr") if predict.use_dft else ("l", "m", "chan", "corr"),
            msds.UVW.data, ("row", "uvw"),
            predict.lm, ("npix", "lm") if predict.use_dft else None,
            freqs, ("chan",),
            polarisation = predict.is_polarisation,
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
        
    if opts.transient:
        
        unique_times, inv_idx = np.unique(msds.TIME.values, return_inverse=True)
        t0 = unique_times.min()
        unique_times_rel = unique_times - t0

        transient_profile = exoplanet_transient_logistic(
            start_time=unique_times_rel.min(),
            end_time=unique_times_rel.max(),
            ntimes=unique_times_rel.shape[0],
            transient_start=opts.transient[0],  # now relative to obs start
            transient_absorb=opts.transient[1],
            transient_ingress=opts.transient[2],
            transient_period=opts.transient[3],
            baseline=1.0
        )

        if transient_profile.shape[0] != unique_times.shape[0]:
            raise ValueError("Transient profile length does not match number of unique times.")
        
        modulation = transient_profile[inv_idx]
    
        outvis = outvis * modulation[:, np.newaxis,np.newaxis]
    
    print(f"simvis:{simvis.shape}")
    ms_dsl = [                      
    msds.assign(**{
                opts.column: (("row", "chan", "corr"), outvis),
            },
        )]
    
    writes = xds_to_table(ms_dsl, ms, columns=[opts.column])

    with TqdmCallback(desc="Computing and writing visibilities"):
        da.compute(writes)


def exoplanet_transient_logistic(
    start_time, end_time, ntimes,
    transient_start,
    transient_absorb,
    transient_ingress,
    transient_period,
    baseline
):

    times = np.linspace(start_time, end_time, ntimes)

    def logistic_step(z, L=10.0):
        "Logistic function mapped to [0, 1] using internal steepness scaling L."
        z = np.clip(z, 0, 1)
        k = L  # steepness across [0, 1]
        raw = 1 / (1 + np.exp(-k * (z - 0.5)))
        f0 = 1 / (1 + np.exp(k / 2))
        f1 = 1 / (1 + np.exp(-k / 2))
        normalized = (raw - f0) / (f1 - f0)
        return normalized

    intensity = np.full_like(times, baseline, dtype=np.float64)

    ingress_start = transient_start
    ingress_end = ingress_start + transient_ingress

    egress_end = transient_start + transient_period
    egress_start = egress_end - transient_ingress

    plateau_start = ingress_end
    plateau_end = egress_start

    # Ingress
    mask_ingress = (times >= ingress_start) & (times < ingress_end)
    z_ingress = (times[mask_ingress] - ingress_start) / transient_ingress
    intensity[mask_ingress] = baseline - transient_absorb * logistic_step(z_ingress, L=10)

    # Flat bottom
    mask_plateau = (times >= plateau_start) & (times < plateau_end)
    intensity[mask_plateau] = baseline - transient_absorb

    # Egress
    mask_egress = (times >= egress_start) & (times < egress_end)
    z_egress = (times[mask_egress] - egress_start) / transient_ingress
    intensity[mask_egress] = baseline - transient_absorb * (1 - logistic_step(z_egress, L=10))

    
    return intensity
