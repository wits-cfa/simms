import glob
import os.path

import dask.array as da
import numpy as np
from dask import config as dask_config
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm.dask import TqdmCallback

from simms import BIN, set_logger
from simms.skymodel.mstools import (
    augmented_im_to_vis,
    compute_vis,
    sim_noise,
    vis_noise_from_sefd_and_ms,
)
from simms.skymodel.skymods import (
    load_sources,
    skymodel_from_fits,
    skymodel_from_sources,
)
from simms.utilities import ParameterError

thisdir = os.path.abspath(os.path.dirname(__file__))


def runit(opts):
    log = set_logger(BIN.skysim, opts["log_level"])
    ms = opts.ms
    cat = opts.ascii_sky
    fs = opts.fits_sky

    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    if cat and fs:
        raise ParameterError("Cannot use both a catalogue and a FITS sky model simultaneously")

    msds = xds_from_ms(
        ms,
        group_cols=["DATA_DESC_ID"],
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
            log.warning(
                f"No mapping file specified nor built-in map selected. Assuming default column names (see {map_path})"
            )

        sources = load_sources(cat, map_path=map_path, delimiter=opts.cat_delim)

        if any([src.is_transient for src in sources]):
            unique_times = np.unique(msds.TIME.values)
            skymodel = skymodel_from_sources(
                sources, chan_freqs=freqs, full_stokes=opts.polarisation, unique_times=unique_times
            )
            simvis = da.blockwise(
                compute_vis,
                ("row", "chan", "corr"),
                skymodel,
                ("source",),
                msds.UVW.data,
                ("row", "uvw"),
                freqs,
                ("chan",),
                msds.TIME.data,
                ("row",),
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
        else:
            skymodel = skymodel_from_sources(sources, chan_freqs=freqs, full_stokes=opts.polarisation)

            simvis = da.blockwise(
                compute_vis,
                ("row", "chan", "corr"),
                skymodel,
                ("source",),
                msds.UVW.data,
                ("row", "uvw"),
                freqs,
                ("chan",),
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
                    fs.append(glob(f"{fs}/*I.fits")[0])  # the wildcard pattern must be a user input
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
            fs, ra0, dec0, freqs, dfreq, ncorr, opts.pol_basis, tol=opts.pixel_tol, interpolation=opts.fits_sky_interp
        )

        dtype = np.finfo(predict.image.dtype).dtype
        if dtype == np.float32:
            epsilon = 1e-6
        else:
            epsilon = 1e-7 if opts.fft_precision == "double" else 1e-6

        simvis = da.blockwise(
            augmented_im_to_vis,
            ("row", "chan", "corr"),
            predict.image,
            ("npix", "chan", "corr") if predict.use_dft else ("l", "m", "chan", "corr"),
            msds.UVW.data,
            ("row", "uvw"),
            predict.lm,
            ("npix", "lm") if predict.use_dft else None,
            freqs,
            ("chan",),
            polarisation=predict.is_polarised,
            expand_freq_dim=predict.expand_freq_dim,
            use_dft=predict.use_dft,
            ncorr=ncorr,
            delta_ra=predict.ra_pixel_size,
            delta_dec=predict.dec_pixel_size,
            epsilon=epsilon,
            do_wstacking=opts.do_wstacking,
            noise=vis_noise,
            dtype=msds.DATA.data.dtype,
            concatenate=True,
        )

    elif opts.sefd:
        # Simulate noise if no skymodel is given
        simvis = da.blockwise(
            sim_noise,
            ("nrow", "nchan", "ncorr"),
            dshape=msds.DATA.data.shape,
            noise=vis_noise,
            dtype=msds.DATA.data.dtype,
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
        msds.assign(
            **{
                opts.column: (("row", "chan", "corr"), outvis),
            },
        )
    ]

    writes = xds_to_table(ms_dsl, ms, columns=[opts.column])

    with TqdmCallback(desc="Computing and writing visibilities"):
        da.compute(writes)
