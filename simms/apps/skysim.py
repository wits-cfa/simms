import glob
import os.path

import dask.array as da
import numpy as np
from dask import config as dask_config
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm.dask import TqdmCallback

from simms import BIN, SCHEMADIR, set_logger
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.fits_skies import skymodel_from_fits
from simms.skymodel.mstools import (
    augmented_im_to_vis,
    compute_vis,
    sim_noise,
    vis_noise_from_sefd_and_ms,
)


def runit(opts):
    # Set logger here, so subsequent modeules get it via logging.getLogger(<name>)
    set_logger(BIN.skysim, opts["log_level"])
    ms = opts.ms
    ascii_sky = opts.ascii_sky
    fs = opts.fits_sky

    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    if ascii_sky and fs:
        raise RuntimeError("Cannot use an ASCII and FITS sky model simultaneously")

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
    linear_basis = opts.pol_basis == "linear"

    if ascii_sky:
        if opts.source_schema and opts.ascii_species:
            raise RuntimeError("Cannot use custom map and built-in map simultaneously")
        elif opts.source_schema:
            source_schema = opts.source_schema
        elif opts.ascii_species:
            source_schema = f"{SCHEMADIR}/{opts.ascii_species}.yaml"
        else:
            source_schema = f"{SCHEMADIR}/source_schema.yaml"

        skymodel = ASCIISkymodel(ascii_sky, delimiter=opts.ascii_delimiter, source_schema_file=source_schema)
        times_var = msds.TIME.data, ("row",) if skymodel.has_transient else None, (None,)

        def compute_vis_sky(*args, **kwargs):
            return compute_vis(skymodel, *args, **kwargs)

        simvis = da.blockwise(
            compute_vis_sky,
            ("row", "chan", "corr"),
            msds.UVW.data,
            ("row", "uvw"),
            freqs,
            ("chan",),
            times_var[0],
            times_var[1],
            ncorr=ncorr,
            polarisation=opts.polarisation,
            linear_basis=linear_basis,
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
                    raise RuntimeError("Could not find all required FITS files in the specified directory")

                if len(fs) > 4:
                    raise RuntimeError("Too many FITS files found in the specified directory")

            elif not fs.endswith(".fits"):
                raise IOError("Invalid FITS file specified")
        else:
            raise FileNotFoundError("FITS file/directory does not exist")

        # process FITS sky model
        predict = skymodel_from_fits(
            fs,
            ra0,
            dec0,
            freqs,
            dfreq,
            ncorr,
            linear_basis=linear_basis,
            tol=opts.pixel_tol,
            interpolation=opts.fits_sky_interp,
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
            ref_freq=predict.ref_freq,
            use_dft=predict.use_dft,
            ncorr=ncorr,
            dtype=msds.DATA.data.dtype,
            delta_ra=predict.ra_pixel_size,
            delta_dec=predict.dec_pixel_size,
            epsilon=epsilon,
            do_wstacking=opts.do_wstacking,
            noise=vis_noise,
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
