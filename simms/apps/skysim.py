import glob
import os.path

import dask
import dask.array as da
import numpy as np
from dask import config as dask_config
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm.dask import TqdmCallback

from simms import BIN, SCHEMADIR, set_logger
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.fits_skies import predict_fits_block, prepare_fits_sky
from simms.skymodel.mstools import (
    predict_block,
    prepare_skymodel,
    sim_noise_block,
    vis_noise_from_sefd_and_ms,
)
from simms.skymodel.wsclean_skies import prepare_wsclean_sky


def runit(opts):
    # Set logger here, so subsequent modeules get it via logging.getLogger(<name>)
    set_logger(BIN.skysim, opts["log_level"])
    ms = opts.ms
    ascii_sky = opts.ascii_sky
    fs = opts.fits_sky
    wsclean_sky = opts.wsclean_sky

    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    if sum(bool(x) for x in (ascii_sky, fs, wsclean_sky)) > 1:
        raise RuntimeError("Choose a single sky model: one of --ascii-sky, --fits-sky, or --wsclean-sky.")

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

    radec0, freqs, dfreq = dask.compute(
        field_ds.PHASE_DIR.data[opts.field_id],
        spw_ds.CHAN_FREQ.data[opts.spw_id],
        spw_ds.CHAN_WIDTH.data[opts.spw_id][0],
    )
    ra0, dec0 = radec0[0][0], radec0[0][1]
    ncorr = msds.DATA.data.shape[-1]
    vis_dtype = msds.DATA.data.dtype
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

        # Transient lightcurves are defined relative to the start of the
        # observation, so the time axis has to be resolved globally rather than
        # per row block.
        unique_times = np.unique(msds.TIME.data.compute()) if skymodel.has_transient else None

        # Prepared once and shared by every block, rather than rebuilt per block.
        # Sources are summed in double precision regardless of the column dtype.
        prepared = prepare_skymodel(
            skymodel,
            freqs,
            ra0,
            dec0,
            ncorr=ncorr,
            polarisation=opts.polarisation,
            linear_basis=linear_basis,
            unique_times=unique_times,
        )

        # A blockwise index of None passes the argument through untouched, so a
        # model without transients must be handed a literal None, not the
        # (unblocked) TIME dask array.
        time_args = (msds.TIME.data, ("row",)) if skymodel.has_transient else (None, None)

        simvis = da.blockwise(
            predict_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            *time_args,
            noise_vis=vis_noise,
            out_dtype=vis_dtype,
            new_axes={"chan": freqs.size, "corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    elif wsclean_sky:
        # A WSClean component list shares the ASCII prediction path: it is flattened
        # into a PreparedSky and handed to the same kernel.
        prepared = prepare_wsclean_sky(wsclean_sky, freqs, ra0, dec0, ncorr=ncorr)

        simvis = da.blockwise(
            predict_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            noise_vis=vis_noise,
            out_dtype=vis_dtype,
            new_axes={"chan": freqs.size, "corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    # TODO(Sphe,Senkhosi) This is too tailored to a specific use case and input file schema.
    # We need more generic API for handling multiple FITS images as input.
    # Something like we do in skymods.skymodel_from_fits(stack_axis)
    elif fs:
        if os.path.exists(fs):
            if os.path.isdir(fs):
                fits_dir = fs
                fs = []
                try:
                    fs.append(glob.glob(f"{fits_dir}/*I.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*Q.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*U.fits")[0])
                    fs.append(glob.glob(f"{fits_dir}/*V.fits")[0])
                except IndexError:
                    raise RuntimeError("Could not find all required FITS files in the specified directory")

                if len(fs) > 4:
                    raise RuntimeError("Too many FITS files found in the specified directory")

            elif not fs.endswith(".fits"):
                raise IOError("Invalid FITS file specified")
        else:
            raise FileNotFoundError("FITS file/directory does not exist")

        # Prepared once and shared by every row block.
        prepared = prepare_fits_sky(
            fs,
            ra0,
            dec0,
            freqs,
            dfreq,
            ncorr,
            nrow=msds.UVW.data.shape[0],
            linear_basis=linear_basis,
            polarisation=opts.polarisation,
            tol=opts.pixel_tol,
            backend=opts.predict_backend,
            spectrum=opts.fits_spectrum,
            spi_maps=opts.fits_spi,
            ref_freq=opts.fits_ref_freq,
            spectrum_order=opts.fits_spectrum_order,
            interpolation=opts.fits_sky_interp,
        )

        epsilon = 1e-7 if opts.fft_precision == "double" else 1e-6

        simvis = da.blockwise(
            predict_fits_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            noise_vis=vis_noise,
            out_dtype=vis_dtype,
            epsilon=epsilon,
            do_wgridding=opts.do_wstacking,
            new_axes={"chan": freqs.size, "corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    elif opts.sefd:
        # Simulate noise if no skymodel is given. UVW only supplies the row
        # extent and its chunking; blockwise cannot size the output without it.
        simvis = da.blockwise(
            sim_noise_block,
            ("row", "chan", "corr"),
            msds.UVW.data,
            ("row", "uvw"),
            freqs,
            ("chan",),
            ncorr=ncorr,
            vis_noise=vis_noise,
            out_dtype=vis_dtype,
            dtype=vis_dtype,
            new_axes={"corr": ncorr},
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
