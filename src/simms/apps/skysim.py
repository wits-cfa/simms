import glob
import logging
import os.path

import dask
import dask.array as da
import numpy as np
from dask import config as dask_config
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm.dask import TqdmCallback

from simms import BIN, SCHEMADIR, set_logger
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.beams import load_beam_config, resolve_antenna_beams
from simms.skymodel.fits_skies import predict_fits_channel_block, prepare_fits_sky
from simms.skymodel.mstools import (
    attach_beam,
    noise_visibilities,
    predict_channel_block,
    prepare_skymodel,
    to_full_corr,
    vis_noise_from_sefd_and_ms,
)
from simms.skymodel.wsclean_skies import prepare_wsclean_sky

log = logging.getLogger(BIN.skysim)


def _beam_row_args(msds, beam_ctx):
    """The ``(ANTENNA1, idx, ANTENNA2, idx)`` blockwise args, or ``None`` slots when off."""
    if beam_ctx:
        return (msds.ANTENNA1.data, ("row",), msds.ANTENNA2.data, ("row",))
    return (None, None, None, None)


def _array_lonlat(positions):
    """Geodetic (lon, lat) in radians from the mean of ITRF/ECEF antenna positions."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    x, y, z = np.asarray(positions, dtype=np.float64).mean(axis=0)
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    return loc.lon.to_value(u.rad), loc.lat.to_value(u.rad)


def _corr_basis(codes):
    """'linear' or 'circular' from POLARIZATION.CORR_TYPE codes; raise on anything else."""
    codes = list(codes)
    if codes in ([9, 12], [9, 10, 11, 12]):  # XX(9) XY(10) YX(11) YY(12)
        return "linear"
    if codes in ([5, 8], [5, 6, 7, 8]):  # RR(5) RL(6) LR(7) LL(8)
        return "circular"
    raise RuntimeError(
        f"Primary beam needs standard linear (XX..YY) or circular (RR..LL) correlations; got CORR_TYPE codes {codes}."
    )


class _BeamContext:
    """Everything needed to attach a primary beam to a prepared sky."""

    def __init__(self, opts, ms, msds, ra0, dec0, ncorr):
        beam_config = load_beam_config(opts.primary_beam)
        ant_ds = xds_from_table(f"{ms}::ANTENNA")[0]
        pol_ds = xds_from_table(f"{ms}::POLARIZATION")[0]
        telescope_col = opts.telescope_name_column
        if telescope_col not in ant_ds:
            # The per-antenna telescope name is required and never inferred (a single,
            # authoritative source). Fail clearly instead of guessing from e.g. dish size.
            raise RuntimeError(
                f"The ANTENNA table has no '{telescope_col}' column (the per-antenna telescope "
                f"name that selects a primary beam). Create the MS with telsim, or point "
                f"--telescope-name-column at the column that holds it."
            )
        tnames, mount, pos, t0, t1, interval, corr_type = dask.compute(
            ant_ds[telescope_col].data,
            ant_ds.MOUNT.data,
            ant_ds.POSITION.data,
            msds.TIME.data.min(),
            msds.TIME.data.max(),
            msds.INTERVAL.data[0],
            pol_ds.CORR_TYPE.data[0],
        )
        self.basis = _corr_basis(list(np.asarray(corr_type).ravel()))
        self.full_jones = opts.beam_jones == "full"
        self.ant_type, self.providers, self.type_is_altaz = resolve_antenna_beams(
            tnames, mount, beam_config, opts.beam_band
        )
        self.lon, self.lat = _array_lonlat(pos)
        # The beam is centred on the antenna pointing centre (POINTING.DIRECTION), not the phase
        # centre; source l/m are prepared for the phase centre, so keep it too for reprojection.
        from simms.skymodel.beams import read_pointing_centre

        self.phase_ra0, self.phase_dec0 = ra0, dec0
        self.ra0, self.dec0 = read_pointing_centre(ms, ra0, dec0)
        self.ncorr = ncorr
        self.t_start = float(t0)
        self.duration = float(t1 - t0) + float(interval)
        self.pa_step = opts.beam_pa_step

    @property
    def brightness_linear_basis(self):
        """Brightness is built in the linear feed basis when beams are on.

        Diagonal beams require linear correlations anyway; full Jones needs the
        feed-basis coherency (the basis transform to the MS frame is folded into the
        beam Jones).
        """
        return True

    def attach_image(self, prepared, freqs):
        """Apply an approximate PA-averaged power beam to a FITS-image sky model (any basis)."""
        from simms.skymodel.fits_skies import attach_image_beam

        if len(self.providers) > 1:
            log.warning(
                "Primary beam on the FITS-image path uses a single representative antenna "
                "type, but %d types are present; heterogeneity is ignored.",
                len(self.providers),
            )
        mid_freq = 0.5 * (float(freqs[0]) + float(freqs[-1]))
        return attach_image_beam(
            prepared,
            self.providers[0],
            bool(self.type_is_altaz[0]),
            self.ra0,
            self.dec0,
            self.lon,
            self.lat,
            self.t_start,
            self.duration,
            self.pa_step,
            mid_freq,
            phase_ra0=self.phase_ra0,
            phase_dec0=self.phase_dec0,
        )

    def attach(self, prepared):
        """Force full-correlation brightness and attach the beam grid (diagonal or full Jones)."""
        from simms.skymodel.beams import corr_basis_transform

        if self.full_jones:
            if self.ncorr != 4:
                raise RuntimeError("--beam-jones full requires 4 correlations (XX,XY,YX,YY or RR,RL,LR,LL).")
            basis_transform = corr_basis_transform(self.basis == "circular")
        elif self.basis != "linear":
            raise RuntimeError(
                "The diagonal primary beam requires linear correlations; use --beam-jones full "
                "for a circular-correlation MS."
            )
        else:
            basis_transform = None
        prepared = to_full_corr(prepared)
        return attach_beam(
            prepared,
            self.ant_type,
            self.providers,
            self.type_is_altaz,
            self.ra0,
            self.dec0,
            self.lon,
            self.lat,
            self.t_start,
            self.duration,
            self.pa_step,
            self.ncorr,
            full_jones=self.full_jones,
            basis_transform=basis_transform,
            phase_ra0=self.phase_ra0,
            phase_dec0=self.phase_dec0,
        )


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

    # Primary beam (sky models only, not noise-only runs). When enabled, the model must
    # carry every correlation so the per-feed voltage can be applied (nspec == ncorr).
    has_sky = bool(ascii_sky or wsclean_sky or fs)
    if opts.primary_beam and not has_sky:
        log.warning("--primary-beam is set but no sky model was given; the primary beam is ignored.")
    beam_ctx = _BeamContext(opts, ms, msds, ra0, dec0, ncorr) if opts.primary_beam and has_sky else None
    if beam_ctx and fs and opts.beam_jones == "full":
        log.warning("--beam-jones full is ignored on the FITS-image path (an approximate power beam is used).")

    # Channel chunking. A channel-index array carries the chan dimension into
    # da.blockwise; each predict task restricts the model to its own channels.
    nchan = freqs.size
    chan_chunks = opts.chan_chunks if opts.chan_chunks and opts.chan_chunks > 0 else nchan
    chan_ids = da.arange(nchan, chunks=chan_chunks)

    simvis = None

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
            polarisation=opts.polarisation or bool(beam_ctx),
            # Beams build the coherency in the linear feed basis (the MS-basis transform is
            # folded into the beam Jones); otherwise honour the requested pol basis.
            linear_basis=beam_ctx.brightness_linear_basis if beam_ctx else linear_basis,
            unique_times=unique_times,
        )
        if beam_ctx:
            prepared = beam_ctx.attach(prepared)

        # A blockwise index of None passes the argument through untouched, so a
        # model without transients (and no beam) must be handed a literal None, not
        # the (unblocked) TIME dask array.
        need_time = skymodel.has_transient or bool(beam_ctx)
        time_args = (msds.TIME.data, ("row",)) if need_time else (None, None)
        ant_args = _beam_row_args(msds, beam_ctx)

        simvis = da.blockwise(
            predict_channel_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            chan_ids,
            ("chan",),
            *time_args,
            *ant_args,
            out_dtype=vis_dtype,
            new_axes={"corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    elif wsclean_sky:
        # A WSClean component list shares the ASCII prediction path: it is flattened
        # into a PreparedSky and handed to the same kernel.
        prepared = prepare_wsclean_sky(wsclean_sky, freqs, ra0, dec0, ncorr=ncorr)
        if beam_ctx:
            prepared = beam_ctx.attach(prepared)

        time_args = (msds.TIME.data, ("row",)) if beam_ctx else (None, None)
        ant_args = _beam_row_args(msds, beam_ctx)

        simvis = da.blockwise(
            predict_channel_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            chan_ids,
            ("chan",),
            *time_args,
            *ant_args,
            out_dtype=vis_dtype,
            new_axes={"corr": ncorr},
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
        if beam_ctx:
            # Approximate: one PA-averaged power beam on the apparent sky (see attach_image).
            prepared = beam_ctx.attach_image(prepared, freqs)

        epsilon = 1e-7 if opts.fft_precision == "double" else 1e-6

        simvis = da.blockwise(
            predict_fits_channel_block,
            ("row", "chan", "corr"),
            prepared,
            None,
            msds.UVW.data,
            ("row", "uvw"),
            chan_ids,
            ("chan",),
            out_dtype=vis_dtype,
            epsilon=epsilon,
            do_wgridding=opts.do_wstacking,
            new_axes={"corr": ncorr},
            dtype=vis_dtype,
            concatenate=True,
        )

    # Thermal noise, added once for every path. With --seed the draw is
    # reproducible across runs at a given chunking; changing the chunking changes
    # the realisation, as dask keys each block's stream to its position in the grid.
    if vis_noise:
        noise = noise_visibilities(
            msds.DATA.data.shape,
            (msds.UVW.data.chunks[0], chan_chunks, ncorr),
            vis_noise,
            vis_dtype,
            seed=opts.seed,
        )
        simvis = noise if simvis is None else simvis + noise

    if simvis is None:
        raise RuntimeError("Nothing to simulate: provide a sky model (--ascii-sky/--fits-sky/--wsclean-sky) or --sefd.")

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
