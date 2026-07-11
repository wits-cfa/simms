"""Operations for the ``simms primary-beam`` command.

Four modes, none of which run a visibility simulation:
- ``to_fits``      : sample an analytic cosine-taper beam onto a FITS beam cube.
- ``tag_ms``       : write the per-antenna telescope-name column onto an existing MS.
- ``apply``/``correct`` : multiply / divide a sky model (FITS image or ASCII components) by
  the frequency- and parallactic-angle-averaged Stokes-I power beam ``A(l, m)``.
"""

from __future__ import annotations

import logging

import numpy as np

from simms import BIN

log = logging.getLogger(BIN.primary_beam)


# --------------------------------------------------------------------- geometry


def _observation(ms):
    """Read the geometry an averaged beam needs from an MS."""
    import dask
    from daskms import xds_from_ms, xds_from_table

    from simms.skymodel.beams import array_lonlat

    ant = xds_from_table(f"{ms}::ANTENNA")[0]
    spw = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field = xds_from_table(f"{ms}::FIELD")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"])[0]
    pos, t0, t1, interval, chan_freq, phase_dir = dask.compute(
        ant.POSITION.data,
        msds.TIME.data.min(),
        msds.TIME.data.max(),
        msds.INTERVAL.data[0],
        spw.CHAN_FREQ.data[0],
        field.PHASE_DIR.data[0],
    )
    lon, lat = array_lonlat(pos)
    return {
        "t_start": float(t0),
        "duration": float(t1 - t0) + float(interval),
        "lon": lon,
        "lat": lat,
        "freqs": np.asarray(chan_freq, dtype=np.float64),
        "ra0": float(phase_dir[0][0]),
        "dec0": float(phase_dir[0][1]),
    }


def _averaged_beam(provider, ell, emm, ra0, dec0, obs, pa_step):
    """Freq- and PA-averaged power beam ``A(l, m)`` at the given directions (beam centre ra0/dec0)."""
    from simms.skymodel.beams import averaged_power_beam, pa_sample_grid

    _, chi_grid = pa_sample_grid(obs["t_start"], obs["duration"], ra0, dec0, obs["lon"], obs["lat"], pa_step)
    return averaged_power_beam(provider, ell, emm, obs["freqs"], chi_grid)


# --------------------------------------------------------------------- to-fits


def to_fits(opts):
    from astropy.coordinates import Angle

    from simms.skymodel.beams import JimBeamProvider, resolve_beam, write_beam_fits

    provider = resolve_beam(opts.beam_pattern, opts.beam_band)
    if not isinstance(provider, JimBeamProvider):
        raise RuntimeError("to-fits needs an analytic cosine-taper beam (CSV or built-in), not a FITS cube.")
    beam = provider.beam

    pixel_rad = Angle(opts.pixel_size).to_value("rad")
    npix = int(opts.npix)
    grid = (np.arange(npix) - npix // 2) * pixel_rad  # direction cosines, centred at 0

    if opts.freqs:
        start, end, nfreq = opts.freqs.split(",")
        freqs = np.linspace(float(start), float(end), int(nfreq))
    else:
        # Uniform grid over the beam's tabulated range (the FITS FREQ axis is linear; the
        # model is continuous in frequency, so a uniform resample loses nothing).
        freqs = np.linspace(beam.freqs_mhz[0] * 1e6, beam.freqs_mhz[-1] * 1e6, beam.freqs_mhz.size)

    output = opts.output or "beam.fits"
    write_beam_fits(beam, grid, grid, freqs, output)
    log.info("Wrote beam FITS cube %s (%d x %d pixels, %d channels)", output, npix, npix, freqs.size)


# --------------------------------------------------------------------- tag-ms


def _resolve_labels(opts, names):
    """Per-antenna telescope-name labels from --label, --label-map, or --from-layout."""
    nant = len(names)
    if opts.label:
        return [str(opts.label)] * nant
    if opts.label_map:
        from omegaconf import OmegaConf

        mapping = OmegaConf.to_container(OmegaConf.load(opts.label_map), resolve=True)
        missing = [n for n in names if n not in mapping]
        if missing:
            raise RuntimeError(f"--label-map has no entry for antennas: {missing[:5]}")
        return [str(mapping[n]) for n in names]
    if opts.from_layout:
        from simms.telescope.array_utilities import Array

        arr = Array(opts.from_layout)
        layout = dict(zip([str(x) for x in arr.names], [str(x) for x in arr.telescope_name]))
        missing = [n for n in names if n not in layout]
        if missing:
            raise RuntimeError(
                f"--from-layout {opts.from_layout!r} has no entry for MS antennas: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return [layout[n] for n in names]
    raise RuntimeError("tag-ms needs one of --label, --label-map or --from-layout.")


def tag_ms(opts):
    import dask
    import dask.array as da
    from daskms import xds_from_table, xds_to_table

    ms, col = opts.ms, opts.telescope_name_column
    ant = xds_from_table(f"{ms}::ANTENNA")[0]
    names = [str(x) for x in np.asarray(ant.NAME.data.compute()).astype(str)]
    labels = _resolve_labels(opts, names)
    if col in ant:
        log.warning("ANTENNA already has a %r column; overwriting it.", col)
    # casacore STRING columns are numpy object dtype, one chunk (see generate_ms).
    values = np.array(labels, dtype=object)
    tagged = ant.assign(**{col: (("row",), da.from_array(values, chunks=len(names)))})
    writes = xds_to_table([tagged], f"{ms}::ANTENNA", columns=[col], descriptor="mssubtable('ANTENNA')")
    dask.compute(writes)
    log.info("Tagged %d antennas in %s::ANTENNA[%s] -> %s", len(names), ms, col, sorted(set(labels)))


# --------------------------------------------------------------- apply / correct


def apply_correct_image(opts, invert):
    """Multiply (apply) or divide (correct) a FITS image by the averaged power beam."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from simms.skymodel.fits_skies import pixel_lm

    with fits.open(opts.fits_sky) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header
        out_dtype = hdul[0].data.dtype

    cel = WCS(header).celestial
    ra0 = np.radians(cel.wcs.crval[cel.wcs.lng])
    dec0 = np.radians(cel.wcs.crval[cel.wcs.lat])
    # Standard axis order: FITS axis 1 = RA (numpy last), axis 2 = DEC (numpy second-last).
    npix_dec, npix_ra = data.shape[-2], data.shape[-1]
    i_ra, j_dec = np.meshgrid(np.arange(npix_ra), np.arange(npix_dec))  # (npix_dec, npix_ra)

    ell, emm = pixel_lm(cel, ra0, dec0, i_ra.ravel(), j_dec.ravel())
    obs = _observation(opts.ms)
    A = _averaged_beam(provider_from(opts), ell, emm, ra0, dec0, obs, opts.beam_pa_step)
    A = A.reshape(npix_dec, npix_ra)

    if invert:
        safe = np.where(A < opts.pb_cutoff, np.nan, A)  # blank where the beam is negligible
        result = data / safe
    else:
        result = data * A

    output = opts.output or ("corrected.fits" if invert else "apparent.fits")
    fits.PrimaryHDU(data=result.astype(out_dtype, copy=False), header=header).writeto(output, overwrite=True)
    log.info("%s primary beam -> %s", "Corrected" if invert else "Applied", output)


def apply_correct_ascii(opts, invert):
    """Scale ASCII component fluxes by the averaged power beam (apply) or its inverse (correct)."""
    from simms import SCHEMADIR
    from simms.skymodel.ascii_skies import ASCIISkymodel
    from simms.utilities import radec2lm

    obs = _observation(opts.ms)
    sky = ASCIISkymodel(opts.ascii_sky, source_schema_file=f"{SCHEMADIR}/source_schema.yaml")
    lm = np.array([radec2lm(obs["ra0"], obs["dec0"], s.ra, s.dec) for s in sky.sources])
    ell, emm = (lm[:, 0], lm[:, 1]) if len(lm) else (np.array([]), np.array([]))
    A = _averaged_beam(provider_from(opts), ell, emm, obs["ra0"], obs["dec0"], obs, opts.beam_pa_step)

    lines = open(opts.ascii_sky).read().splitlines()
    header = next(ln for ln in lines if ln.strip().startswith("#format:"))
    cols = header.replace("#format:", "").strip().split()
    stokes_idx = [cols.index(c) for c in ("stokes_i", "stokes_q", "stokes_u", "stokes_v") if c in cols]

    out_lines, src = [], 0
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out_lines.append(ln)
            continue
        a = A[src]
        src += 1
        if invert and a < opts.pb_cutoff:  # source outside the beam -> drop it
            continue
        factor = (1.0 / a) if invert else a
        fields = ln.split()
        for idx in stokes_idx:
            if idx < len(fields) and fields[idx].lower() not in ("null", "none", ""):
                try:
                    fields[idx] = f"{float(fields[idx]) * factor:.8g}"
                except ValueError:
                    pass
        out_lines.append(" ".join(fields))

    output = opts.output or ("corrected.txt" if invert else "apparent.txt")
    with open(output, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    log.info("%s primary beam to %d sources -> %s", "Corrected" if invert else "Applied", src, output)


def provider_from(opts):
    from simms.skymodel.beams import resolve_beam

    return resolve_beam(opts.beam_pattern, opts.beam_band)
