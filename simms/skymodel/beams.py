"""Primary-beam models for skysim.

The cosine-taper ("JimBeam") voltage model in this module is re-implemented from
katbeam (https://github.com/ska-sa/katbeam), Copyright (c) 2020, National Research
Foundation (SARAO), released under the BSD 3-Clause License, originally authored by
Mattieu de Villiers. We vendor the maths and the bundled L/UHF coefficient tables
(``simms/skymodel/beam_data/``) rather than depend on the package so that we can
(a) drop an unmaintained dependency and (b) load arbitrary coefficient CSVs -- e.g.
the MeerKAT-Extension ``MKAT-EA-*`` tables -- which upstream katbeam cannot.

A cosine aperture taper (Essential Radio Astronomy, Condon & Ransom, 2016) models
the co-polarised primary beam. It is parameterised, per frequency and per feed
(H/V), by a pointing offset ("squint", ``l0``/``m0``) and a full-width-half-maximum
in each of the two feed axes. ``HH``/``VV`` are *voltage* patterns: they peak at 1
on axis and equal ``sqrt(0.5)`` at the half-power point, so ``|HH|**2`` is the power
beam and Stokes I is ``0.5*(|HH|**2 + |VV|**2)``.
"""

from __future__ import annotations

import csv
import logging
from importlib import resources
from pathlib import Path

import numpy as np

from simms import BIN

log = logging.getLogger(BIN.skysim)

# Seconds per day; MS TIME is in seconds (MJD * 86400), casacore convention.
SECONDS_PER_DAY = 86400.0


def local_sidereal_time(times: np.ndarray, lon: float) -> np.ndarray:
    """Mean local sidereal time (radians) for MS ``TIME`` seconds at longitude ``lon`` (rad).

    Used to form the *local* hour angle ``LST - RA`` that sets the sky orientation
    (parallactic angle, elevation) at the site. This is a distinct quantity from the
    Greenwich hour angle that :func:`simms.telescope.array_utilities.Array.uvgen` uses
    for the UVW rotation of its ECEF baselines -- the two differ by the array longitude
    and both are correct in their own frame.
    """
    from astropy import units as u
    from astropy.time import Time

    t = Time(np.atleast_1d(np.asarray(times, dtype=np.float64)) / SECONDS_PER_DAY, format="mjd", scale="utc")
    return t.sidereal_time("mean", longitude=lon * u.rad).to_value("rad")


def parallactic_angle(times: np.ndarray, ra0: float, dec0: float, lon: float, lat: float) -> np.ndarray:
    """Parallactic angle (radians) at the field centre for each timestamp.

    Parameters
    ----------
    times : numpy.ndarray
        MS ``TIME`` values in seconds, shape ``(ntime,)``.
    ra0, dec0 : float
        Field-centre right ascension and declination (radians).
    lon, lat : float
        Array-reference geodetic longitude and latitude (radians).

    Returns
    -------
    numpy.ndarray
        Parallactic angle per timestamp, shape ``(ntime,)``. This is the position
        angle of the zenith at the field centre; alt-az feeds rotate the beam on the
        sky by this angle.
    """
    hour_angle = local_sidereal_time(times, lon) - ra0
    return np.arctan2(
        np.sin(hour_angle),
        np.tan(lat) * np.cos(dec0) - np.sin(dec0) * np.cos(hour_angle),
    )


# r_FWHM: normalises the taper so the half-power point falls at r = 0.5, i.e.
# |taper(0.5)|**2 = 0.5. katbeam truncates this to 1.18896478.
R_FWHM = 1.1889647809329453

# Bundled coefficient tables. The MKAT-AA-* models are vendored from katbeam
# (BSD-3-Clause, (c) 2020 SARAO). The MKAT-MA-* (MeerKAT) and MKAT-EA-* (MeerKAT
# Extension) tables are transcribed from SARAO report SSA-0004B-002 (see beam_data/NOTICE).
BUILTIN_BEAMS = {
    "MKAT-AA-L-JIM-2020": "MKAT-AA-L-JIM-2020.csv",
    "MKAT-AA-UHF-JIM-2020": "MKAT-AA-UHF-JIM-2020.csv",
    "MKAT-MA-L-JIM-2026": "MKAT-MA-L-JIM-2026.csv",  # MeerKAT (MK), L band
    "MKAT-EA-L-JIM-2026": "MKAT-EA-L-JIM-2026.csv",  # MeerKAT Extension (MKE), SKA band 2
    "MKAT-MA-S-JIM-2020": "MKAT-MA-S-JIM-2020.csv",  # MeerKAT (MK), S band
    "MKAT-EA-S3-JIM-2026": "MKAT-EA-S3-JIM-2026.csv",  # MeerKAT Extension (MKE), S band (S3)
}


def cosine_taper(r: np.ndarray) -> np.ndarray:
    """Cosine-taper voltage pattern ``cos(pi*rr)/(1 - 4*rr**2)`` with ``rr = r*R_FWHM``.

    ``r`` is normalised so the half-power point is at ``r = 0.5``. The ``rr = 0.5``
    singularity of ``1 - 4*rr**2`` is removable (limit ``pi/4``) and handled here.
    """
    rr = np.asarray(r, dtype=np.float64) * R_FWHM
    denom = 1.0 - 4.0 * rr * rr
    near = np.abs(denom) < 1e-12
    safe = np.where(near, 1.0, denom)
    out = np.cos(np.pi * rr) / safe
    return np.where(near, np.pi / 4.0, out)


class CosineTaperBeam:
    """Frequency-interpolated cosine-taper voltage beam for the H and V feeds.

    Parameters
    ----------
    freqs_mhz : numpy.ndarray
        Tabulated frequencies (MHz), shape ``(nfreq,)``.
    squint_deg : numpy.ndarray
        Per-feed pointing offsets in degrees, shape ``(4, nfreq)`` ordered
        ``Hx, Hy, Vx, Vy``.
    fwhm_deg : numpy.ndarray
        Per-feed FWHMs in degrees, same shape/ordering as ``squint_deg``.
    name : str, optional
        Label for diagnostics.
    """

    def __init__(self, freqs_mhz, squint_deg, fwhm_deg, name: str = ""):
        self.freqs_mhz = np.ascontiguousarray(freqs_mhz, dtype=np.float64)
        self.squint_deg = np.ascontiguousarray(squint_deg, dtype=np.float64)
        self.fwhm_deg = np.ascontiguousarray(fwhm_deg, dtype=np.float64)
        if self.squint_deg.shape != (4, self.freqs_mhz.size):
            raise ValueError("squint_deg must have shape (4, nfreq)")
        if self.fwhm_deg.shape != (4, self.freqs_mhz.size):
            raise ValueError("fwhm_deg must have shape (4, nfreq)")
        self.name = name

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_csv(cls, path, name: str = "") -> "CosineTaperBeam":
        """Load a katbeam-format coefficient CSV.

        The file has two header rows (column names, then units) followed by rows of
        ``freq[MHz], Hx, Hy, Vx, Vy squint[arcmin], Hx, Hy, Vx, Vy fwhm[arcmin]``.
        """
        path = Path(path)
        with open(path, newline="") as fh:
            rows = [row for row in csv.reader(fh) if row and row[0].strip()]
        # Drop the two header rows (name row, units row).
        table = np.array([[float(c) for c in row] for row in rows[2:]], dtype=np.float64)
        freqs_mhz = table[:, 0]
        squint_deg = table[:, 1:5].T / 60.0  # arcmin -> degrees
        fwhm_deg = table[:, 5:9].T / 60.0
        return cls(freqs_mhz, squint_deg, fwhm_deg, name=name or path.stem)

    @classmethod
    def from_builtin(cls, name: str) -> "CosineTaperBeam":
        """Load one of the bundled models in :data:`BUILTIN_BEAMS`."""
        if name not in BUILTIN_BEAMS:
            raise ValueError(f"Unknown built-in beam {name!r}; available: {list(BUILTIN_BEAMS)}")
        with resources.as_file(resources.files("simms.skymodel.beam_data") / BUILTIN_BEAMS[name]) as p:
            return cls.from_csv(p, name=name)

    @classmethod
    def load(cls, spec: str) -> "CosineTaperBeam":
        """Load by built-in name if known, otherwise treat ``spec`` as a CSV path."""
        if spec in BUILTIN_BEAMS:
            return cls.from_builtin(spec)
        return cls.from_csv(spec)

    # -- evaluation ---------------------------------------------------------

    def _interp(self, freqs_mhz: np.ndarray):
        """Linearly interpolate squint/FWHM to ``freqs_mhz`` -> two ``(4, nchan)`` arrays."""
        freqs_mhz = np.atleast_1d(np.asarray(freqs_mhz, dtype=np.float64))
        squint = np.stack([np.interp(freqs_mhz, self.freqs_mhz, s) for s in self.squint_deg])
        fwhm = np.stack([np.interp(freqs_mhz, self.freqs_mhz, f) for f in self.fwhm_deg])
        return squint, fwhm

    def voltages(self, x_deg: np.ndarray, y_deg: np.ndarray, freqs_mhz: np.ndarray) -> np.ndarray:
        """Voltage patterns for both feeds.

        Parameters
        ----------
        x_deg, y_deg : numpy.ndarray
            Feed-frame coordinates in degrees (``x`` toward +AZ, ``y`` toward +EL),
            shape ``(nsrc,)``.
        freqs_mhz : numpy.ndarray
            Frequencies in MHz, shape ``(nchan,)``.

        Returns
        -------
        numpy.ndarray
            Real voltage patterns of shape ``(nsrc, nchan, 2)``; the last axis is
            feed 0 = H/X, 1 = V/Y.
        """
        x = np.asarray(x_deg, dtype=np.float64)[:, None]
        y = np.asarray(y_deg, dtype=np.float64)[:, None]
        squint, fwhm = self._interp(freqs_mhz)  # (4, nchan)
        sq = squint[:, None, :]  # (4, 1, nchan)
        fw = fwhm[:, None, :]
        rh = np.sqrt(((x - sq[0]) / fw[0]) ** 2 + ((y - sq[1]) / fw[1]) ** 2)
        rv = np.sqrt(((x - sq[2]) / fw[2]) ** 2 + ((y - sq[3]) / fw[3]) ** 2)
        return np.stack([cosine_taper(rh), cosine_taper(rv)], axis=-1)


# Built-in analytic models selectable by band shorthand in a beam-config entry.
BAND_BUILTINS = {
    "L": "MKAT-AA-L-JIM-2020",
    "UHF": "MKAT-AA-UHF-JIM-2020",
}


class BeamProvider:
    """Evaluate a per-feed voltage beam, de-rotating the sky into the feed frame.

    Subclasses implement :meth:`_eval` in the feed frame. The base handles the
    parallactic-angle rotation so a source at sky direction cosines ``(l, m)`` is
    evaluated at the feed-frame coordinates ``R(-chi) . (l, m)``.
    """

    def _eval(self, l_feed: np.ndarray, m_feed: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Feed-frame voltages ``(nsrc, nchan, 2)`` (feed 0 = H/X, 1 = V/Y)."""
        raise NotImplementedError

    def _eval_jones(self, l_feed: np.ndarray, m_feed: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Feed-frame 2x2 voltage Jones ``(nsrc, nchan, 2, 2)``.

        Default is the leakage-free diagonal ``diag(g^H, g^V)`` from :meth:`_eval`;
        providers with cross-polarisation (measured FITS cubes) override this.
        """
        g = self._eval(l_feed, m_feed, freqs)  # (nsrc, nchan, 2)
        jones = np.zeros(g.shape[:2] + (2, 2), dtype=np.complex128)
        jones[..., 0, 0] = g[..., 0]
        jones[..., 1, 1] = g[..., 1]
        return jones

    @staticmethod
    def _rotate_to_feed(ell, emm, angle):
        """Rotate sky cosines ``(l, m)`` into the feed frame by ``-angle`` (identity at 0)."""
        if angle:
            c, s = np.cos(angle), np.sin(angle)
            return ell * c + emm * s, -ell * s + emm * c
        return ell, emm

    def jones(self, ell, emm, freqs, chi) -> np.ndarray:
        """2x2 voltage Jones per parallactic-angle sample.

        Same inputs as :meth:`voltage`; returns ``(ntime, nsrc, nchan, 2, 2)`` complex.
        """
        ell = np.asarray(ell, dtype=np.float64)
        emm = np.asarray(emm, dtype=np.float64)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        chi = np.atleast_1d(np.asarray(chi, dtype=np.float64))
        out = np.empty((chi.size, ell.size, freqs.size, 2, 2), dtype=np.complex128)
        for ti, angle in enumerate(chi):
            l_feed, m_feed = self._rotate_to_feed(ell, emm, angle)
            out[ti] = self._eval_jones(l_feed, m_feed, freqs)
        return out

    def voltage(self, ell, emm, freqs, chi) -> np.ndarray:
        """Voltage beam for each parallactic angle.

        Parameters
        ----------
        ell, emm : numpy.ndarray
            Sky-frame direction cosines ``(l, m)``, shape ``(nsrc,)``.
        freqs : numpy.ndarray
            Frequencies in Hz, shape ``(nchan,)``.
        chi : numpy.ndarray
            Parallactic angle per sample (radians), shape ``(ntime,)``. Pass zeros
            for a non-rotating (e.g. equatorial-mount) beam.

        Returns
        -------
        numpy.ndarray
            Complex voltages of shape ``(ntime, nsrc, nchan, 2)``.
        """
        ell = np.asarray(ell, dtype=np.float64)
        emm = np.asarray(emm, dtype=np.float64)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        chi = np.atleast_1d(np.asarray(chi, dtype=np.float64))
        out = np.empty((chi.size, ell.size, freqs.size, 2), dtype=np.complex128)
        for ti, angle in enumerate(chi):
            l_feed, m_feed = self._rotate_to_feed(ell, emm, angle)
            out[ti] = self._eval(l_feed, m_feed, freqs)
        return out


class UnityBeamProvider(BeamProvider):
    """A flat, unity beam (no attenuation). Used for antennas with no configured beam."""

    def _eval(self, l_feed, m_feed, freqs):
        return np.ones((l_feed.size, freqs.size, 2), dtype=np.float64)


class JimBeamProvider(BeamProvider):
    """Analytic cosine-taper ("JimBeam") voltage provider wrapping :class:`CosineTaperBeam`."""

    def __init__(self, beam: CosineTaperBeam):
        self.beam = beam

    def _eval(self, l_feed, m_feed, freqs):
        # The report grids the beam in SIN direction cosines scaled to degrees
        # (l_deg = 180/pi * l), and katbeam consumes MHz.
        x_deg = np.degrees(l_feed)
        y_deg = np.degrees(m_feed)
        return self.beam.voltages(x_deg, y_deg, freqs / 1e6)


def _ascending(grid, values, axis):
    """Return ``(grid, values)`` with ``grid`` ascending, flipping ``values`` along ``axis`` if not."""
    if grid.size > 1 and grid[0] > grid[-1]:
        return grid[::-1].copy(), np.flip(values, axis=axis)
    return grid, values


class FitsBeamProvider(BeamProvider):
    """Per-feed voltage beam interpolated from a gridded FITS cube (measured/eidos beams).

    The cube is a regular grid of complex per-feed voltages over ``(l, m)`` direction
    cosines and frequency; it is defined in the *feed frame* (as holography/eidos beams
    are), so the :class:`BeamProvider` base rotates the sky ``(l, m)`` into that frame by
    ``-chi`` before this interpolates. Bilinear in ``(l, m)``, linear in frequency, zero
    outside the grid.

    Use :meth:`from_fits` for the on-disk layout, or :meth:`from_arrays` to build one
    directly (e.g. in tests).
    """

    def __init__(self, l_grid, m_grid, freqs_hz, values, name: str = ""):
        from scipy.interpolate import RegularGridInterpolator

        l_grid = np.ascontiguousarray(l_grid, dtype=np.float64)
        m_grid = np.ascontiguousarray(m_grid, dtype=np.float64)
        freqs_hz = np.ascontiguousarray(freqs_hz, dtype=np.float64)
        values = np.asarray(values, dtype=np.complex128)  # (nl, nm, nfreq, K); K=2 [HH,VV] or 4 [HH,HV,VH,VV]
        nk = values.shape[-1]
        if nk not in (2, 4) or values.shape != (l_grid.size, m_grid.size, freqs_hz.size, nk):
            raise ValueError("values must have shape (nl, nm, nfreq, 2) or (nl, nm, nfreq, 4)")
        # RegularGridInterpolator requires strictly ascending axes; FITS L/M axes often
        # descend (negative CDELT), so flip any descending axis and its data together.
        l_grid, values = _ascending(l_grid, values, 0)
        m_grid, values = _ascending(m_grid, values, 1)
        freqs_hz, values = _ascending(freqs_hz, values, 2)
        self.name = name
        self.has_leakage = nk == 4
        self.l_grid, self.m_grid, self.freqs_hz = l_grid, m_grid, freqs_hz
        # One interpolator per stored entry; a single-channel cube can't interpolate in
        # frequency, so drop that axis and interpolate in (l, m) only.
        self._single_freq = freqs_hz.size == 1
        grid = (l_grid, m_grid) if self._single_freq else (l_grid, m_grid, freqs_hz)
        self._interp = [
            RegularGridInterpolator(
                grid,
                values[:, :, 0, k] if self._single_freq else values[..., k],
                bounds_error=False,
                fill_value=0.0,
            )
            for k in range(nk)
        ]

    def _interp_entries(self, l_feed, m_feed, freqs):
        """Interpolate every stored entry -> ``(nsrc, nchan, nk)``."""
        nsrc, nchan, nk = l_feed.size, freqs.size, len(self._interp)
        out = np.empty((nsrc, nchan, nk), dtype=np.complex128)
        if self._single_freq:
            pts = np.stack([l_feed, m_feed], axis=-1)  # (nsrc, 2)
            for k in range(nk):
                out[:, :, k] = self._interp[k](pts)[:, None]
        else:
            ll = np.repeat(l_feed, nchan)
            mm = np.repeat(m_feed, nchan)
            ff = np.tile(np.asarray(freqs, dtype=np.float64), nsrc)
            pts = np.stack([ll, mm, ff], axis=-1)
            for k in range(nk):
                out[:, :, k] = self._interp[k](pts).reshape(nsrc, nchan)
        return out

    def _eval(self, l_feed, m_feed, freqs):
        # Diagonal feed voltages (H, V): the co-pol entries HH, VV.
        entries = self._interp_entries(l_feed, m_feed, freqs)
        return np.stack([entries[..., 0], entries[..., -1]], axis=-1)  # [HH, VV]

    def _eval_jones(self, l_feed, m_feed, freqs):
        entries = self._interp_entries(l_feed, m_feed, freqs)
        jones = np.zeros(entries.shape[:2] + (2, 2), dtype=np.complex128)
        if self.has_leakage:  # [HH, HV, VH, VV]
            jones[..., 0, 0] = entries[..., 0]
            jones[..., 0, 1] = entries[..., 1]
            jones[..., 1, 0] = entries[..., 2]
            jones[..., 1, 1] = entries[..., 3]
        else:  # [HH, VV] diagonal
            jones[..., 0, 0] = entries[..., 0]
            jones[..., 1, 1] = entries[..., 1]
        return jones

    @classmethod
    def from_arrays(cls, l_grid, m_grid, freqs_hz, values, name: str = "") -> "FitsBeamProvider":
        """Build from explicit grids and a ``(nl, nm, nfreq, K)`` cube (K=2 [HH,VV] or 4 [HH,HV,VH,VV])."""
        return cls(l_grid, m_grid, freqs_hz, values, name=name)

    @classmethod
    def from_fits(cls, path, name: str = "") -> "FitsBeamProvider":
        """Load a per-feed voltage beam cube.

        The primary HDU leading axis holds the feed voltages as real/imag pairs:
        **4 planes** ``[HH, VV]×(real, imag)`` for a diagonal beam, or **8 planes**
        ``[HH, HV, VH, VV]×(real, imag)`` for a full 2x2 Jones (leakage). A linear WCS
        gives the ``L``/``M`` axes in degrees (SIN direction cosines scaled to degrees,
        ``l_deg = 180/pi * l``) and the ``FREQ`` axis in Hz.
        """
        from astropy.io import fits

        with fits.open(path) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float64)  # (4 or 8, nfreq, nm, nl)
        if data.ndim != 4 or data.shape[0] not in (4, 8):
            raise ValueError("FITS beam cube must have shape (4 or 8, nfreq, nm, nl): feed voltages real/imag.")
        nplane, nfreq, nm, nl = data.shape

        def _axis(fits_axis, n):
            crpix = hdr[f"CRPIX{fits_axis}"]
            crval = hdr[f"CRVAL{fits_axis}"]
            cdelt = hdr[f"CDELT{fits_axis}"]
            return (np.arange(n) - (crpix - 1)) * cdelt + crval

        # FITS axis 1 <-> last numpy axis (nl), axis 2 <-> nm, axis 3 <-> nfreq.
        l_grid = np.radians(_axis(1, nl))  # degrees -> direction cosine (small-angle)
        m_grid = np.radians(_axis(2, nm))
        freqs_hz = _axis(3, nfreq)

        # Even planes are real parts, odd planes imaginary: (nk, nfreq, nm, nl), nk in {2, 4}.
        complex_planes = data[0::2] + 1j * data[1::2]
        values = np.ascontiguousarray(complex_planes.transpose(3, 2, 1, 0))  # (nl, nm, nfreq, nk)
        return cls(l_grid, m_grid, freqs_hz, values, name=name or str(path))


def _build_jimbeam(spec, beam_band: str) -> JimBeamProvider:
    """Build a JimBeam provider from a beam-config spec (band shorthand, built-in name, or CSV path)."""
    if spec in (True, None, ""):
        spec = beam_band
    spec = str(spec)
    if spec in BUILTIN_BEAMS:
        beam = CosineTaperBeam.from_builtin(spec)
    elif spec in BAND_BUILTINS:
        beam = CosineTaperBeam.from_builtin(BAND_BUILTINS[spec])
    else:
        beam = CosineTaperBeam.from_csv(spec)
    return JimBeamProvider(beam)


def _build_provider(label: str, beam_config, beam_band: str) -> BeamProvider:
    entry = beam_config.get(label) if beam_config else None
    if not entry:
        log.warning("No beam configured for telescope_name %r; using a unity (flat) beam.", label)
        return UnityBeamProvider()
    if "jimbeam" in entry:
        return _build_jimbeam(entry["jimbeam"], beam_band)
    if "fits" in entry:
        return FitsBeamProvider.from_fits(entry["fits"])
    log.warning("Beam entry for %r has no 'jimbeam' or 'fits' key; using a unity beam.", label)
    return UnityBeamProvider()


def load_beam_config(path) -> dict:
    """Load a beam-config YAML mapping each ``TELESCOPE_NAME`` to a provider spec."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def resolve_antenna_beams(telescope_names, mount, beam_config, beam_band: str = "L"):
    """Map antennas to beam types and build one provider per type.

    Parameters
    ----------
    telescope_names : sequence of str
        Per-antenna ``ANTENNA.TELESCOPE_NAME`` values, length ``nant``.
    mount : sequence of str
        Per-antenna ``ANTENNA.MOUNT`` values, length ``nant``.
    beam_config : dict or None
        Mapping of telescope_name -> provider spec (see :func:`load_beam_config`).
    beam_band : str
        Default band for JimBeam entries that omit an explicit model.

    Returns
    -------
    ant_type : numpy.ndarray
        Per-antenna type index, shape ``(nant,)``.
    providers : list of BeamProvider
        One provider per type, indexed by ``ant_type``.
    type_is_altaz : numpy.ndarray
        Whether each type's mount rotates the beam (ALT-AZ), shape ``(ntype,)``.
    """
    telescope_names = [str(t) for t in np.asarray(telescope_names).astype(str)]
    mount = [str(m) for m in np.asarray(mount).astype(str)]
    labels = list(dict.fromkeys(telescope_names))  # unique, insertion-ordered

    providers = []
    type_is_altaz = []
    for label in labels:
        providers.append(_build_provider(label, beam_config, beam_band))
        first = telescope_names.index(label)
        type_is_altaz.append("ALT-AZ" in mount[first].upper())

    index = {label: i for i, label in enumerate(labels)}
    ant_type = np.array([index[t] for t in telescope_names], dtype=np.int64)
    return ant_type, providers, np.array(type_is_altaz, dtype=bool)


# Hard cap on the parallactic-angle grid. Sizing by the peak PA rate diverges as a
# source approaches the zenith (rate -> infinity at transit through zenith), which would
# otherwise blow up the beam grid; cap it and warn that the fast region is under-resolved.
MAX_PA_SAMPLES = 2048


def pa_sample_grid(t_start, duration, ra0, dec0, lon, lat, pa_step_deg):
    """A parallactic-angle sample grid spanning the observation, uniform in time.

    The grid is uniform in *time* (so a row is indexed by its timestamp, monotonically
    and without ``arctan2`` wrap issues) and sized by the *maximum* PA rate over the span
    so a fast (near-zenith) transit is resolved to ``pa_step_deg``, up to
    :data:`MAX_PA_SAMPLES` samples.

    Returns
    -------
    tgrid : numpy.ndarray
        Sample times (MS seconds), shape ``(n_pa,)``.
    chi_grid : numpy.ndarray
        Unwrapped parallactic angle at each sample (radians), shape ``(n_pa,)``.
    """
    if duration <= 0:
        n_pa = 2
    else:
        fine_t = t_start + np.linspace(0.0, duration, 512)
        chi_fine = np.unwrap(parallactic_angle(fine_t, ra0, dec0, lon, lat))
        rate = np.abs(np.gradient(chi_fine, fine_t)).max()  # rad/s
        span_deg = np.degrees(rate * duration)
        n_pa = max(2, int(np.ceil(span_deg / max(pa_step_deg, 1e-6))) + 1)
        if n_pa > MAX_PA_SAMPLES:
            log.warning(
                "Parallactic-angle sampling wants %d grid points (fast/near-zenith "
                "transit); capping at %d. The beam near transit is under-resolved -- "
                "raise --beam-pa-step or avoid a near-zenith field for full accuracy.",
                n_pa,
                MAX_PA_SAMPLES,
            )
            n_pa = MAX_PA_SAMPLES
    tgrid = t_start + np.linspace(0.0, duration, n_pa)
    chi_grid = np.unwrap(parallactic_angle(tgrid, ra0, dec0, lon, lat))
    return tgrid, chi_grid


def build_beam_grid(providers, type_is_altaz, ell, emm, freqs, chi_grid):
    """Sample every type's voltage beam on the PA grid.

    Returns an array of shape ``(ntype, n_pa, nsrc, nchan, 2)``. Stored as ``complex64``
    (halving this large array vs ``complex128``); a beam voltage is O(1), so single
    precision is ample and visibilities still accumulate in double. Alt-az types use
    ``chi_grid``; others are evaluated at zero parallactic angle (no rotation).
    """
    ntype = len(providers)
    grid = np.empty((ntype, chi_grid.size, ell.size, freqs.size, 2), dtype=np.complex64)
    zeros = np.zeros_like(chi_grid)
    for ti, prov in enumerate(providers):
        use_chi = chi_grid if type_is_altaz[ti] else zeros
        grid[ti] = prov.voltage(ell, emm, freqs, use_chi)  # complex128 -> complex64 on assign
    return grid


# Linear -> circular feed transform, consistent with simms' circular brightness
# (R=(X+iY)/sqrt2, L=(X-iY)/sqrt2): verified S @ B_linear @ S^H == B_circular.
_S_LIN2CIRC = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128)


def corr_basis_transform(is_circular: bool) -> np.ndarray:
    """2x2 transform folded into the beam Jones so ``V`` lands in the MS correlation basis.

    Identity for a linear-correlation MS; the constant linear->circular feed matrix for a
    circular one (the feeds are physically linear, so circular correlations are a fixed
    rotation of the linear ones).
    """
    return _S_LIN2CIRC.copy() if is_circular else np.eye(2, dtype=np.complex128)


def build_beam_grid_jones(providers, type_is_altaz, ell, emm, freqs, chi_grid, basis_transform):
    """Sample every type's 2x2 voltage Jones on the PA grid, folding the basis transform.

    Returns ``(ntype, n_pa, nsrc, nchan, 2, 2)`` complex64 holding ``E' = S . E``, so the
    kernel's ``V = E'_p B_feed E'_q^H = S (E_p B E_q^H) S^H`` lands in the MS correlation
    basis (``B_feed`` is the linear-feed coherency). Alt-az types use ``chi_grid``.
    """
    ntype = len(providers)
    grid = np.empty((ntype, chi_grid.size, ell.size, freqs.size, 2, 2), dtype=np.complex64)
    zeros = np.zeros_like(chi_grid)
    for ti, prov in enumerate(providers):
        use_chi = chi_grid if type_is_altaz[ti] else zeros
        jones = prov.jones(ell, emm, freqs, use_chi)  # (n_pa, nsrc, nchan, 2, 2)
        grid[ti] = np.einsum("ij,tsfjk->tsfik", basis_transform, jones)
    return grid


def image_power_beam(provider, is_altaz, ell, emm, freqs, chi_grid):
    """Parallactic-angle-averaged power beam ``<0.5(|g^X|^2 + |g^V|^2)>`` at each point.

    For the FITS-*image* path, which grids one apparent sky for all baselines and times:
    there is no per-baseline beam, so the beam is averaged over the observation's
    parallactic-angle range (a single sample when ``is_altaz`` is False). Loops over
    frequency and PA to keep the working set at ``O(npts)`` for large images.

    Parameters
    ----------
    provider : BeamProvider
        A single representative antenna beam.
    is_altaz : bool
        Whether to average over ``chi_grid`` (True) or evaluate once at chi = 0.
    ell, emm : numpy.ndarray
        Direction cosines of the points (pixels/components), shape ``(npts,)``.
    freqs : numpy.ndarray
        Frequencies (Hz), shape ``(nchan,)``.
    chi_grid : numpy.ndarray
        Parallactic-angle samples (radians).

    Returns
    -------
    numpy.ndarray
        Real power beam of shape ``(npts, nchan)`` in ``[0, ~1]``.
    """
    ell = np.asarray(ell, dtype=np.float64)
    emm = np.asarray(emm, dtype=np.float64)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    chis = chi_grid if is_altaz else np.zeros(1)
    power = np.empty((ell.size, freqs.size))
    for k in range(freqs.size):
        fk = freqs[k : k + 1]
        acc = np.zeros(ell.size)
        for chi in chis:
            g = provider.voltage(ell, emm, fk, np.array([chi]))  # (1, npts, 1, 2)
            acc += 0.5 * (np.abs(g[0, :, 0, 0]) ** 2 + np.abs(g[0, :, 0, 1]) ** 2)
        power[:, k] = acc / chis.size
    return power


def corr_feed_maps(ncorr):
    """Feed indices for each correlation in the linear basis.

    Returns ``(corr_feed_p, corr_feed_q)`` giving, per correlation, the feed (0=H/X,
    1=V/Y) of the first and second antenna. 4-corr is ``[XX, XY, YX, YY]``; 2-corr is
    ``[XX, YY]``. Circular basis is unsupported by the diagonal per-feed model.
    """
    if ncorr == 4:
        return np.array([0, 0, 1, 1], dtype=np.int64), np.array([0, 1, 0, 1], dtype=np.int64)
    if ncorr == 2:
        return np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)
    raise ValueError(f"Primary beam supports 2 or 4 correlations, got {ncorr}.")


def array_lonlat(positions):
    """Geodetic (lon, lat) in radians from the mean of ITRF/ECEF antenna positions."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    x, y, z = np.asarray(positions, dtype=np.float64).mean(axis=0)
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    return loc.lon.to_value(u.rad), loc.lat.to_value(u.rad)


def read_pointing_centre(ms, fallback_ra0, fallback_dec0):
    """Antenna pointing centre (radians) from ``POINTING.DIRECTION``.

    This is where the dishes point, and hence where the primary beam is centred -- distinct
    from ``FIELD.PHASE_DIR`` (the correlator phase centre, a freely-shiftable quantity). Reads
    the poly order-0 term of the first row's direction (J2000 RA/Dec) and falls back to the
    given phase centre, with a warning, when the MS has no usable POINTING table.
    """
    from daskms import xds_from_table

    try:
        pnt = xds_from_table(f"{ms}::POINTING")[0]
        direction = pnt.DIRECTION.data
        if direction.shape[0] == 0:
            raise ValueError("empty POINTING table")
        radec = np.asarray(direction[0, 0].compute(), dtype=np.float64)
        return float(radec[0]), float(radec[1])
    except Exception as exc:
        log.warning("No usable POINTING.DIRECTION (%s); using the phase centre as the beam centre.", exc)
        return float(fallback_ra0), float(fallback_dec0)


def reproject_lm(ell, emm, from_ra0, from_dec0, to_ra0, to_dec0):
    """Re-reference direction cosines from one tangent centre to another (a no-op if they match).

    Source ``(l, m)`` prepared for the phase centre are re-expressed relative to the beam
    (pointing) centre, so the beam is sampled at each source's offset from where the dish points.
    """
    if from_ra0 == to_ra0 and from_dec0 == to_dec0:
        return ell, emm
    from simms.skymodel.fits_skies import lm_to_radec
    from simms.utilities import radec2lm

    ra, dec = lm_to_radec(np.asarray(ell), np.asarray(emm), from_ra0, from_dec0)
    return radec2lm(to_ra0, to_dec0, ra, dec)


def resolve_beam(spec, band: str = "L") -> BeamProvider:
    """Build a beam provider from a spec: a ``.fits`` cube, a CSV/built-in JimBeam, or a band."""
    spec = band if spec in (None, "") else str(spec)
    if spec.lower().endswith(".fits"):
        return FitsBeamProvider.from_fits(spec)
    return _build_jimbeam(spec, band)


def averaged_power_beam(provider, ell, emm, freqs, chi_grid):
    """Frequency- and parallactic-angle-averaged Stokes-I power beam ``A(l, m)``, shape ``(npts,)``.

    Averages :func:`image_power_beam` (already PA-averaged) over frequency. Used for the
    image-/component-domain apply/correct, which is a direct multiply/divide by one map.
    """
    return image_power_beam(provider, True, ell, emm, freqs, chi_grid).mean(axis=1)


def write_beam_fits(beam: CosineTaperBeam, l_grid, m_grid, freqs_hz, path):
    """Write a cosine-taper beam to a 4-plane FITS cube that :meth:`FitsBeamProvider.from_fits` reads.

    Samples ``beam`` on the ``(l_grid, m_grid)`` direction-cosine grid at ``freqs_hz`` and stores
    ``[HH, VV]`` feed voltages as real/imag planes: ``(4, nfreq, nm, nl)``, with a linear WCS giving
    L/M in degrees and FREQ in Hz.
    """
    from astropy.io import fits

    l_grid = np.ascontiguousarray(l_grid, dtype=np.float64)
    m_grid = np.ascontiguousarray(m_grid, dtype=np.float64)
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    nl, nm, nf = l_grid.size, m_grid.size, freqs_hz.size
    # The FITS FREQ axis is linear (CRVAL + CDELT*pixel), so the frequencies must be
    # uniformly spaced. The cosine-taper model is continuous in frequency, so resample
    # onto a uniform grid rather than write a non-uniform one that reloads incorrectly.
    if nf > 2 and not np.allclose(np.diff(freqs_hz), freqs_hz[1] - freqs_hz[0]):
        raise ValueError("write_beam_fits needs uniformly-spaced frequencies for the linear FITS FREQ axis.")

    ll, mm = np.meshgrid(l_grid, m_grid, indexing="ij")
    v = beam.voltages(np.degrees(ll.ravel()), np.degrees(mm.ravel()), freqs_hz / 1e6)  # (nl*nm, nf, 2)
    v = v.reshape(nl, nm, nf, 2)
    hh = v[..., 0].transpose(2, 1, 0)  # (nf, nm, nl)
    vv = v[..., 1].transpose(2, 1, 0)
    data = np.stack([hh.real, hh.imag, vv.real, vv.imag], axis=0).astype(np.float32)  # (4, nf, nm, nl)

    dl = np.degrees(l_grid[1] - l_grid[0]) if nl > 1 else 1.0
    dm = np.degrees(m_grid[1] - m_grid[0]) if nm > 1 else 1.0
    df = float(freqs_hz[1] - freqs_hz[0]) if nf > 1 else 1e6
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CRPIX1"], hdr["CRVAL1"], hdr["CDELT1"], hdr["CUNIT1"] = "L", 1, np.degrees(l_grid[0]), dl, "deg"
    hdr["CTYPE2"], hdr["CRPIX2"], hdr["CRVAL2"], hdr["CDELT2"], hdr["CUNIT2"] = "M", 1, np.degrees(m_grid[0]), dm, "deg"
    hdr["CTYPE3"], hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CUNIT3"] = "FREQ", 1, float(freqs_hz[0]), df, "Hz"
    hdr["CTYPE4"], hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = "FEED", 1, 0, 1
    hdr["BUNIT"] = "1"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
