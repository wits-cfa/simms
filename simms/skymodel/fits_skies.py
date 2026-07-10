"""
Visibility prediction from FITS sky models.

Conventions
-----------
Everything here predicts the same quantity, verified in ``tests/predict_fits_tests.py``
against a brute-force transcription of the RIME::

    V(u, v, w, nu) = sum_pix S_pix * exp(2j*pi * (u*l + v*m + w*(n - 1)) * nu / c)

where ``(l, m)`` are the direction cosines of a pixel *relative to the MS phase
centre*, ``n = sqrt(1 - l**2 - m**2)``, and there is no ``1/n`` factor because the
image is in Jy/pixel rather than surface brightness.

``ducc0.wgridder.dirty2vis`` reproduces this exactly when called with
``divide_by_n=False`` and the pixel-to-direction-cosine mapping
``l = -(i - npix//2) * pixsize_x``, ``m = -(j - npix//2) * pixsize_y``. It therefore
requires a *regular* ``(l, m)`` grid, which only an orthographic (SIN) projection
whose tangent point coincides with the phase centre provides. Any other projection,
or a tangent point away from the phase centre, is reprojected onto such a grid first.

The DFT backend needs no regular grid and is exact for every projection, so sparse
component-like images are predicted without resampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from astropy import units
from astropy.wcs import WCS
from ducc0.wgridder import dirty2vis
from fitstoolz.reader import FitsData
from scabha.basetypes import File
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from simms import BIN
from simms.exceptions import FITSSkymodelError
from simms.skymodel.kernels import is_uniform_grid, predict_vis
from simms.skymodel.mstools import add_noise, stack_unpolarised_vis
from simms.utilities import is_range_in_range, radec2lm

log = logging.getLogger(__name__)

# A pixel grid is treated as regular in (l, m) - and so usable by the gridder -
# when the direction cosines deviate from a linear function of pixel index by
# less than this, in pixels. SIN at the phase centre gives exactly zero.
MAX_GRID_DEVIATION_PIXELS = 1e-2

# Cost-model constants, measured at 1 thread with nchan=16, npix in {1024, 2048}
# and nrow in {2e3, 2e4, 1e5}. The gridder's degridding term grows with the image
# side (wider kernel support, more w planes), hence the per-side factor. These only
# choose between two correct backends, so an imprecise fit costs time, never
# accuracy. Break-even lands at a few hundred components.
DFT_SECONDS_PER_COMPONENT_ROW_CHAN = 6.4e-9
FFT_SECONDS_PER_PIXEL_LOGPIXEL_CHAN = 2.5e-9
FFT_SECONDS_PER_ROW_CHAN_PER_SIDE = 7.0e-10

STOKES_CODES = {1: "I", 2: "Q", 3: "U", 4: "V"}


@dataclass
class FitsGrid:
    """Mapping from image pixel indices to direction cosines, for ``dirty2vis``.

    ``l = l_ref + delta_l * (i - ref_l)`` and likewise for ``m``.
    """

    delta_l: float  # signed, radians per pixel
    delta_m: float
    ref_l: float  # 0-based pixel index of the reference point
    ref_m: float
    l_ref: float  # direction cosine at the reference pixel
    m_ref: float
    deviation_pixels: float

    @property
    def is_regular(self) -> bool:
        return self.deviation_pixels < MAX_GRID_DEVIATION_PIXELS

    def ducc_kwargs(self, npix_l: int, npix_m: int) -> dict:
        """Arguments placing this grid in the frame ``dirty2vis`` assumes."""
        centre_l, centre_m = npix_l // 2, npix_m // 2
        return dict(
            pixsize_x=abs(self.delta_l),
            pixsize_y=abs(self.delta_m),
            flip_u=bool(self.delta_l > 0),
            flip_v=bool(self.delta_m > 0),
            center_x=-(self.l_ref + self.delta_l * (centre_l - self.ref_l)),
            center_y=-(self.m_ref + self.delta_m * (centre_m - self.ref_m)),
            divide_by_n=False,
        )


@dataclass
class PreparedFitsSky:
    """A FITS sky model reduced to what a prediction backend needs."""

    chan_freqs: np.ndarray  # MS channel centres (Hz)
    flat_spectrum: bool  # True when the model has a single, frequency-independent plane
    backend: str  # "dft" or "fft"
    ncorr: int
    polarisation: bool
    linear_basis: bool
    ncomp: int
    npix_l: int
    npix_m: int

    # FFT backend: real Stokes planes, (nstokes, npix_l, npix_m, nchan_model)
    planes: Optional[np.ndarray] = None
    stokes_names: Optional[List[str]] = None
    grid: Optional[FitsGrid] = None

    # DFT backend: brightness per correlation, (ncomp, nspec, nchan)
    lmn: Optional[np.ndarray] = None
    bmat: Optional[np.ndarray] = None
    uniform_freqs: bool = True

    @property
    def nspec(self) -> int:
        return self.ncorr if self.polarisation else 1


# --------------------------------------------------------------------------- geometry


def pixel_lm(cel: WCS, ra0: float, dec0: float, i_pix, j_pix):
    """
    Direction cosines of image pixels relative to the phase centre.

    Exact for every projection: the pixel centres are pushed through the WCS and
    then converted with the same ``radec2lm`` the ASCII path uses.

    Parameters
    ----------
    cel : astropy.wcs.WCS
        Celestial sub-WCS of the image.
    ra0, dec0 : float
        Phase centre (radians).
    i_pix, j_pix : array_like
        0-based pixel indices along the longitude and latitude axes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(l, m)``, each of shape ``i_pix.shape``.
    """
    lng, lat = cel.wcs.lng, cel.wcs.lat
    i_pix = np.atleast_1d(np.asarray(i_pix, dtype=np.float64))
    j_pix = np.atleast_1d(np.asarray(j_pix, dtype=np.float64))

    pixels = np.empty((i_pix.size, 2))
    pixels[:, lng] = i_pix.ravel()
    pixels[:, lat] = j_pix.ravel()
    world = cel.wcs_pix2world(pixels, 0)

    to_rad_lng = units.Unit(cel.wcs.cunit[lng]).to("rad")
    to_rad_lat = units.Unit(cel.wcs.cunit[lat]).to("rad")
    ra = world[:, lng] * to_rad_lng
    dec = world[:, lat] * to_rad_lat

    el, em = radec2lm(ra0, dec0, ra, dec)
    return el.reshape(i_pix.shape), em.reshape(j_pix.shape)


def lm_to_radec(el: np.ndarray, em: np.ndarray, ra0: float, dec0: float):
    """Inverse of :func:`simms.utilities.radec2lm`, in radians."""
    en = np.sqrt(np.maximum(1.0 - el * el - em * em, 0.0))
    dec = np.arcsin(em * np.cos(dec0) + en * np.sin(dec0))
    ra = ra0 + np.arctan2(el, en * np.cos(dec0) - em * np.sin(dec0))
    return ra, dec


def fit_lm_grid(cel: WCS, ra0: float, dec0: float, npix_l: int, npix_m: int, samples: int = 9) -> FitsGrid:
    """
    Fit ``l`` and ``m`` as linear functions of pixel index, and measure the misfit.

    The misfit is zero for a SIN projection tangent at the phase centre and grows
    with field size for any other projection, so it is the criterion for whether
    the gridder may be used at all.
    """
    ref_l, ref_m = npix_l // 2, npix_m // 2
    grid_l = np.unique(np.linspace(0, npix_l - 1, min(samples, npix_l)).astype(int))
    grid_m = np.unique(np.linspace(0, npix_m - 1, min(samples, npix_m)).astype(int))
    i_pix, j_pix = (a.ravel() for a in np.meshgrid(grid_l, grid_m, indexing="ij"))

    el, em = pixel_lm(cel, ra0, dec0, i_pix, j_pix)

    design_l = np.column_stack([np.ones(i_pix.size), i_pix - ref_l])
    design_m = np.column_stack([np.ones(j_pix.size), j_pix - ref_m])
    (l_ref, delta_l), *_ = np.linalg.lstsq(design_l, el, rcond=None)
    (m_ref, delta_m), *_ = np.linalg.lstsq(design_m, em, rcond=None)

    residual = max(
        np.abs(el - design_l @ [l_ref, delta_l]).max(),
        np.abs(em - design_m @ [m_ref, delta_m]).max(),
    )
    cell = min(abs(delta_l), abs(delta_m))
    return FitsGrid(
        delta_l=float(delta_l),
        delta_m=float(delta_m),
        ref_l=float(ref_l),
        ref_m=float(ref_m),
        l_ref=float(l_ref),
        m_ref=float(m_ref),
        deviation_pixels=float(residual / cell) if cell else np.inf,
    )


def reproject_to_sin(planes: np.ndarray, cel: WCS, ra0: float, dec0: float, cell: float, order: int = 3):
    """
    Resample image planes onto a SIN grid tangent at the phase centre.

    Flux is conserved exactly (to interpolation accuracy) without any solid-angle
    bookkeeping: Jy/pixel is a density with respect to pixel index, so scaling the
    interpolated value by the Jacobian determinant of the target-to-source pixel
    map accounts for the change of pixel area.

    Parameters
    ----------
    planes : numpy.ndarray
        Image planes of shape ``(..., npix_l, npix_m)``, in Jy/pixel.
    cel : astropy.wcs.WCS
        Celestial sub-WCS of the source image.
    ra0, dec0 : float
        Phase centre (radians).
    cell : float
        Pixel size of the target grid, in radians.
    order : int, optional
        Spline order for :func:`scipy.ndimage.map_coordinates`. Default 3.

    Returns
    -------
    tuple
        ``(resampled_planes, grid)`` where ``grid`` is the exactly regular
        :class:`FitsGrid` of the new image.
    """
    lng, lat = cel.wcs.lng, cel.wcs.lat
    npix_l, npix_m = planes.shape[-2], planes.shape[-1]

    # Angular extent of the source, sampled on its border.
    edge = np.linspace(0, npix_l - 1, 64).astype(int)
    edge_m = np.linspace(0, npix_m - 1, 64).astype(int)
    border_i = np.concatenate([edge, edge, np.zeros_like(edge_m), np.full_like(edge_m, npix_l - 1)])
    border_j = np.concatenate([np.zeros_like(edge), np.full_like(edge, npix_m - 1), edge_m, edge_m])
    el, em = pixel_lm(cel, ra0, dec0, border_i, border_j)

    half_l = int(np.ceil(max(abs(el).max(), cell) / cell)) + 2
    half_m = int(np.ceil(max(abs(em).max(), cell) / cell)) + 2
    out_l, out_m = 2 * half_l, 2 * half_m

    # Target grid: l decreases with index (as a negative CDELT1 does), m increases.
    delta_l, delta_m = -cell, cell
    ii = (np.arange(out_l) - half_l)[:, None]
    jj = (np.arange(out_m) - half_m)[None, :]
    target_l = np.broadcast_to(delta_l * ii, (out_l, out_m))
    target_m = np.broadcast_to(delta_m * jj, (out_l, out_m))

    ra, dec = lm_to_radec(target_l, target_m, ra0, dec0)
    world = np.empty((ra.size, 2))
    world[:, lng] = np.rad2deg(ra.ravel()) * units.deg.to(cel.wcs.cunit[lng])
    world[:, lat] = np.rad2deg(dec.ravel()) * units.deg.to(cel.wcs.cunit[lat])
    source = cel.wcs_world2pix(world, 0)
    src_i = source[:, lng].reshape(out_l, out_m)
    src_j = source[:, lat].reshape(out_l, out_m)

    # |det J| of the target-pixel -> source-pixel map, by central differences.
    di_dl, di_dm = np.gradient(src_i)
    dj_dl, dj_dm = np.gradient(src_j)
    jacobian = np.abs(di_dl * dj_dm - di_dm * dj_dl)

    coords = np.stack([src_i.ravel(), src_j.ravel()])
    flat = planes.reshape(-1, npix_l, npix_m)
    out = np.empty((flat.shape[0], out_l, out_m), dtype=np.float64)
    for k, plane in enumerate(flat):
        resampled = map_coordinates(plane, coords, order=order, mode="constant", cval=0.0)
        out[k] = resampled.reshape(out_l, out_m) * jacobian

    grid = FitsGrid(
        delta_l=delta_l,
        delta_m=delta_m,
        ref_l=float(half_l),
        ref_m=float(half_m),
        l_ref=0.0,
        m_ref=0.0,
        deviation_pixels=0.0,
    )
    return out.reshape(planes.shape[:-2] + (out_l, out_m)), grid


# --------------------------------------------------------------------------- polarisation


def stokes_to_correlations(get, ncorr: int, polarisation: bool, linear_basis: bool) -> list:
    """
    Combine Stokes quantities into correlations.

    ``get(name)`` returns the I, Q, U or V quantity, or 0 when the model does not
    carry it. The combination is linear, so it may be applied either to image
    planes or, equivalently, to the visibilities they predict.
    """
    if ncorr not in (2, 4):
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    stokes_i = get("I")
    if not polarisation:
        zero = np.zeros_like(stokes_i)
        return [stokes_i, stokes_i] if ncorr == 2 else [stokes_i, zero, zero, stokes_i]

    stokes_q, stokes_u, stokes_v = get("Q"), get("U"), get("V")
    if linear_basis:
        corrs = [stokes_i + stokes_q, stokes_u + 1j * stokes_v, stokes_u - 1j * stokes_v, stokes_i - stokes_q]
    else:
        corrs = [stokes_i + stokes_v, stokes_q + 1j * stokes_u, stokes_q - 1j * stokes_u, stokes_i - stokes_v]

    return [corrs[0], corrs[3]] if ncorr == 2 else corrs


def _stokes_getter(planes: np.ndarray, names: List[str]):
    index = {name: k for k, name in enumerate(names)}

    def get(name):
        if name in index:
            return planes[index[name]]
        return np.zeros_like(planes[0])

    return get


# --------------------------------------------------------------------------- backend choice


def choose_backend(ncomp: int, npix_l: int, npix_m: int, nrow: int, nchan: int, nplanes: int) -> str:
    """
    Pick the cheaper of the two exact backends.

    The DFT costs one phasor per component, row and channel. The gridder costs an
    image FFT per plane and channel, plus a degridding term proportional to rows.
    Break-even is therefore a few hundred components, not a fraction of the image:
    a "sparse" CLEAN model with thousands of components still wants the gridder.
    """
    npix = npix_l * npix_m
    side = max(npix_l, npix_m, 2)
    dft = DFT_SECONDS_PER_COMPONENT_ROW_CHAN * ncomp * nrow * nchan
    fft = (
        nplanes
        * nchan
        * (FFT_SECONDS_PER_PIXEL_LOGPIXEL_CHAN * npix * np.log2(side) + FFT_SECONDS_PER_ROW_CHAN_PER_SIDE * side * nrow)
    )
    return "dft" if dft < fft else "fft"


# --------------------------------------------------------------------------- reading


def _ensure_axis(fds: FitsData, name: str, crval: float, cdelt: float, cunit: str):
    if name not in fds.coord_names:
        fds.add_axis(name, fds.ndim + 1, crval=crval, cdelt=cdelt, crpix=0, cunit=cunit)


def _fits_frequencies(fds: FitsData):
    """Channel centres and width of the FITS spectral axis, in Hz."""
    if fds.spectral_coord == "VRAD":
        freqs = fds.get_freq_from_vrad()
        step = fds.coords["VRAD"].pixel_size * getattr(units, fds.coords["VRAD"].units)
        width = step.to(units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="radio").value
    elif fds.spectral_coord == "VOPT":
        freqs = fds.get_freq_from_vopt()
        step = fds.coords["VOPT"].pixel_size * getattr(units, fds.coords["VOPT"].units)
        width = step.to(units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="optical").value
    else:
        freqs = np.squeeze(np.asarray(fds.coords["FREQ"].data))
        width = fds.coords["FREQ"].pixel_size
    return np.atleast_1d(freqs).astype(np.float64), float(width)


def _beam_axis_radians(column) -> np.ndarray:
    """Beam axis in radians. FITS beam keywords are degrees unless a unit says otherwise."""
    unit = getattr(column, "unit", None) or units.deg
    return np.atleast_1d((np.asarray(column) * unit).to("rad").value)


def _jy_per_pixel(cube: np.ndarray, fds: FitsData, pixel_area: float) -> np.ndarray:
    """Convert the image to Jy/pixel, using the beam table when BUNIT is Jy/beam."""
    if fds.data_units == "Jy/beam":
        if fds.beam_table is None:
            log.warning("FITS sky model is in Jy/beam but carries no beam information. Assuming Jy/pixel.")
            return cube
        bmaj = _beam_axis_radians(fds.beam_table["BMAJ"])
        bmin = _beam_axis_radians(fds.beam_table["BMIN"])
        beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2))
        pixels_per_beam = beam_area / pixel_area
        return cube / pixels_per_beam[np.newaxis, np.newaxis, np.newaxis, :]

    if fds.data_units == "":
        log.warning("FITS sky model has no BUNIT. Assuming Jy/pixel.")
    elif fds.data_units not in ("Jy", "Jy/pixel"):
        log.warning(f"FITS sky model has unknown BUNIT='{fds.data_units}'. Assuming Jy/pixel.")
    return cube


def _interpolate_spectrum(cube, fits_freqs, chan_freqs, method):
    """Resample the spectral axis of ``(nstokes, l, m, nchan_fits)`` onto the MS grid."""
    if fits_freqs.size == len(chan_freqs) and np.allclose(fits_freqs, chan_freqs):
        return cube
    log.warning("Interpolating the FITS sky model onto the MS frequency grid.")
    interp = interp1d(fits_freqs, cube, kind=method, axis=-1, bounds_error=False, fill_value="extrapolate")
    return interp(chan_freqs)


# --------------------------------------------------------------------------- preparation


def prepare_fits_sky(
    input_fitsimages: Union[File, str, List[File]],
    ra0: float,
    dec0: float,
    chan_freqs: np.ndarray,
    ms_delta_nu: float,
    ncorr: int,
    nrow: int,
    linear_basis: bool = True,
    polarisation: bool = True,
    tol: float = 1e-7,
    backend: str = "auto",
    interpolation: str = "nearest",
    stack_axis: str = "STOKES",
    reproject_order: int = 3,
) -> PreparedFitsSky:
    """
    Read one or more FITS images into a form a prediction backend can consume.

    Parameters
    ----------
    input_fitsimages : File or list of File
        A single FITS image, or an ordered list of per-Stokes images.
    ra0, dec0 : float
        MS phase centre (radians).
    chan_freqs : numpy.ndarray
        MS channel centres (Hz).
    ms_delta_nu : float
        MS channel width (Hz).
    ncorr : int
        Number of correlations to predict (2 or 4).
    nrow : int
        Total rows in the MS. Used only by the backend cost model.
    linear_basis : bool, optional
        Linear (XX/XY/YX/YY) or circular (RR/RL/LR/LL) correlations.
    polarisation : bool, optional
        Predict every correlation. Forced off when the model carries only Stokes I.
    tol : float, optional
        Pixels below this brightness (Jy) are dropped from the component list.
    backend : {"auto", "dft", "fft"}, optional
        Prediction backend. "auto" uses :func:`choose_backend`.
    interpolation : str, optional
        Spectral interpolation method when the FITS and MS channel grids differ.
    stack_axis : str, optional
        Axis along which a list of FITS images is stacked.
    reproject_order : int, optional
        Spline order used when reprojecting a non-SIN image.

    Returns
    -------
    PreparedFitsSky

    Raises
    ------
    FITSSkymodelError
        If the MS band lies outside the FITS band, or an unsupported Stokes axis
        or backend is requested.
    """
    log = logging.getLogger(BIN.skysim)
    chan_freqs = np.ascontiguousarray(chan_freqs, dtype=np.float64)
    nchan = chan_freqs.size

    files = [input_fitsimages] if isinstance(input_fitsimages, (str, File)) else list(input_fitsimages)

    ms_start_freq = chan_freqs[0] - 0.5 * ms_delta_nu
    ms_end_freq = chan_freqs[-1] + 0.5 * ms_delta_nu

    fds = FitsData(files[0])
    # A flat image is a frequency-independent sky; give it one channel spanning the band.
    _ensure_axis(fds, "FREQ", crval=0.5 * (ms_start_freq + ms_end_freq), cdelt=ms_end_freq - ms_start_freq, cunit="Hz")
    _ensure_axis(fds, stack_axis, crval=1, cdelt=1, cunit="")
    if len(files) > 1:
        fds.expand_along_axis_from_files(stack_axis, files[1:])

    stokes_codes = np.atleast_1d(np.squeeze(np.asarray(fds.coords["STOKES"].data))).astype(int)
    unknown = [int(c) for c in stokes_codes if int(c) not in STOKES_CODES]
    if unknown:
        raise FITSSkymodelError(f"Unsupported FITS STOKES codes {unknown}. Only 1..4 (I, Q, U, V) are supported.")
    stokes_names = [STOKES_CODES[int(c)] for c in stokes_codes]
    if "I" not in stokes_names:
        raise FITSSkymodelError(f"FITS sky model has no Stokes I plane (found {stokes_names}).")

    fits_freqs, fits_delta_nu = _fits_frequencies(fds)
    nchan_fits = fits_freqs.size
    fits_range = (fits_freqs[0] - 0.5 * fits_delta_nu, fits_freqs[-1] + 0.5 * fits_delta_nu)
    ms_range = (ms_start_freq, ms_end_freq)
    if nchan_fits > 1 and not is_range_in_range(ms_range, fits_range):
        raise FITSSkymodelError(
            f"MS frequencies [{ms_range[0] / 1e9:.6f} GHz, {ms_range[1] / 1e9:.6f} GHz] are outside the "
            f"FITS image frequencies [{fits_range[0] / 1e9:.6f} GHz, {fits_range[1] / 1e9:.6f} GHz]. "
            "Cannot interpolate the FITS image onto the MS frequency grid."
        )

    cel = fds.wcs.celestial
    npix_l = fds.coords["RA"].size
    npix_m = fds.coords["DEC"].size

    # One evaluation of the lazy cube, not one per correlation.
    cube = np.asarray(fds.get_xds(transpose=["STOKES", "RA", "DEC", "FREQ"]).data.compute(), dtype=np.float64)

    grid = fit_lm_grid(cel, ra0, dec0, npix_l, npix_m)
    pixel_area = abs(grid.delta_l * grid.delta_m)
    cube = _jy_per_pixel(cube, fds, pixel_area)
    fds.close()

    flat_spectrum = nchan_fits == 1
    if not flat_spectrum:
        cube = _interpolate_spectrum(cube, fits_freqs, chan_freqs, interpolation)

    polarisation = bool(polarisation) and len(stokes_names) > 1

    support = np.abs(cube).max(axis=(0, 3)) > tol
    ncomp = int(support.sum())
    if ncomp == 0:
        log.warning(f"No FITS pixel exceeds the {tol * 1e6:.2f}-uJy tolerance; the model is empty.")

    nplanes = len(stokes_names) if polarisation else 1
    if backend == "auto":
        backend = choose_backend(ncomp, npix_l, npix_m, nrow, nchan, nplanes)
    elif backend not in ("dft", "fft"):
        raise FITSSkymodelError(f"Unknown predict backend '{backend}'. Choose from auto, dft, fft.")

    if backend == "fft" and (npix_l % 2 or npix_m % 2):
        log.warning(f"Image is {npix_l}x{npix_m}; the gridder needs even dimensions. Using the DFT instead.")
        backend = "dft"

    prepared = PreparedFitsSky(
        chan_freqs=chan_freqs,
        flat_spectrum=flat_spectrum,
        backend=backend,
        ncorr=ncorr,
        polarisation=polarisation,
        linear_basis=linear_basis,
        ncomp=ncomp,
        npix_l=npix_l,
        npix_m=npix_m,
    )

    if backend == "dft":
        log.info(f"Predicting from {ncomp} FITS pixels with the DFT (image is {npix_l}x{npix_m}).")
        i_pix, j_pix = np.nonzero(support)
        el, em = pixel_lm(cel, ra0, dec0, i_pix, j_pix)
        lmn = np.empty((ncomp, 3))
        lmn[:, 0], lmn[:, 1] = el, em
        lmn[:, 2] = np.sqrt(np.maximum(1 - el * el - em * em, 0.0)) - 1

        # (nstokes, ncomp, nchan_model) -> correlations -> (ncomp, nspec, nchan)
        comps = cube[:, i_pix, j_pix, :]
        corrs = stokes_to_correlations(_stokes_getter(comps, stokes_names), ncorr, polarisation, linear_basis)
        nspec = prepared.nspec
        bmat = np.zeros((ncomp, nspec, nchan), dtype=np.complex128)
        for c in range(nspec):
            bmat[:, c, :] = corrs[c] if not flat_spectrum else corrs[c][:, 0:1]
        prepared.lmn = lmn
        prepared.bmat = bmat
        prepared.uniform_freqs = is_uniform_grid(chan_freqs)
    else:
        if not grid.is_regular:
            log.info(
                f"Image (l, m) grid departs from a regular grid by {grid.deviation_pixels:.3f} pixels; "
                "reprojecting onto a SIN grid tangent at the phase centre."
            )
            cell = min(abs(grid.delta_l), abs(grid.delta_m))
            cube = np.moveaxis(cube, (1, 2), (-2, -1))  # (nstokes, nchan, l, m)
            cube, grid = reproject_to_sin(cube, cel, ra0, dec0, cell, order=reproject_order)
            cube = np.moveaxis(cube, (-2, -1), (1, 2))
            npix_l, npix_m = cube.shape[1], cube.shape[2]
            prepared.npix_l, prepared.npix_m = npix_l, npix_m
        log.info(f"Predicting from a {npix_l}x{npix_m} FITS image with the gridder.")
        keep = stokes_names if polarisation else ["I"]
        index = [stokes_names.index(name) for name in keep]
        prepared.planes = np.ascontiguousarray(cube[index])
        prepared.stokes_names = keep
        prepared.grid = grid

    return prepared


# --------------------------------------------------------------------------- prediction


def _fft_stokes_visibilities(plane, grid, uvw, chan_freqs, flat_spectrum, npix_l, npix_m, epsilon, wgridding, nthreads):
    """Visibilities of one real Stokes plane stack, shape ``(npix_l, npix_m, nchan_model)``."""
    nrow, nchan = uvw.shape[0], chan_freqs.size
    vis = np.zeros((nrow, nchan), dtype=np.complex128)
    kwargs = dict(
        uvw=uvw, epsilon=epsilon, do_wgridding=wgridding, nthreads=nthreads, **grid.ducc_kwargs(npix_l, npix_m)
    )

    if flat_spectrum:
        image = np.ascontiguousarray(plane[:, :, 0])
        if np.any(image):
            vis[:] = dirty2vis(freq=chan_freqs, dirty=image, **kwargs)
        return vis

    for chan in range(nchan):
        image = np.ascontiguousarray(plane[:, :, chan])
        # A channel with no emission needs no gridding pass at all.
        if not np.any(image):
            continue
        vis[:, chan] = dirty2vis(freq=chan_freqs[chan : chan + 1], dirty=image, **kwargs)[:, 0]
    return vis


def predict_fits_block(
    prepared: PreparedFitsSky,
    uvw: np.ndarray,
    noise_vis: float | None = None,
    out_dtype: np.dtype = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    nthreads: int = 1,
) -> np.ndarray:
    """
    Predict visibilities for one block of rows.

    Parameters
    ----------
    prepared : PreparedFitsSky
        Sky model from :func:`prepare_fits_sky`.
    uvw : numpy.ndarray
        UVW coordinates of shape ``(nrow, 3)``, in metres.
    noise_vis : float, optional
        RMS noise per visibility (Jy).
    out_dtype : numpy.dtype, optional
        Complex dtype to cast to. Named ``out_dtype`` because ``da.blockwise``
        consumes any ``dtype`` kwarg itself and never forwards it.
    epsilon : float, optional
        Gridder accuracy.
    do_wgridding : bool, optional
        Apply the w term with w-gridding.
    nthreads : int, optional
        Threads for the gridder. Default 1; dask supplies row parallelism.

    Returns
    -------
    numpy.ndarray
        Visibilities of shape ``(nrow, nchan, ncorr)``.
    """
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    nrow, nchan, ncorr = uvw.shape[0], prepared.chan_freqs.size, prepared.ncorr

    if prepared.backend == "dft":
        vis = np.zeros((nrow, nchan, prepared.nspec), dtype=np.complex128)
        if prepared.ncomp:
            ncomp = prepared.ncomp
            predict_vis(
                uvw,
                prepared.chan_freqs,
                prepared.uniform_freqs,
                prepared.lmn,
                np.zeros((ncomp, 3)),
                np.zeros(ncomp, dtype=np.bool_),
                prepared.bmat,
                np.ones((ncomp, 1)),
                np.zeros(nrow, dtype=np.int64),
                vis,
            )
        if prepared.nspec != ncorr:
            vis = stack_unpolarised_vis(vis[..., 0], ncorr)
    else:
        stokes_vis = np.stack(
            [
                _fft_stokes_visibilities(
                    plane,
                    prepared.grid,
                    uvw,
                    prepared.chan_freqs,
                    prepared.flat_spectrum,
                    prepared.npix_l,
                    prepared.npix_m,
                    epsilon,
                    do_wgridding,
                    nthreads,
                )
                for plane in prepared.planes
            ]
        )
        corrs = stokes_to_correlations(
            _stokes_getter(stokes_vis, prepared.stokes_names),
            ncorr,
            prepared.polarisation,
            prepared.linear_basis,
        )
        vis = np.stack(corrs, axis=-1)

    if noise_vis:
        vis = add_noise(vis, noise_vis)
    if out_dtype is not None:
        vis = vis.astype(out_dtype, copy=False)
    return vis
