"""
Spectral models for FITS sky models.

A FITS sky model carries frequency information in one of three ways:

``flat``
    A single plane, taken to be frequency independent.
``poly``
    A flux at a reference frequency plus per-pixel log-polynomial coefficients,
    so that ``S(nu) = S(nu0) * (nu/nu0) ** (c1 + c2*x + c3*x**2 + ...)`` with
    ``x = ln(nu/nu0)``. This is the convention
    :func:`simms.skymodel.source_factory.contspec` uses for ASCII sources.
    The coefficients are either fitted from a spectral cube or read from
    spectral-index maps.
``cube``
    Explicit per-channel planes, resampled onto the MS channel grid.

The ``poly`` form is what makes a continuum cube cheap: the cube is read once,
in channel blocks, and only ``order + 1`` planes are kept rather than ``nchan``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from astropy.io import fits

from simms.exceptions import FITSSkymodelError

log = logging.getLogger(__name__)

# Channels read per streaming pass over the cube.
CHANNEL_BLOCK = 16

# A log-polynomial fit is accepted when the RMS residual in ln(flux) - which for
# small residuals is the fractional flux error - stays below this for every pixel
# bright enough to matter.
DEFAULT_FIT_TOLERANCE = 1e-3


class SpectralKind(str, Enum):
    FLAT = "flat"
    POLY = "poly"
    CUBE = "cube"


@dataclass
class FitsSpectrum:
    """How a FITS sky model varies with frequency."""

    kind: SpectralKind
    ref_freq: Optional[float] = None
    # POLY: exponent coefficients c1..c_order, one map per order
    coeffs: Optional[np.ndarray] = None  # (order, npix_l, npix_m)
    # Fractional flux residual of the fit, when it was fitted rather than supplied
    residual: Optional[float] = None

    @property
    def nchan_model(self) -> int:
        """Channels the model itself stores. CUBE stores the MS grid; the rest store one."""
        return 0 if self.kind is SpectralKind.CUBE else 1

    def scale(self, freqs: np.ndarray) -> np.ndarray | float:
        """
        Flux scaling relative to ``ref_freq``.

        Returns 1.0 for a flat spectrum, otherwise an array of shape
        ``(nchan,) + coeffs.shape[1:]``.
        """
        if self.kind is not SpectralKind.POLY:
            return 1.0
        return evaluate_scale(self.coeffs, freqs, self.ref_freq)

    def scale_at_pixels(self, freqs: np.ndarray, i_pix, j_pix) -> np.ndarray | float:
        """Flux scaling for a selection of pixels, shape ``(nchan, npix)``."""
        if self.kind is not SpectralKind.POLY:
            return 1.0
        return evaluate_scale(self.coeffs[:, i_pix, j_pix], freqs, self.ref_freq)


def evaluate_scale(coeffs: np.ndarray, freqs: np.ndarray, ref_freq: float) -> np.ndarray:
    """
    Evaluate ``(nu/nu0) ** (c1 + c2*x + ...)`` with ``x = ln(nu/nu0)``.

    Parameters
    ----------
    coeffs : numpy.ndarray
        Coefficients ``c1..c_order`` along the leading axis; any trailing shape.
    freqs : numpy.ndarray
        Frequencies (Hz).
    ref_freq : float
        Reference frequency (Hz).

    Returns
    -------
    numpy.ndarray
        Shape ``(freqs.size,) + coeffs.shape[1:]``.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    log_ratio = np.log(freqs / ref_freq)
    trailing = (1,) * (coeffs.ndim - 1)

    ln_scale = np.zeros((freqs.size,) + coeffs.shape[1:], dtype=np.float64)
    power = np.ones_like(log_ratio)
    for order in range(coeffs.shape[0]):
        power = power * log_ratio  # x ** (order + 1)
        ln_scale += power.reshape((-1,) + trailing) * coeffs[order]
    return np.exp(ln_scale)


def _design_matrix(freqs: np.ndarray, ref_freq: float, order: int) -> np.ndarray:
    """Columns ``[1, x, x**2, ..., x**order]`` for ``ln S = ln S0 + c1*x + c2*x**2 + ...``."""
    log_ratio = np.log(np.asarray(freqs, dtype=np.float64) / ref_freq)
    return np.vander(log_ratio, order + 1, increasing=True)


def iter_channel_blocks(cube, block: int = CHANNEL_BLOCK):
    """Yield ``(start, planes)`` over the trailing channel axis, computing lazily."""
    nchan = cube.shape[-1]
    for start in range(0, nchan, block):
        stop = min(start + block, nchan)
        yield start, np.asarray(cube[..., start:stop], dtype=np.float64)


def fit_log_polynomial(
    cube,
    fits_freqs: np.ndarray,
    ref_freq: float,
    stokes_i: int,
    order: int = 2,
    tol: float = 1e-7,
    block: int = CHANNEL_BLOCK,
):
    """
    Fit a per-pixel log-polynomial spectrum to a cube, streaming over channels.

    The cube is read once, ``block`` channels at a time. Only the normal equations
    are accumulated, so peak memory is ``order + 1`` planes rather than ``nchan``.
    The residual sum of squares falls out of the same accumulators, which is what
    lets a caller decide between ``poly`` and ``cube`` without a second pass. Because
    it is a difference of two close quantities, the reported residual has a floor
    around 1e-8; an exact fit reports that rather than zero.

    Parameters
    ----------
    cube : array_like
        Sliceable ``(nstokes, npix_l, npix_m, nchan_fits)``; may be lazy.
    fits_freqs : numpy.ndarray
        FITS channel centres (Hz).
    ref_freq : float
        Reference frequency for the fit (Hz).
    stokes_i : int
        Index of the Stokes I plane along the leading axis.
    order : int, optional
        Polynomial order. ``order=1`` is a plain spectral index. Default 2.
    tol : float, optional
        Pixels peaking below this are ignored when judging the residual.
    block : int, optional
        Channels per read.

    Returns
    -------
    tuple
        ``(ref_plane_i, coeffs, residual, peak)``: Stokes I at ``ref_freq`` with
        shape ``(npix_l, npix_m)``; ``coeffs`` of shape ``(order, npix_l, npix_m)``;
        the worst fractional residual over bright pixels; and the peak absolute
        brightness per pixel across all Stokes and channels.

    Raises
    ------
    FITSSkymodelError
        If the cube has too few channels to constrain the fit.
    """
    nstokes, npix_l, npix_m, nchan_fits = cube.shape
    nterms = order + 1
    if nchan_fits < nterms:
        raise FITSSkymodelError(
            f"A log-polynomial of order {order} needs at least {nterms} FITS channels, but the cube has {nchan_fits}."
        )

    npix = npix_l * npix_m
    design = _design_matrix(fits_freqs, ref_freq, order)

    normal = np.zeros((nterms, nterms))
    projection = np.zeros((nterms, npix))
    sum_sq = np.zeros(npix)
    positive = np.ones(npix, dtype=bool)
    peak = np.zeros(npix)

    for start, planes in iter_channel_blocks(cube, block):
        stop = start + planes.shape[-1]
        flat = planes.reshape(nstokes, npix, -1)
        # max|x| without materialising abs(x)
        np.maximum(peak, np.maximum(flat.max(axis=(0, 2)), -flat.min(axis=(0, 2))), out=peak)

        stokes_plane = flat[stokes_i]  # (npix, nchan_block)
        positive &= np.all(stokes_plane > 0, axis=1)
        # Invalid pixels are accumulated too, then discarded; clipping only keeps
        # the logarithm finite.
        log_flux = np.maximum(stokes_plane, np.finfo(np.float64).tiny)
        np.log(log_flux, out=log_flux)

        block_design = design[start:stop]
        normal += block_design.T @ block_design
        projection += block_design.T @ log_flux.T
        sum_sq += np.einsum("pc,pc->p", log_flux, log_flux)

    solution = np.linalg.solve(normal, projection)  # (nterms, npix)

    # Residual sum of squares straight from the accumulators. This is a difference of
    # two close quantities, so it bottoms out near sqrt(eps) ~ 1e-8 in units of
    # ln(flux) even for an exact fit. That floor sits far below any tolerance worth
    # accepting a fit at, so a second pass to compute it stably would buy nothing.
    residual_ss = np.maximum(sum_sq - np.einsum("tp,tp->p", solution, projection), 0.0)
    dof = max(nchan_fits - nterms, 1)
    rms = np.sqrt(residual_ss / dof)

    bright = positive & (peak > tol)
    residual = float(rms[bright].max()) if bright.any() else 0.0

    ref_plane = np.where(positive, np.exp(np.clip(solution[0], -700, 700)), 0.0)
    coeffs = np.where(positive[np.newaxis], solution[1:], 0.0)

    return (
        ref_plane.reshape(npix_l, npix_m),
        coeffs.reshape(order, npix_l, npix_m),
        residual,
        peak.reshape(npix_l, npix_m),
    )


def interpolate_planes_at(cube, fits_freqs: np.ndarray, ref_freq: float) -> np.ndarray:
    """
    Linearly interpolate every Stokes plane onto ``ref_freq``.

    Stokes Q, U and V change sign, so they cannot share the logarithmic fit. They
    are carried at the reference frequency and scaled by Stokes I's spectral shape.
    """
    upper = int(np.clip(np.searchsorted(fits_freqs, ref_freq), 1, fits_freqs.size - 1))
    lower = upper - 1
    freq_lo, freq_hi = fits_freqs[lower], fits_freqs[upper]
    planes = np.asarray(cube[..., lower : upper + 1], dtype=np.float64)

    if freq_hi == freq_lo:
        return planes[..., 0]
    weight = (ref_freq - freq_lo) / (freq_hi - freq_lo)
    return planes[..., 0] * (1 - weight) + planes[..., 1] * weight


def read_spi_maps(paths: List[str], shape: tuple) -> np.ndarray:
    """
    Read spectral-index (and higher-order) coefficient maps.

    Parameters
    ----------
    paths : list of str
        One FITS image per coefficient, ordered ``c1, c2, ...``.
    shape : tuple
        Expected ``(npix_l, npix_m)`` of the intensity map, in (RA, Dec) order.

    Returns
    -------
    numpy.ndarray
        ``(order, npix_l, npix_m)``.

    Raises
    ------
    FITSSkymodelError
        If a map's spatial shape does not match the intensity map.
    """
    coeffs = []
    for order, path in enumerate(paths, start=1):
        with fits.open(path) as hdul:
            data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float64))
        if data.ndim != 2:
            raise FITSSkymodelError(f"Spectral-index map '{path}' must be a 2-D image, but has shape {data.shape}.")
        # FITS is (dec, ra); the sky model is transposed to (ra, dec)
        data = data.T
        if data.shape != shape:
            raise FITSSkymodelError(
                f"Spectral-index map '{path}' has shape {data.shape[::-1]}, but the intensity map is {shape[::-1]}."
            )
        coeffs.append(data)
        log.debug("read spectral coefficient c%d from %s", order, path)
    return np.stack(coeffs)
