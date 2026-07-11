"""
Visibility prediction from WSClean component lists.

A WSClean component list is a text catalogue of point and Gaussian components,
each with a Stokes I flux at a reference frequency and a polynomial spectrum. The
format is described at
https://sourceforge.net/p/wsclean/wiki/ComponentList/.

Parsing is delegated to ``africanus.model.wsclean.load``, which handles the
sexagesimal coordinates, the bracketed per-source spectral arrays, and the
``LogarithmicSI`` flag. The parsed components are flattened into the same
:class:`~simms.skymodel.mstools.PreparedSky` the ASCII path produces, so the
shared kernel predicts the visibilities. WSClean models carry Stokes I only, so
prediction is unpolarised: XX = YY = I and the cross-hands vanish.
"""

from __future__ import annotations

import logging

import numpy as np
from africanus.model.wsclean import load, spectra

from simms import BIN
from simms.constants import FWHM_TO_GAUSS_SCALE
from simms.exceptions import ASCIISkymodelError
from simms.skymodel.kernels import is_uniform_grid
from simms.skymodel.mstools import PreparedSky
from simms.utilities import radec2lm

log = logging.getLogger(__name__)

SUPPORTED_TYPES = ("POINT", "GAUSSIAN")


def _column(components: dict, name: str, nsrc: int, default=None):
    values = components.get(name)
    if values is None:
        if default is None:
            raise ASCIISkymodelError(f"WSClean model is missing the required '{name}' column.")
        return [default] * nsrc
    return values


def _spectral_coefficients(spectral_index: list, nsrc: int) -> np.ndarray:
    """Pad the ragged per-source coefficient lists into ``(nsrc, ncomp)``.

    Zero padding is safe for both spectral conventions: a zero log coefficient
    contributes a unit factor, and a zero ordinary coefficient adds nothing.
    """
    lengths = [len(np.atleast_1d(c)) for c in spectral_index]
    ncomp = max(lengths) if lengths else 0
    coeffs = np.zeros((nsrc, ncomp), dtype=np.float64)
    for row, values in enumerate(spectral_index):
        values = np.atleast_1d(values)
        coeffs[row, : values.size] = values
    return coeffs


def prepare_wsclean_sky(
    wsclean_file: str,
    freqs: np.ndarray,
    ra0: float,
    dec0: float,
    ncorr: int = 2,
    dtype: np.dtype = np.complex128,
) -> PreparedSky:
    """
    Flatten a WSClean component list into the arrays the prediction kernel consumes.

    Parameters
    ----------
    wsclean_file : str
        Path to a WSClean component list.
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    ra0, dec0 : float
        Phase centre (radians).
    ncorr : int, optional
        Number of correlations (2 or 4). Default 2.
    dtype : numpy.dtype, optional
        Complex dtype of the brightness matrix.

    Returns
    -------
    PreparedSky

    Raises
    ------
    ASCIISkymodelError
        If the model is empty, names an unsupported component type, or gives a
        spectral index without a usable reference frequency.
    ValueError
        If `ncorr` is not 2 or 4.
    """
    log = logging.getLogger(BIN.skysim)
    if ncorr not in (2, 4):
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    components = dict(load(wsclean_file))
    nsrc = len(components.get("I", []))
    if nsrc == 0:
        raise ASCIISkymodelError(f"WSClean model '{wsclean_file}' contains no components.")

    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    nchan = freqs.size

    source_type = np.asarray(_column(components, "Type", nsrc, default="POINT"))
    unsupported = sorted(set(source_type) - set(SUPPORTED_TYPES))
    if unsupported:
        raise ASCIISkymodelError(
            f"WSClean model has unsupported component type(s) {unsupported}. "
            f"Only {list(SUPPORTED_TYPES)} are supported."
        )

    ra = np.asarray(_column(components, "Ra", nsrc), dtype=np.float64)
    dec = np.asarray(_column(components, "Dec", nsrc), dtype=np.float64)
    stokes_i = np.asarray(_column(components, "I", nsrc), dtype=np.float64)

    coeffs = _spectral_coefficients(_column(components, "SpectralIndex", nsrc, default=[]), nsrc)
    log_poly = np.asarray(_column(components, "LogarithmicSI", nsrc, default=True), dtype=np.bool_)
    ref_freq = np.asarray(_column(components, "ReferenceFrequency", nsrc, default=0.0), dtype=np.float64)
    if coeffs.shape[1] and np.any((ref_freq <= 0) & np.any(coeffs != 0, axis=1)):
        raise ASCIISkymodelError(
            "A WSClean component has a spectral index but no positive ReferenceFrequency to anchor it."
        )

    major = np.asarray(_column(components, "MajorAxis", nsrc, default=0.0), dtype=np.float64)
    minor = np.asarray(_column(components, "MinorAxis", nsrc, default=0.0), dtype=np.float64)
    orientation = np.asarray(_column(components, "Orientation", nsrc, default=0.0), dtype=np.float64)

    # Direction cosines
    el, em = radec2lm(ra0, dec0, ra, dec)
    lmn = np.empty((nsrc, 3), dtype=np.float64)
    lmn[:, 0], lmn[:, 1] = el, em
    lmn[:, 2] = np.sqrt(np.maximum(1 - el * el - em * em, 0.0)) - 1

    # Gaussian shape, converting the FWHM axes into the kernel's parameterisation.
    is_gauss = source_type == "GAUSSIAN"
    gauss_shape = np.zeros((nsrc, 3), dtype=np.float64)
    axis_major = major * FWHM_TO_GAUSS_SCALE
    axis_minor = minor * FWHM_TO_GAUSS_SCALE
    safe_major = np.where(axis_major == 0.0, 1.0, axis_major)
    gauss_shape[:, 0] = axis_major * np.sin(orientation)
    gauss_shape[:, 1] = axis_major * np.cos(orientation)
    gauss_shape[:, 2] = axis_minor / safe_major

    # Stokes I spectrum per source; unpolarised, so one correlation carried.
    if coeffs.shape[1]:
        spectrum = spectra(stokes_i, coeffs, log_poly, ref_freq, freqs)
    else:
        spectrum = np.repeat(stokes_i[:, np.newaxis], nchan, axis=1)
    bmat = spectrum.astype(dtype)[:, np.newaxis, :]  # (nsrc, 1, nchan)

    ngauss = int(is_gauss.sum())
    log.info(f"Predicting {nsrc} WSClean components ({nsrc - ngauss} point, {ngauss} Gaussian).")

    return PreparedSky(
        lmn=lmn,
        gauss_shape=gauss_shape,
        is_gauss=is_gauss,
        bmat=bmat,
        lightcurve=np.ones((nsrc, 1), dtype=np.float64),
        unique_times=None,
        freqs=freqs,
        uniform_freqs=is_uniform_grid(freqs),
        ncorr=ncorr,
        polarisation=False,
    )
