"""Tests for visibility prediction from WSClean component lists.

Two independent references are used. ``africanus.rime.wsclean_predict`` is the
canonical WSClean predictor and shares no code with simms' kernel; it validates
the spectral and Gaussian conventions. A hand-written brute-force RIME from the
component Ra/Dec validates the coordinate handling end to end.
"""

import numpy as np
import pytest
from africanus.model.wsclean import load
from africanus.rime.wsclean_predict import wsclean_predict

from simms.constants import C
from simms.exceptions import ASCIISkymodelError
from simms.skymodel.mstools import predict_block
from simms.skymodel.wsclean_skies import prepare_wsclean_sky
from simms.utilities import radec2lm

from . import InitTest

# bench-style phase centre: RA 15 deg = 1h, Dec -31 deg
RA0, DEC0 = np.deg2rad(15.0), np.deg2rad(-31.0)

HEADER = (
    "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
    "ReferenceFrequency, MajorAxis, MinorAxis, Orientation"
)


class InitThisTest(InitTest):
    def write_model(self, lines) -> str:
        path = self.random_named_file(suffix=".txt")
        body = "".join(f"{line}\n" for line in lines)
        with open(path, "w") as fh:
            fh.write(HEADER + "\n" + body)
        return path


@pytest.fixture
def params():
    return InitThisTest()


@pytest.fixture
def uvw():
    return np.random.default_rng(11).normal(0, 700, (80, 3))


# a spread of components near the phase centre (1h, -31d)
MODEL = [
    "p0,POINT,01:00:04.0,-31.02.00.0,1.0,[-0.7],true,1400000000,,,",
    "p1,POINT,00:59:52.0,-30.57.30.0,2.5,[-0.8,0.1],false,1400000000,,,",
    "p2,POINT,01:00:10.0,-31.05.00.0,0.7,[],true,1400000000,,,",
    "g0,GAUSSIAN,00:59:58.0,-30.59.00.0,3.0,[-0.6],true,1400000000,40.0,20.0,30.0",
    "g1,GAUSSIAN,01:00:06.0,-31.03.00.0,1.5,[-0.9,0.05],true,1400000000,25.0,25.0,0.0",
]


def africanus_reference(path, uvw, freqs):
    """Predict with the canonical wsclean_predict, sharing simms' lm computation."""
    cols = dict(load(path))
    nsrc = len(cols["I"])
    ncomp = max((len(np.atleast_1d(c)) for c in cols["SpectralIndex"]), default=0)
    coeffs = np.zeros((nsrc, max(ncomp, 1)))
    for i, c in enumerate(cols["SpectralIndex"]):
        c = np.atleast_1d(c)
        coeffs[i, : c.size] = c

    lm = np.zeros((nsrc, 2))
    for i in range(nsrc):
        lm[i] = radec2lm(RA0, DEC0, cols["Ra"][i], cols["Dec"][i])

    gauss = np.column_stack([cols["MajorAxis"], cols["MinorAxis"], cols["Orientation"]])
    return wsclean_predict(
        uvw,
        lm,
        np.asarray(cols["Type"]),
        np.asarray(cols["I"], float),
        coeffs,
        np.asarray(cols["LogarithmicSI"], bool),
        np.asarray(cols["ReferenceFrequency"], float),
        gauss,
        freqs,
    )[:, :, 0]


def brute_force(path, uvw, freqs):
    """Independent RIME from the component Ra/Dec, points only (no shapes)."""
    cols = dict(load(path))
    vis = np.zeros((uvw.shape[0], freqs.size), np.complex128)
    for i in range(len(cols["I"])):
        el, em = radec2lm(RA0, DEC0, cols["Ra"][i], cols["Dec"][i])
        en = np.sqrt(1 - el * el - em * em)
        coeffs = np.atleast_1d(cols["SpectralIndex"][i])
        nu0 = cols["ReferenceFrequency"][i]
        x = np.log(freqs / nu0) if coeffs.size else np.zeros_like(freqs)
        if not coeffs.size:
            spectrum = np.full(freqs.size, cols["I"][i])
        elif cols["LogarithmicSI"][i]:
            spectrum = cols["I"][i] * np.exp(sum(c * x ** (k + 1) for k, c in enumerate(coeffs)))
        else:
            r = freqs / nu0
            spectrum = cols["I"][i] + sum(c * (r - 1) ** (k + 1) for k, c in enumerate(coeffs))
        phase = np.exp(2j * np.pi * (uvw[:, 0:1] * el + uvw[:, 1:2] * em + uvw[:, 2:3] * (en - 1)) * freqs[None] / C)
        vis += spectrum[None, :] * phase
    return vis


@pytest.mark.parametrize("nchan", [1, 8])
def test_matches_africanus_wsclean_predict(params, uvw, nchan):
    """Points and Gaussians, log and ordinary spectra, against the canonical predictor."""
    freqs = 1.2e9 + np.arange(nchan) * 5e7
    path = params.write_model(MODEL)
    prepared = prepare_wsclean_sky(path, freqs, RA0, DEC0, ncorr=2)

    got = predict_block(prepared, uvw)
    expected = africanus_reference(path, uvw, freqs)

    assert np.abs(got[..., 0] - expected).max() / np.abs(expected).max() < 1e-6
    np.testing.assert_array_equal(got[..., 0], got[..., 1])  # XX == YY, Stokes I
    assert np.abs(uvw[:, 2]).max() > 100  # the w term is exercised


def test_points_match_brute_force_rime(params, uvw):
    """Coordinate handling, end to end from Ra/Dec, against an independent RIME."""
    freqs = np.array([1.3e9, 1.4e9, 1.6e9])
    points = [line for line in MODEL if "POINT" in line]
    path = params.write_model(points)
    got = predict_block(prepare_wsclean_sky(path, freqs, RA0, DEC0, ncorr=2), uvw)
    expected = brute_force(path, uvw, freqs)
    assert np.abs(got[..., 0] - expected).max() / np.abs(expected).max() < 1e-9


def test_four_correlations_fill_only_the_diagonal(params, uvw):
    freqs = np.array([1.4e9, 1.5e9])
    path = params.write_model(MODEL)
    vis = predict_block(prepare_wsclean_sky(path, freqs, RA0, DEC0, ncorr=4), uvw)
    np.testing.assert_array_equal(vis[..., 0], vis[..., 3])
    assert not np.any(vis[..., 1])
    assert not np.any(vis[..., 2])


def test_flat_source_without_a_spectral_index(params, uvw):
    """An empty SpectralIndex means a frequency-independent flux."""
    freqs = np.linspace(1.2e9, 1.7e9, 6)
    path = params.write_model(["p0,POINT,01:00:00.0,-31.00.00.0,2.0,[],true,1400000000,,,"])
    prepared = prepare_wsclean_sky(path, freqs, RA0, DEC0, ncorr=2)
    np.testing.assert_allclose(np.abs(prepared.bmat[0, 0, :]), 2.0)


def test_logarithmic_and_ordinary_conventions_differ(params, uvw):
    """A guard that the two spectral conventions are actually distinguished."""
    freqs = np.linspace(1.2e9, 1.7e9, 6)
    log_path = params.write_model(["s,POINT,01:00:00.0,-31.00.00.0,1.0,[-0.8,0.2],true,1400000000,,,"])
    ord_path = params.write_model(["s,POINT,01:00:00.0,-31.00.00.0,1.0,[-0.8,0.2],false,1400000000,,,"])
    log_spec = np.abs(prepare_wsclean_sky(log_path, freqs, RA0, DEC0).bmat[0, 0, :])
    ord_spec = np.abs(prepare_wsclean_sky(ord_path, freqs, RA0, DEC0).bmat[0, 0, :])

    nu0 = 1.4e9
    x = np.log(freqs / nu0)
    r = freqs / nu0
    np.testing.assert_allclose(log_spec, np.exp(-0.8 * x + 0.2 * x**2), rtol=1e-6)
    np.testing.assert_allclose(ord_spec, 1.0 - 0.8 * (r - 1) + 0.2 * (r - 1) ** 2, rtol=1e-6)
    assert not np.allclose(log_spec, ord_spec)


def test_gaussian_axes_are_the_fwhm(params, uvw):
    """A circular Gaussian's visibility must fall as the analytic FT of its FWHM."""
    freqs = np.array([1.4e9])
    fwhm_arcsec = 30.0
    path = params.write_model(
        [f"g,GAUSSIAN,01:00:00.0,-31.00.00.0,1.0,[],true,1400000000,{fwhm_arcsec},{fwhm_arcsec},0.0"]
    )
    prepared = prepare_wsclean_sky(path, freqs, RA0, DEC0, ncorr=2)

    # phase-centre source, so |vis| is the Gaussian envelope alone
    uvw_probe = np.array([[500.0, 0.0, 0.0], [0.0, 800.0, 0.0]])
    vis = predict_block(prepared, uvw_probe, out_dtype=np.complex128)[:, 0, 0]

    fwhm_rad = np.deg2rad(fwhm_arcsec / 3600)
    sigma = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))
    u_lambda = uvw_probe[:, :2] * freqs[0] / C
    expected = np.exp(-2 * np.pi**2 * sigma**2 * (u_lambda**2).sum(axis=1))
    np.testing.assert_allclose(np.abs(vis), expected, rtol=1e-6)


def test_empty_model_raises(params):
    path = params.write_model([])
    with pytest.raises(ASCIISkymodelError, match="no components"):
        prepare_wsclean_sky(path, np.array([1.4e9]), RA0, DEC0)


def test_unsupported_component_type_raises(params):
    path = params.write_model(["s,SHAPELET,01:00:00.0,-31.00.00.0,1.0,[],true,1400000000,,,"])
    with pytest.raises(ASCIISkymodelError, match="unsupported component type"):
        prepare_wsclean_sky(path, np.array([1.4e9]), RA0, DEC0)


def test_invalid_ncorr_raises(params):
    path = params.write_model(MODEL)
    with pytest.raises(ValueError, match="two or four correlations"):
        prepare_wsclean_sky(path, np.array([1.4e9]), RA0, DEC0, ncorr=3)
