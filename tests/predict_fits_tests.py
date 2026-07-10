"""Tests for visibility prediction from FITS sky models.

The reference is a direct transcription of the RIME, evaluated with an explicit
loop over pixels, using astropy's WCS for the pixel coordinates. It shares no
code with either prediction backend.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from simms.constants import C
from simms.exceptions import FITSSkymodelError
from simms.skymodel.fits_skies import (
    choose_backend,
    fit_lm_grid,
    predict_fits_block,
    prepare_fits_sky,
    reproject_to_sin,
)
from simms.skymodel.fits_spectrum import SpectralKind

from . import InitTest

RA0_DEG, DEC0_DEG = 15.0, -31.0
RA0, DEC0 = np.deg2rad(RA0_DEG), np.deg2rad(DEC0_DEG)
CELL = 20.0 / 3600  # degrees


class InitThisTest(InitTest):
    pass


@pytest.fixture
def params():
    return InitThisTest()


@pytest.fixture
def uvw():
    """Long enough baselines that the fringe winds many times across the band."""
    return np.random.default_rng(11).normal(0, 800, (120, 3))


def make_header(
    npix,
    nchan=1,
    nstokes=1,
    proj="SIN",
    crval1=RA0_DEG,
    crval2=DEC0_DEG,
    cell=CELL,
    freqs=None,
    crpix_off=(1, 1),
    bunit="Jy/pixel",
):
    header = fits.Header()
    header["CTYPE1"], header["CRVAL1"] = f"RA---{proj}", crval1
    header["CDELT1"], header["CRPIX1"], header["CUNIT1"] = -cell, npix // 2 + crpix_off[0], "deg"
    header["CTYPE2"], header["CRVAL2"] = f"DEC--{proj}", crval2
    header["CDELT2"], header["CRPIX2"], header["CUNIT2"] = cell, npix // 2 + crpix_off[1], "deg"
    if nchan > 1 or freqs is not None:
        freqs = np.asarray(freqs)
        header["CTYPE3"], header["CRVAL3"] = "FREQ", float(freqs[0])
        header["CDELT3"] = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1e8
        header["CRPIX3"], header["CUNIT3"] = 1, "Hz"
    if nstokes > 1:
        idx = 4 if (nchan > 1 or freqs is not None) else 3
        header[f"CTYPE{idx}"], header[f"CRVAL{idx}"] = "STOKES", 1
        header[f"CDELT{idx}"], header[f"CRPIX{idx}"] = 1, 1
    header["BUNIT"] = bunit
    return header


def write_image(params, data, header):
    path = params.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
    return path


def brute_force_vis(image, header, uvw, chan_freqs, ncorr, polarisation, linear_basis, stokes_names=("I",)):
    """Textbook RIME over the non-zero pixels; astropy WCS for the coordinates.

    ``image`` has shape (nstokes, npix_l, npix_m, nchan_model).
    """
    wcs = WCS(header).celestial
    nstokes, npix_l, npix_m, nchan_model = image.shape
    nrow, nchan = uvw.shape[0], chan_freqs.size

    support = np.abs(image).max(axis=(0, 3)) > 0
    i_pix, j_pix = np.nonzero(support)
    ra, dec = wcs.wcs_pix2world(np.column_stack([i_pix, j_pix]), 0).T
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    dra = ra - RA0
    el = np.cos(dec) * np.sin(dra)
    em = np.sin(dec) * np.cos(DEC0) - np.cos(dec) * np.sin(DEC0) * np.cos(dra)
    en = np.sqrt(1 - el * el - em * em)

    stokes = {name: image[k][i_pix, j_pix] for k, name in enumerate(stokes_names)}  # (ncomp, nchan_model)
    get = lambda n: stokes.get(n, np.zeros_like(stokes["I"]))  # noqa: E731

    vis = np.zeros((nrow, nchan, ncorr), np.complex128)
    for r in range(nrow):
        u, v, w = uvw[r]
        for f in range(nchan):
            phasor = np.exp(2j * np.pi * (u * el + v * em + w * (en - 1)) * chan_freqs[f] / C)
            k = 0 if nchan_model == 1 else f
            i_, q_, u_, v_ = (get(n)[:, k] for n in "IQUV")
            if not polarisation:
                corrs = [i_, 0 * i_, 0 * i_, i_]
            elif linear_basis:
                corrs = [i_ + q_, u_ + 1j * v_, u_ - 1j * v_, i_ - q_]
            else:
                corrs = [i_ + v_, q_ + 1j * u_, q_ - 1j * u_, i_ - v_]
            sel = [0, 3] if ncorr == 2 else [0, 1, 2, 3]
            for c, s in enumerate(sel):
                vis[r, f, c] = np.sum(corrs[s] * phasor)
    return vis


def scattered_image(npix, nsrc, rng, nchan=1, amp=1.0):
    """A few off-centre pixels, deliberately away from the phase centre."""
    img = np.zeros((npix, npix, nchan))
    idx = rng.choice(npix * npix, nsrc, replace=False)
    ii, jj = np.unravel_index(idx, (npix, npix))
    img[ii, jj, :] = amp * rng.uniform(0.3, 1.0, (nsrc, 1))
    return img


# --------------------------------------------------------------------------- the core bug


@pytest.mark.parametrize("backend", ["dft", "fft"])
def test_single_plane_many_channels_off_centre(params, uvw, backend):
    """The case `expand_freq_dim` got 199% wrong: one FITS plane, many MS channels,
    emission away from the phase centre. Amplitudes alone cannot catch it."""
    npix, nchan = 64, 8
    chan_freqs = 1.4e9 + np.arange(nchan) * 2e7
    rng = np.random.default_rng(3)
    plane = scattered_image(npix, 6, rng)  # (npix, npix, 1)

    header = make_header(npix)
    write = np.ascontiguousarray(plane[:, :, 0].T)  # FITS is (dec, ra)
    path = write_image(params, write, header)

    prepared = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 2e7, 2, nrow=uvw.shape[0], backend=backend)
    assert prepared.backend == backend
    got = predict_fits_block(prepared, uvw, epsilon=1e-11)

    expected = brute_force_vis(plane[np.newaxis], header, uvw, chan_freqs, 2, False, True)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-6

    # and the phase genuinely varies across the band, so this is not degenerate
    assert np.ptp(np.angle(expected[0, :, 0])) > 1.0


def test_dft_and_fft_agree(params, uvw):
    """The two backends must agree; this is the guard for sign and 1/n conventions."""
    npix, nchan = 64, 4
    chan_freqs = 1.4e9 + np.arange(nchan) * 1e7
    rng = np.random.default_rng(5)
    plane = scattered_image(npix, 10, rng)
    header = make_header(npix)
    path = write_image(params, np.ascontiguousarray(plane[:, :, 0].T), header)

    out = {}
    for backend in ("dft", "fft"):
        prepared = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 1e7, 2, nrow=uvw.shape[0], backend=backend)
        out[backend] = predict_fits_block(prepared, uvw, epsilon=1e-11)

    scale = np.abs(out["dft"]).max()
    assert np.abs(out["dft"] - out["fft"]).max() / scale < 1e-6

    # w must be doing something, otherwise the w-term sign is untested
    assert np.abs(uvw[:, 2]).max() > 100


# --------------------------------------------------------------------------- polarisation


@pytest.mark.parametrize("backend", ["dft", "fft"])
@pytest.mark.parametrize("ncorr", [2, 4])
@pytest.mark.parametrize("linear_basis", [True, False])
def test_full_stokes(params, uvw, backend, ncorr, linear_basis):
    """Full-Stokes prediction, including via the gridder, which was impossible before."""
    npix, nchan = 64, 3
    chan_freqs = 1.4e9 + np.arange(nchan) * 1e7
    rng = np.random.default_rng(7)

    cube = np.zeros((4, npix, npix, nchan))
    idx = rng.choice(npix * npix, 5, replace=False)
    ii, jj = np.unravel_index(idx, (npix, npix))
    for s, amp in enumerate([1.0, 0.2, -0.15, 0.05]):
        cube[s, ii, jj, :] = amp * rng.uniform(0.5, 1.0, (5, 1))

    header = make_header(npix, nchan=nchan, nstokes=4, freqs=chan_freqs)
    data = np.transpose(cube, (0, 3, 2, 1))  # (stokes, freq, dec, ra)
    path = write_image(params, data, header)

    prepared = prepare_fits_sky(
        path,
        RA0,
        DEC0,
        chan_freqs,
        1e7,
        ncorr,
        nrow=uvw.shape[0],
        linear_basis=linear_basis,
        backend=backend,
    )
    assert prepared.polarisation
    got = predict_fits_block(prepared, uvw, epsilon=1e-11)
    expected = brute_force_vis(cube, header, uvw, chan_freqs, ncorr, True, linear_basis, "IQUV")
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-6


@pytest.mark.parametrize("backend", ["dft", "fft"])
def test_unpolarised_four_correlations(params, uvw, backend):
    npix = 64
    chan_freqs = np.array([1.4e9, 1.42e9])
    rng = np.random.default_rng(9)
    plane = scattered_image(npix, 4, rng)
    header = make_header(npix)
    path = write_image(params, np.ascontiguousarray(plane[:, :, 0].T), header)

    prepared = prepare_fits_sky(
        path, RA0, DEC0, chan_freqs, 2e7, 4, nrow=uvw.shape[0], polarisation=False, backend=backend
    )
    vis = predict_fits_block(prepared, uvw, epsilon=1e-11)
    np.testing.assert_allclose(vis[..., 0], vis[..., 3], rtol=1e-12, atol=1e-12)
    assert not np.any(vis[..., 1])
    assert not np.any(vis[..., 2])


# --------------------------------------------------------------------------- spectra


@pytest.mark.parametrize("backend", ["dft", "fft"])
def test_spectral_cube_on_the_ms_grid(params, uvw, backend):
    npix, nchan = 64, 5
    chan_freqs = 1.4e9 + np.arange(nchan) * 1e7
    rng = np.random.default_rng(13)
    cube = scattered_image(npix, 5, rng, nchan=nchan)
    cube *= (chan_freqs / chan_freqs[0]) ** -0.7  # a real spectral slope

    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    path = write_image(params, np.transpose(cube, (2, 1, 0)), header)

    prepared = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 1e7, 2, nrow=uvw.shape[0], backend=backend)
    assert not prepared.flat_spectrum
    got = predict_fits_block(prepared, uvw, epsilon=1e-11)
    expected = brute_force_vis(cube[np.newaxis], header, uvw, chan_freqs, 2, False, True)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-6


def test_ms_band_outside_fits_band_raises(params, uvw):
    npix, nchan = 32, 4
    fits_freqs = 1.4e9 + np.arange(nchan) * 1e6
    header = make_header(npix, nchan=nchan, freqs=fits_freqs)
    path = write_image(params, np.zeros((nchan, npix, npix)), header)
    with pytest.raises(FITSSkymodelError, match="outside the FITS image frequencies"):
        prepare_fits_sky(path, RA0, DEC0, np.array([2.0e9, 2.1e9]), 1e6, 2, nrow=uvw.shape[0])


# --------------------------------------------------------------------------- geometry


@pytest.mark.parametrize("proj,expect_regular", [("SIN", True), ("TAN", False), ("ZEA", False)])
def test_grid_regularity_detection(proj, expect_regular):
    """Only SIN tangent at the phase centre gives an exactly regular (l, m) grid."""
    npix, cell = 512, 20.0 / 3600
    header = make_header(npix, proj=proj, cell=cell)
    grid = fit_lm_grid(WCS(header).celestial, RA0, DEC0, npix, npix)
    assert grid.is_regular is expect_regular
    if expect_regular:
        assert grid.deviation_pixels < 1e-9
        np.testing.assert_allclose(grid.delta_l, -np.deg2rad(cell), rtol=1e-9)
        np.testing.assert_allclose(grid.delta_m, np.deg2rad(cell), rtol=1e-9)


def test_sin_grid_is_irregular_when_the_tangent_point_is_offset():
    npix = 512
    header = make_header(npix, crval1=RA0_DEG + 0.5)
    grid = fit_lm_grid(WCS(header).celestial, RA0, DEC0, npix, npix)
    assert not grid.is_regular


# A TAN image is only irregular enough to need reprojecting once the field is wide;
# at 1 degree TAN already sits within MAX_GRID_DEVIATION_PIXELS of a regular grid.
# The pixel must still sample the fringe (cell < 1 / (2 * u_max / lambda)), so a wide
# field means many pixels rather than coarse ones.
WIDE_TAN_NPIX = 512
WIDE_TAN_CELL = 20.0 / 3600  # 2.8 degrees across 512 pixels


def test_reprojection_conserves_flux(params):
    """Jy/pixel is a density in pixel index, so |det J| makes the total flux invariant."""
    npix = WIDE_TAN_NPIX
    rng = np.random.default_rng(17)
    # smooth emission: reprojection resamples, so a delta would not survive interpolation
    yy, xx = np.mgrid[:npix, :npix]
    plane = np.exp(-(((xx - 300) ** 2 + (yy - 220) ** 2) / (2 * 9.0**2)))
    plane += 0.5 * np.exp(-(((xx - 160) ** 2 + (yy - 330) ** 2) / (2 * 12.0**2)))
    plane += 1e-3 * rng.random((npix, npix))

    header = make_header(npix, proj="TAN", cell=WIDE_TAN_CELL)
    cel = WCS(header).celestial
    grid = fit_lm_grid(cel, RA0, DEC0, npix, npix)
    assert not grid.is_regular, "the source grid must actually need reprojecting"
    cell = min(abs(grid.delta_l), abs(grid.delta_m))

    out, new_grid = reproject_to_sin(plane[np.newaxis], cel, RA0, DEC0, cell)
    assert new_grid.is_regular
    assert new_grid.deviation_pixels == 0.0
    assert out.shape[-1] % 2 == 0 and out.shape[-2] % 2 == 0
    assert out.sum() == pytest.approx(plane.sum(), rel=1e-3)


def test_fft_on_a_wide_tan_image_matches_the_dft(params, uvw):
    """A TAN image too distorted to grid directly is reprojected to SIN; the DFT
    uses the WCS and needs no resampling. The two must still agree."""
    npix = WIDE_TAN_NPIX
    chan_freqs = np.array([1.4e9, 1.41e9])
    yy, xx = np.mgrid[:npix, :npix]
    plane = np.exp(-(((xx - 320) ** 2 + (yy - 205) ** 2) / (2 * 6.0**2)))
    header = make_header(npix, proj="TAN", cell=WIDE_TAN_CELL)
    path = write_image(params, np.ascontiguousarray(plane), header)

    assert not fit_lm_grid(WCS(header).celestial, RA0, DEC0, npix, npix).is_regular

    kw = dict(nrow=uvw.shape[0], tol=1e-6)
    dft = predict_fits_block(prepare_fits_sky(path, RA0, DEC0, chan_freqs, 1e7, 2, backend="dft", **kw), uvw)
    prepared_fft = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 1e7, 2, backend="fft", **kw)
    assert prepared_fft.grid.is_regular, "the gridder must be handed a reprojected image"
    assert (prepared_fft.npix_l, prepared_fft.npix_m) != (npix, npix)
    fft = predict_fits_block(prepared_fft, uvw, epsilon=1e-9)

    # resampling is not lossless, but the two must agree well for smooth emission
    assert np.abs(dft - fft).max() / np.abs(dft).max() < 1e-5


# --------------------------------------------------------------------------- backend choice


def test_cost_model_prefers_the_gridder_beyond_a_few_hundred_components():
    """The old rule chose the DFT at 20% of the image, up to 2750x slower."""
    npix, nchan = 1024, 16
    assert choose_backend(10, npix, npix, 20000, nchan, 1) == "dft"
    assert choose_backend(1000, npix, npix, 20000, nchan, 1) == "fft"
    # the old 80%-sparsity rule would have chosen the DFT here
    assert choose_backend(int(0.2 * npix * npix), npix, npix, 20000, nchan, 1) == "fft"
    # break-even falls as rows grow, because the image FFT amortises
    assert choose_backend(200, npix, npix, 2000, nchan, 1) == "dft"
    assert choose_backend(200, npix, npix, 200000, nchan, 1) == "fft"


def test_odd_image_falls_back_to_the_dft(params, uvw):
    """ducc0 requires even image dimensions."""
    npix = 65
    header = make_header(npix)
    data = np.zeros((npix, npix))
    data[40, 30] = 1.0
    path = write_image(params, data, header)
    prepared = prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], backend="fft")
    assert prepared.backend == "dft"


def test_unknown_backend_raises(params, uvw):
    header = make_header(32)
    path = write_image(params, np.zeros((32, 32)), header)
    with pytest.raises(FITSSkymodelError, match="Unknown predict backend"):
        prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], backend="wavelet")


# --------------------------------------------------------------------------- inputs


def test_stokes_from_a_list_of_files(params, uvw):
    """Four per-Stokes FITS images stacked along a synthesised STOKES axis."""
    npix, nchan = 32, 2
    chan_freqs = 1.4e9 + np.arange(nchan) * 1e7
    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    amps = [1.0, 0.2, -0.1, 0.05]
    paths = []
    cube = np.zeros((4, npix, npix, nchan))
    for s, amp in enumerate(amps):
        plane = np.zeros((nchan, npix, npix))
        plane[:, 20, 11] = amp
        cube[s, 11, 20, :] = amp
        paths.append(write_image(params, plane, header))

    prepared = prepare_fits_sky(paths, RA0, DEC0, chan_freqs, 1e7, 4, nrow=uvw.shape[0], backend="dft")
    assert prepared.polarisation
    got = predict_fits_block(prepared, uvw)
    expected = brute_force_vis(cube, header, uvw, chan_freqs, 4, True, True, "IQUV")
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-9


def test_jy_per_beam_is_converted(params, uvw):
    """Jy/beam is divided by pixels-per-beam so the component flux is in Jy."""
    npix = 64
    header = make_header(npix, bunit="Jy/beam")
    bmaj = bmin = 4 * CELL
    header["BMAJ"], header["BMIN"], header["BPA"] = bmaj, bmin, 0.0
    data = np.zeros((npix, npix))
    data[40, 30] = 1.0
    path = write_image(params, data, header)

    prepared = prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], backend="dft")
    beam_area = np.pi * np.deg2rad(bmaj) * np.deg2rad(bmin) / (4 * np.log(2))
    pixels_per_beam = beam_area / np.deg2rad(CELL) ** 2
    assert prepared.ncomp == 1
    assert np.abs(prepared.bmat[0, 0, 0]) == pytest.approx(1.0 / pixels_per_beam, rel=1e-6)


def test_empty_model_predicts_zeros(params, uvw):
    header = make_header(32)
    path = write_image(params, np.zeros((32, 32)), header)
    prepared = prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], backend="dft")
    assert prepared.ncomp == 0
    assert not np.any(predict_fits_block(prepared, uvw))


def test_spectral_interpolation_onto_the_ms_grid(params, uvw):
    """FITS and MS channel grids differ, so the cube is resampled in frequency."""
    npix, nchan_fits = 32, 5
    fits_freqs = np.linspace(1.30e9, 1.50e9, nchan_fits)
    chan_freqs = np.array([1.36e9, 1.40e9, 1.44e9])

    cube = np.zeros((npix, npix, nchan_fits))
    cube[20, 11, :] = 2.0 * (fits_freqs / fits_freqs[0])  # exactly linear in frequency
    header = make_header(npix, nchan=nchan_fits, freqs=fits_freqs)
    path = write_image(params, np.transpose(cube, (2, 1, 0)), header)

    prepared = prepare_fits_sky(
        path,
        RA0,
        DEC0,
        chan_freqs,
        4e7,
        2,
        nrow=uvw.shape[0],
        backend="dft",
        spectrum="cube",
        interpolation="linear",
    )
    assert prepared.spectrum.kind is SpectralKind.CUBE
    assert prepared.ncomp == 1
    # linear interpolation of a linear spectrum is exact
    expected = 2.0 * (chan_freqs / fits_freqs[0])
    np.testing.assert_allclose(np.abs(prepared.bmat[0, 0, :]), expected, rtol=1e-9)


def test_beam_table_scales_the_beam_as_one_over_frequency(params, uvw):
    """A one-row beam table on a cube is broadcast with a ref_freq/freq scaling,
    so pixels-per-beam - and hence the Jy/pixel flux - varies with channel."""
    from astropy.table import Table

    npix, nchan = 32, 3
    chan_freqs = 1.4e9 + np.arange(nchan) * 1e8
    bmaj = bmin = 4 * CELL

    cube = np.zeros((nchan, npix, npix))
    cube[:, 20, 11] = 1.0
    header = make_header(npix, nchan=nchan, freqs=chan_freqs, bunit="Jy/beam")
    beam_hdu = fits.BinTableHDU(Table({"BMAJ": [bmaj], "BMIN": [bmin], "BPA": [0.0]}))
    path = params.random_named_file(suffix=".fits")
    fits.HDUList([fits.PrimaryHDU(cube, header=header), beam_hdu]).writeto(path, overwrite=True)

    prepared = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 1e8, 2, nrow=uvw.shape[0], backend="dft")
    scale = chan_freqs[0] / chan_freqs
    beam_area = np.pi * np.deg2rad(bmaj * scale) * np.deg2rad(bmin * scale) / (4 * np.log(2))
    pixels_per_beam = beam_area / np.deg2rad(CELL) ** 2
    np.testing.assert_allclose(np.abs(prepared.bmat[0, 0, :]), 1.0 / pixels_per_beam, rtol=1e-6)


def test_no_stokes_i_plane_raises(params, uvw):
    npix = 32
    header = make_header(npix, nstokes=2)
    header["CRVAL3"] = 2  # STOKES axis starts at Q
    data = np.zeros((2, npix, npix))
    path = write_image(params, data, header)
    with pytest.raises(FITSSkymodelError, match="no Stokes I plane"):
        prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0])


# --------------------------------------------------------------------------- analytic spectra


def powerlaw_cube(npix, nsrc, rng, freqs, ref_freq, c1, c2=0.0):
    """A cube whose every pixel follows S0 * (nu/nu0)**(c1 + c2*ln(nu/nu0))."""
    plane = np.zeros((npix, npix))
    idx = rng.choice(npix * npix, nsrc, replace=False)
    ii, jj = np.unravel_index(idx, (npix, npix))
    plane[ii, jj] = rng.uniform(0.3, 1.0, nsrc)
    x = np.log(freqs / ref_freq)
    scale = np.exp(c1 * x + c2 * x**2)
    return plane[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]


@pytest.mark.parametrize("backend", ["dft", "fft"])
def test_auto_fits_a_log_polynomial_and_keeps_one_plane(params, uvw, backend):
    """A power-law cube is reduced to one reference plane plus coefficient maps,
    and still predicts the same visibilities as the cube it came from."""
    npix, nchan = 64, 6
    chan_freqs = np.linspace(1.2e9, 1.7e9, nchan)
    ref_freq = float(0.5 * (chan_freqs[0] + chan_freqs[-1]))
    rng = np.random.default_rng(23)
    cube = powerlaw_cube(npix, 5, rng, chan_freqs, ref_freq, c1=-0.7, c2=0.05)

    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    path = write_image(params, np.transpose(cube, (2, 1, 0)), header)

    prepared = prepare_fits_sky(
        path,
        RA0,
        DEC0,
        chan_freqs,
        chan_freqs[1] - chan_freqs[0],
        2,
        nrow=uvw.shape[0],
        backend=backend,
        ref_freq=ref_freq,
    )
    assert prepared.spectrum.kind is SpectralKind.POLY
    # an exact fit; the residual floor is set by cancellation in the normal equations
    assert prepared.spectrum.residual < 1e-6
    if backend == "fft":
        # one plane held, not nchan: this is the memory claim
        assert prepared.planes.shape[-1] == 1
        assert prepared.spectrum.coeffs.shape == (2, npix, npix)

    got = predict_fits_block(prepared, uvw, epsilon=1e-11)
    expected = brute_force_vis(cube[np.newaxis], header, uvw, chan_freqs, 2, False, True)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-6


def test_auto_keeps_the_cube_when_the_spectrum_is_not_a_power_law(params, uvw):
    """A spectral line is not a log-polynomial; auto must not force one."""
    npix, nchan = 32, 8
    chan_freqs = np.linspace(1.2e9, 1.7e9, nchan)
    rng = np.random.default_rng(29)
    cube = powerlaw_cube(npix, 3, rng, chan_freqs, chan_freqs[0], c1=-0.7)
    cube[..., nchan // 2] *= 4.0  # an emission line

    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    path = write_image(params, np.transpose(cube, (2, 1, 0)), header)

    prepared = prepare_fits_sky(
        path, RA0, DEC0, chan_freqs, chan_freqs[1] - chan_freqs[0], 2, nrow=uvw.shape[0], backend="dft"
    )
    assert prepared.spectrum.kind is SpectralKind.CUBE

    got = predict_fits_block(prepared, uvw)
    expected = brute_force_vis(cube[np.newaxis], header, uvw, chan_freqs, 2, False, True)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-9


@pytest.mark.parametrize("backend", ["dft", "fft"])
def test_intensity_map_plus_spectral_index_map(params, uvw, backend):
    """Issue #110: predict from an intensity map and a spectral-index map."""
    npix, nchan = 64, 5
    chan_freqs = np.linspace(1.2e9, 1.7e9, nchan)
    ref_freq = 1.4e9
    rng = np.random.default_rng(31)

    intensity = np.zeros((npix, npix))
    idx = rng.choice(npix * npix, 6, replace=False)
    ii, jj = np.unravel_index(idx, (npix, npix))
    intensity[ii, jj] = rng.uniform(0.3, 1.0, 6)
    alpha = np.full((npix, npix), -0.8)
    alpha[ii[:3], jj[:3]] = -0.2  # a couple of flatter-spectrum sources

    header = make_header(npix)
    image_path = write_image(params, np.ascontiguousarray(intensity.T), header)
    alpha_path = write_image(params, np.ascontiguousarray(alpha.T), header)

    prepared = prepare_fits_sky(
        image_path,
        RA0,
        DEC0,
        chan_freqs,
        1e8,
        2,
        nrow=uvw.shape[0],
        backend=backend,
        spi_maps=[alpha_path],
        ref_freq=ref_freq,
    )
    assert prepared.spectrum.kind is SpectralKind.POLY
    assert prepared.spectrum.ref_freq == ref_freq

    # the cube the maps describe
    cube = intensity[:, :, np.newaxis] * (chan_freqs / ref_freq)[np.newaxis, np.newaxis, :] ** alpha[:, :, np.newaxis]
    cube_header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    got = predict_fits_block(prepared, uvw, epsilon=1e-11)
    expected = brute_force_vis(cube[np.newaxis], cube_header, uvw, chan_freqs, 2, False, True)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-6


def test_spi_maps_require_a_reference_frequency(params, uvw):
    npix = 32
    header = make_header(npix)
    image_path = write_image(params, np.zeros((npix, npix)), header)
    alpha_path = write_image(params, np.full((npix, npix), -0.7), header)
    with pytest.raises(FITSSkymodelError, match="reference frequency"):
        prepare_fits_sky(image_path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], spi_maps=[alpha_path])


def test_spi_map_shape_must_match(params, uvw):
    header = make_header(32)
    image_path = write_image(params, np.zeros((32, 32)), header)
    alpha_path = write_image(params, np.full((16, 16), -0.7), make_header(16))
    with pytest.raises(FITSSkymodelError, match="but the intensity map is"):
        prepare_fits_sky(
            image_path,
            RA0,
            DEC0,
            np.array([1.4e9]),
            1e6,
            2,
            nrow=uvw.shape[0],
            spi_maps=[alpha_path],
            ref_freq=1.4e9,
        )


def test_reprojection_does_not_flux_scale_the_coefficient_maps(params, uvw):
    """Spectral indices are exponents, not densities. Scaling them by |det J| would
    tilt the spectrum wherever the pixel area changes."""
    npix = WIDE_TAN_NPIX
    chan_freqs = np.array([1.3e9, 1.5e9])
    yy, xx = np.mgrid[:npix, :npix]
    intensity = np.exp(-(((xx - 320) ** 2 + (yy - 205) ** 2) / (2 * 6.0**2)))
    alpha = np.full((npix, npix), -0.7)  # uniform, so any scaling is visible

    header = make_header(npix, proj="TAN", cell=WIDE_TAN_CELL)
    assert not fit_lm_grid(WCS(header).celestial, RA0, DEC0, npix, npix).is_regular

    image_path = write_image(params, np.ascontiguousarray(intensity.T), header)
    alpha_path = write_image(params, np.ascontiguousarray(alpha.T), header)

    prepared = prepare_fits_sky(
        image_path,
        RA0,
        DEC0,
        chan_freqs,
        1e8,
        2,
        nrow=uvw.shape[0],
        backend="fft",
        spi_maps=[alpha_path],
        ref_freq=1.4e9,
        tol=1e-6,
    )
    assert prepared.grid.is_regular, "the image must have been reprojected"
    assert (prepared.npix_l, prepared.npix_m) != (npix, npix)

    # the Jacobian is not unity across a wide TAN image, so an erroneous scaling would show
    interior = prepared.spectrum.coeffs[0][8:-8, 8:-8]
    np.testing.assert_allclose(interior, -0.7, rtol=1e-6)


def test_explicit_flat_rejects_a_cube(params, uvw):
    npix, nchan = 32, 4
    chan_freqs = np.linspace(1.3e9, 1.5e9, nchan)
    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    path = write_image(params, np.zeros((nchan, npix, npix)), header)
    with pytest.raises(FITSSkymodelError, match="'flat' spectrum needs a single frequency plane"):
        prepare_fits_sky(path, RA0, DEC0, chan_freqs, 5e7, 2, nrow=uvw.shape[0], spectrum="flat")


def test_explicit_poly_rejects_a_single_plane(params, uvw):
    header = make_header(32)
    path = write_image(params, np.zeros((32, 32)), header)
    with pytest.raises(FITSSkymodelError, match="single frequency plane"):
        prepare_fits_sky(path, RA0, DEC0, np.array([1.4e9]), 1e6, 2, nrow=uvw.shape[0], spectrum="poly")


def test_short_cube_lowers_the_fit_order_rather_than_failing(params, uvw):
    """auto must degrade an order-2 fit to order 1 when only two channels exist."""
    npix, nchan = 32, 2
    chan_freqs = np.array([1.3e9, 1.5e9])
    rng = np.random.default_rng(37)
    cube = powerlaw_cube(npix, 3, rng, chan_freqs, 1.4e9, c1=-0.7)
    header = make_header(npix, nchan=nchan, freqs=chan_freqs)
    path = write_image(params, np.transpose(cube, (2, 1, 0)), header)

    prepared = prepare_fits_sky(path, RA0, DEC0, chan_freqs, 2e8, 2, nrow=uvw.shape[0], backend="dft")
    assert prepared.spectrum.kind is SpectralKind.POLY
    assert prepared.spectrum.coeffs.shape[0] == 1  # order dropped from 2 to 1
