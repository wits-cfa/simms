"""Tests for the `simms primary-beam` utility (to-fits, tag-ms, apply, correct)."""

import logging

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from daskms import xds_from_table
from omegaconf import OmegaConf

from simms.apps import primary_beam
from simms.skymodel.beams import CosineTaperBeam, FitsBeamProvider, JimBeamProvider
from simms.telescope.generate_ms import create_ms

from . import InitTest
from .predict_fits_tests import DEC0_DEG, RA0_DEG, make_header


def _opts(mode, **over):
    base = {
        "mode": mode,
        "beam_pattern": "MKAT-EA-L-JIM-2026",
        "beam_band": "L",
        "beam_pa_step": 1.0,
        "ms": None,
        "fits_sky": None,
        "ascii_sky": None,
        "output": None,
        "telescope_name_column": "TELESCOPE_NAME",
        "label": None,
        "label_map": None,
        "from_layout": None,
        "pb_cutoff": 0.1,
        "field_id": 0,
        "spw_id": 0,
        "pixel_size": "2arcmin",
        "npix": 64,
        "start_freq": None,
        "chan_width": None,
        "nchan": None,
        "nworkers": 1,
        "log_level": "CRITICAL",
    }
    base.update(over)
    return OmegaConf.create(base)


class _Fixtures(InitTest):
    def __init__(self):
        self.test_files = []
        # A small heterogeneous skamid subarray (M060..M063 + SKA001..SKA004).
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="skamid",
            pointing_direction=["J2000", "1h0m0s", "-31deg"],  # matches make_header RA0/DEC0
            dtime=600,
            ntimes=3,
            start_freq="1420MHz",
            dfreq="4MHz",
            nchan=2,
            correlations=["XX", "YY"],
            row_chunks=100000,
            sefd=None,
            column="DATA",
            start_time="2025-03-06T20:00:00",
            smooth=None,
            fit_order=None,
            subarray_range=[60, 68],
        )


@pytest.fixture
def fx():
    return _Fixtures()


def test_to_fits_roundtrips_through_provider(fx):
    out = fx.random_named_file(suffix=".fits")
    # start 1300 MHz, 100 MHz channels x 3 -> [1.3, 1.4, 1.5] GHz; 1.4 GHz is a node, so the
    # round-trip has no frequency-interp error.
    primary_beam.runit(
        _opts(
            "to-fits",
            beam_pattern="MKAT-EA-L-JIM-2026",
            pixel_size="1arcmin",
            npix=128,
            start_freq="1300MHz",
            chan_width="100MHz",
            nchan=3,
            output=out,
        )
    )

    prov = FitsBeamProvider.from_fits(out)  # must reload
    jim = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026"))
    # Off-beam-core points; bilinear interpolation off the 1' grid is accurate to a few 1e-3.
    ell, emm = np.array([0.0, 0.004]), np.array([0.0, -0.003])
    freqs = np.array([1.4e9])
    np.testing.assert_allclose(
        prov.voltage(ell, emm, freqs, np.array([0.0])),
        jim.voltage(ell, emm, freqs, np.array([0.0])),
        atol=3e-3,
    )


def test_tag_ms_scalar_and_layout_and_map(fx):
    col = "TELESCOPE_NAME"

    primary_beam.runit(_opts("tag-ms", ms=fx.ms, label="FOO"))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {"FOO"}

    primary_beam.runit(_opts("tag-ms", ms=fx.ms, from_layout="skamid"))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {"MKAT-MA", "MKAT-EA"}

    names = [str(x) for x in np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0].NAME.data.compute()).astype(str)]
    mp = fx.random_named_file(suffix=".yaml")
    with open(mp, "w") as fh:
        fh.write("\n".join(f"{n}: T{i}" for i, n in enumerate(names)) + "\n")
    primary_beam.runit(_opts("tag-ms", ms=fx.ms, label_map=mp))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {f"T{i}" for i in range(len(names))}


def _write_image(fx, npix=256, off=90):
    data = np.zeros((npix, npix), dtype=np.float32)
    data[npix // 2, npix // 2] = 3.0  # centre source
    data[npix // 2 - off, npix // 2] = 2.0  # ~0.5 deg south source (within the beam)
    path = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=data, header=make_header(npix, nstokes=1, nchan=1)).writeto(path)
    return path, (npix // 2, npix // 2), (npix // 2 - off, npix // 2)


def test_apply_then_correct_image_is_identity(fx):
    img, centre, offsrc = _write_image(fx)
    original = fits.getdata(img)

    apparent = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=apparent))
    app = fits.getdata(apparent)
    # Off-centre source is attenuated; centre source ~unchanged.
    assert app[offsrc] < original[offsrc]
    assert app[offsrc] > 0

    recovered = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("correct", ms=fx.ms, fits_sky=apparent, output=recovered))
    rec = fits.getdata(recovered)
    # Round-trip recovers the source fluxes (both are inside the beam).
    np.testing.assert_allclose(rec[centre], original[centre], rtol=1e-4)
    np.testing.assert_allclose(rec[offsrc], original[offsrc], rtol=1e-4)
    # Corners (beam below cutoff) are blanked by correct.
    assert np.isnan(rec[0, 0])


def test_apply_centres_beam_on_pointing_centre(fx, caplog):
    # Image reference pixel offset 0.4 deg north of the pointing centre (dec -31). The primary
    # beam belongs to the antenna pointing centre (POINTING.DIRECTION), not the image reference.
    npix = 256
    header = make_header(npix, nstokes=1, nchan=1, crval2=DEC0_DEG + 0.4)
    img = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=np.ones((npix, npix), np.float32), header=header).writeto(img)

    out = fx.random_named_file(suffix=".fits")
    with caplog.at_level(logging.WARNING):
        primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=out, log_level="WARNING"))
    assert any("pointing centre" in r.message for r in caplog.records)

    # Uniform input -> the apparent image *is* the power beam A(l, m). It must peak on the
    # pointing centre, not on the image reference pixel (which is 0.4 deg off it).
    app = fits.getdata(out)
    ((col, row),) = WCS(header).celestial.wcs_world2pix([[RA0_DEG, DEC0_DEG]], 0)
    pc = (int(round(row)), int(round(col)))  # numpy [dec, ra]
    ref = (npix // 2, npix // 2)  # image reference pixel
    assert app[pc] > app[ref]  # would fail if the beam were centred on the image reference
    assert app[pc] > 0.9  # ~on-axis at the pointing centre


def _set_pointing_direction(ms, ra_rad, dec_rad):
    """Overwrite POINTING.DIRECTION so it differs from FIELD.PHASE_DIR (which simms keeps equal)."""
    import dask
    import dask.array as da
    from daskms import xds_from_table, xds_to_table

    pnt = xds_from_table(f"{ms}::POINTING")[0]
    nrow = pnt.DIRECTION.shape[0]
    newdir = np.broadcast_to(np.array([ra_rad, dec_rad]), (nrow, 1, 2)).copy()
    pnt = pnt.assign(DIRECTION=(("row", "point-poly", "radec"), da.from_array(newdir, chunks=(nrow, 1, 2))))
    dask.compute(xds_to_table([pnt], f"{ms}::POINTING", columns=["DIRECTION"]))


def test_apply_uses_pointing_direction_not_phase_centre(fx):
    # Point the dishes 0.3 deg north of the phase centre; the image WCS stays on the phase centre.
    _set_pointing_direction(fx.ms, np.radians(RA0_DEG), np.radians(DEC0_DEG + 0.3))

    npix = 256
    header = make_header(npix, nstokes=1, nchan=1)  # reference == phase centre (dec -31)
    img = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=np.ones((npix, npix), np.float32), header=header).writeto(img)

    out = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=out))

    # The beam must follow POINTING.DIRECTION (dec -30.7), not FIELD.PHASE_DIR (dec -31).
    app = fits.getdata(out)
    wcs = WCS(header).celestial
    ((pc_col, pc_row),) = wcs.wcs_world2pix([[RA0_DEG, DEC0_DEG + 0.3]], 0)  # pointing pixel
    pnt_pix = (int(round(pc_row)), int(round(pc_col)))
    phase_pix = (npix // 2, npix // 2)  # phase centre / image reference pixel
    assert app[pnt_pix] > app[phase_pix]  # follows POINTING, not PHASE_DIR
    assert app[pnt_pix] > 0.9


def test_apply_then_correct_ascii_is_identity(fx):
    sky = fx.random_named_file(suffix=".txt")
    with open(sky, "w") as fh:
        fh.write("#format: name ra dec stokes_i\n")
        fh.write("A 1h0m0s -31d0m0s 5.0\n")  # at phase centre
        fh.write("\n")  # blank line between sources
        fh.write("# a comment between sources -- lineno mapping must skip it\n")
        fh.write("B 1h0m0s -31d30m0s 3.0\n")  # ~0.5 deg off

    apparent = fx.random_named_file(suffix=".txt")
    primary_beam.runit(_opts("apply", ms=fx.ms, ascii_sky=sky, output=apparent))
    app_flux = _read_ascii_flux(apparent)
    assert app_flux["A"] == pytest.approx(5.0, rel=1e-3)  # on-axis ~unattenuated
    assert app_flux["B"] < 3.0  # off-axis attenuated

    recovered = fx.random_named_file(suffix=".txt")
    primary_beam.runit(_opts("correct", ms=fx.ms, ascii_sky=apparent, output=recovered))
    rec_flux = _read_ascii_flux(recovered)
    assert rec_flux["A"] == pytest.approx(5.0, rel=1e-3)
    assert rec_flux["B"] == pytest.approx(3.0, rel=1e-3)


def _read_ascii_flux(path):
    flux = {}
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        flux[f[0]] = float(f[3])  # name ra dec stokes_i
    return flux
