"""End-to-end primary-beam test: telsim MS + ASCII sky + beam config through skysim.runit."""

import numpy as np
import pytest
from daskms import xds_from_ms
from omegaconf import OmegaConf

from simms.apps import skysim
from simms.telescope.generate_ms import create_ms

from . import InitTest


def _opts(ms, ascii_sky, primary_beam=None, column="DATA"):
    """A minimal skysim opts object with sensible defaults for the ASCII path."""
    return OmegaConf.create(
        {
            "ms": ms,
            "ascii_sky": ascii_sky,
            "fits_sky": None,
            "wsclean_sky": None,
            "nworkers": 1,
            "row_chunks": 100000,
            "field_id": 0,
            "spw_id": 0,
            "sefd": None,
            "polarisation": False,
            "pol_basis": "linear",
            "chan_chunks": None,
            "source_schema": None,
            "ascii_species": None,
            "ascii_delimiter": None,
            "primary_beam": primary_beam,
            "beam_band": "L",
            "beam_pa_step": 1.0,
            "beam_jones": "diagonal",
            "telescope_name_column": "TELESCOPE_NAME",
            "input_column": None,
            "mode": "sim",
            "column": column,
            "seed": None,
            "log_level": "CRITICAL",
            # FITS-image path defaults (used only when fits_sky is set).
            "pixel_tol": 1e-7,
            "predict_backend": "auto",
            "fits_spectrum": "flat",
            "fits_spi": None,
            "fits_ref_freq": None,
            "fits_spectrum_order": 2,
            "fits_sky_interp": "linear",
            "fft_precision": "double",
            "do_wstacking": True,
        }
    )


class _E2E(InitTest):
    def __init__(self):
        self.test_files = []
        self.ms = self.random_named_directory(suffix=".ms")
        # Heterogeneous subarray spanning the skamid 13.5/15 m boundary.
        create_ms(
            self.ms,
            telescope_name="skamid",
            pointing_direction=["J2000", "0h0m20s", "-30deg"],
            dtime=600,
            ntimes=4,
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
        # A single source ~1 deg south of the phase centre (near the L-band half power).
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\n")
            fh.write("S 0h0m20s -31d0m0s 4.0\n")
        self.beams = self.random_named_file(suffix=".yaml")
        with open(self.beams, "w") as fh:
            fh.write("MKAT-MA:\n  jimbeam: MKAT-MA-L-JIM-2026\nMKAT-EA:\n  jimbeam: MKAT-EA-L-JIM-2026\n")


@pytest.fixture
def e2e():
    return _E2E()


def test_beam_attenuates_and_differs_from_nobeam(e2e):
    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=None, column="NOBEAM"))
    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=e2e.beams, column="BEAM"))

    ds = xds_from_ms(e2e.ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()

    # The beam changes the visibilities and attenuates an off-centre source.
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() < np.abs(nobeam).mean()
    # Unpolarised source: cross-hand-free 2-corr stays finite and non-zero.
    assert np.all(np.isfinite(beam))
    assert np.abs(beam).max() > 0


def test_beam_rejects_nonlinear_correlations(e2e):
    # A circular-correlation MS must be refused: the beam's feed mapping is linear-only.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=2,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    with pytest.raises(RuntimeError, match="linear correlations"):
        skysim.runit(_opts(ms, e2e.sky, primary_beam=e2e.beams))


def test_primary_beam_ignored_without_sky_model(e2e):
    # --primary-beam with a noise-only run (no sky model) must be a no-op, not a crash.
    opts = _opts(e2e.ms, ascii_sky=None, primary_beam=e2e.beams, column="NOISEONLY")
    opts.sefd = 500.0
    skysim.runit(opts)  # must not raise (beam ignored, noise written)
    ds = xds_from_ms(e2e.ms)[0]
    assert np.any(ds.NOISEONLY.data.compute() != 0)


def test_full_jones_requires_four_correlations(e2e):
    # The fixture MS is 2-corr (XX, YY); full Jones needs all four.
    opts = _opts(e2e.ms, e2e.sky, primary_beam=e2e.beams)
    opts.beam_jones = "full"
    with pytest.raises(RuntimeError, match="4 correlations"):
        skysim.runit(opts)


def test_full_jones_circular_ms(e2e):
    # Full 2x2 Jones on a circular-correlation (RR,RL,LR,LL) MS: runs and attenuates.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=3,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "RL", "LR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    nb = _opts(ms, e2e.sky, primary_beam=None, column="NOBEAM")
    nb.pol_basis = "circular"
    skysim.runit(nb)
    bm = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM")
    bm.beam_jones = "full"
    skysim.runit(bm)

    ds = xds_from_ms(ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()
    assert np.all(np.isfinite(beam))
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() < np.abs(nobeam).mean()


def test_fits_image_beam_accepts_circular_ms(e2e):
    # The FITS-image scalar power beam is basis-independent, so an RR/LL MS is accepted.
    from astropy.io import fits

    from .predict_fits_tests import make_header

    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "1h0m0s", "-31deg"],  # matches make_header RA0/DEC0
        dtime=600,
        ntimes=3,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    npix = 256
    data = np.zeros((npix, npix), dtype=np.float32)
    data[npix // 2 - 90, npix // 2] = 1.0  # ~0.5 deg south of centre (near half power)
    img = e2e.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=data, header=make_header(npix, nstokes=1, nchan=1)).writeto(img)

    def run(column, primary_beam):
        opts = _opts(ms, ascii_sky=None, primary_beam=primary_beam, column=column)
        opts.fits_sky = img
        opts.pol_basis = "circular"
        skysim.runit(opts)

    run("NOBEAM", None)
    run("BEAM", e2e.beams)  # must not raise on a circular MS
    ds = xds_from_ms(ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()
    assert np.all(np.isfinite(beam))
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() <= np.abs(nobeam).mean()


def test_configurable_telescope_name_column(e2e):
    # telsim writes the label to a custom column; skysim must read the same name, and
    # the default name (absent here) must fail clearly rather than infer the metadata.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=2,
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
        telescope_name_column="DISH_TYPE",
    )
    ok = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM")
    ok.telescope_name_column = "DISH_TYPE"
    skysim.runit(ok)  # reads the custom column
    assert np.all(np.isfinite(xds_from_ms(ms)[0].BEAM.data.compute()))

    bad = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM2")  # default TELESCOPE_NAME, absent
    with pytest.raises(RuntimeError, match="TELESCOPE_NAME"):
        skysim.runit(bad)
