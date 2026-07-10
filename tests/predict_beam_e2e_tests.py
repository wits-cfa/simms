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
            "input_column": None,
            "mode": "sim",
            "column": column,
            "seed": None,
            "log_level": "CRITICAL",
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
            fh.write("MK:\n  jimbeam: L\nMKE:\n  jimbeam: L\n")


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


def test_beam_requires_linear_basis(e2e):
    opts = _opts(e2e.ms, e2e.sky, primary_beam=e2e.beams)
    opts.pol_basis = "circular"
    with pytest.raises(RuntimeError, match="linear polarisation basis"):
        skysim.runit(opts)


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
