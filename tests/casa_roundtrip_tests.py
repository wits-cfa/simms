"""CASA round-trip test: telsim MS -> tclean PSF -> skysim -> tclean image.

The ultimate check that an MS is written correctly is to run real CASA tasks on it. This
exercises the MS against the exact tooling users run, catching table/metadata defects that
unit tests miss (e.g. an undefined SPECTRAL_WINDOW.MEAS_FREQ_REF, which makes tclean fail).

Opt-in: it needs ``casatasks`` (the ``casa`` dependency group) and is skipped otherwise.
"""

import os

import numpy as np
import pytest
from daskms import xds_from_ms
from omegaconf import OmegaConf

from simms.apps import skysim
from simms.telescope.generate_ms import create_ms

from . import InitTest

C = 299792458.0
REF_FREQ_HZ = 1.42e9
# Point sources at known positions with distinct fluxes (so a swap/mis-location is caught):
# one on the phase centre, one offset in RA, one offset in Dec -- all well inside the field.
SOURCES = [
    ("A", "0h0m0s", "-30d0m0s", 5.0),  # phase centre
    ("B", "0h2m0s", "-30d0m0s", 3.0),  # ~0.43 deg east
    ("C", "0h0m0s", "-30d40m0s", 2.0),  # ~0.67 deg south
]


def _skysim_opts(ms, ascii_sky, column):
    """Minimal skysim opts for a plain (no-beam) ASCII-model simulation."""
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
            "primary_beam": None,
            "beam_band": "L",
            "beam_pa_step": 1.0,
            "beam_jones": "diagonal",
            "telescope_name_column": "TELESCOPE_NAME",
            "input_column": None,
            "mode": "sim",
            "column": column,
            "seed": None,
            "log_level": "CRITICAL",
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


class _Roundtrip(InitTest):
    def __init__(self):
        self.test_files = []
        # kat-7: a small, compact array -> short baselines, large synthesised beam, so imaging
        # is coarse and cheap. Declination -30 transits near the KAT-7 zenith for good coverage.
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="kat-7",
            pointing_direction=["J2000", "0h0m0s", "-30deg"],
            dtime=120,
            ntimes=20,
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
        )
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\n")
            for name, ra, dec, flux in SOURCES:
                fh.write(f"{name} {ra} {dec} {flux}\n")
        self.outdir = self.random_named_directory(suffix=".img")

    def imaging_grid(self):
        """A cell/imsize that oversamples the synthesised beam, derived from the uv coverage."""
        uvw = xds_from_ms(self.ms)[0].UVW.data.compute()
        uvmax_lambda = np.hypot(uvw[:, 0], uvw[:, 1]).max() / (C / REF_FREQ_HZ)
        cell_arcsec = np.degrees(1.0 / uvmax_lambda) * 3600.0 / 6.0  # ~6x oversampling
        return cell_arcsec, 256


@pytest.fixture
def rt():
    return _Roundtrip()


def test_casa_roundtrip_recovers_point_source(rt):
    tclean = pytest.importorskip("casatasks").tclean
    from casatasks import imstat

    cell_arcsec, imsize = rt.imaging_grid()

    def image(prefix, niter=0, savemodel="none"):
        tclean(
            vis=rt.ms,
            imagename=prefix,
            imsize=[imsize, imsize],
            cell=[f"{cell_arcsec}arcsec"],
            specmode="mfs",
            gridder="standard",
            weighting="natural",
            niter=niter,
            threshold="0.01Jy",
            savemodel=savemodel,
            datacolumn="data",
            stokes="I",
            pblimit=-1,  # no PB correction; we test the bare imaging round trip
        )

    # Step 1+2: the freshly created MS must be imageable and yield a unit-peak PSF. This is what
    # regressed when SPECTRAL_WINDOW.MEAS_FREQ_REF was left undefined (tclean raised on it).
    psf_prefix = os.path.join(rt.outdir, "psf")
    image(psf_prefix)
    assert imstat(f"{psf_prefix}.psf")["max"][0] == pytest.approx(1.0, abs=1e-4)

    # Step 3: simulate the sources into DATA.
    skysim.runit(_skysim_opts(rt.ms, rt.sky, column="DATA"))
    data = xds_from_ms(rt.ms)[0].DATA.data.compute()
    assert np.all(np.isfinite(data))
    assert np.abs(data).max() > 0

    # Step 4: clean the simulated MS and write the model back to MODEL_DATA. niter>0 +
    # savemodel='modelcolumn' also exercises tclean's *write* path into the MS, which only
    # works if the MS is well formed.
    img_prefix = os.path.join(rt.outdir, "img")
    image(img_prefix, niter=5000, savemodel="modelcolumn")

    # Each source is recovered at its known sky position with its input flux. imstat over a
    # small region centred on the source reads the restored peak (Jy/beam == Jy for a point
    # source); recovery is a few percent low from sub-pixel gridding and finite cleaning.
    for name, ra, dec, flux in SOURCES:
        st = imstat(imagename=f"{img_prefix}.image", region=f"circle[[{ra}, {dec}], 120arcsec]")
        assert st["max"][0] == pytest.approx(flux, rel=0.1), f"source {name}: {st['max'][0]} vs {flux}"

    # tclean must have created and populated MODEL_DATA (a real sky model -> finite, non-zero).
    model = xds_from_ms(rt.ms, columns=["MODEL_DATA"])[0].MODEL_DATA.data.compute()
    assert np.all(np.isfinite(model))
    assert np.abs(model).max() > 0
