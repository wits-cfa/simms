import numpy as np
import pytest

from simms.skymodel.mstools import compute_vis
from simms.skymodel.skymods import skymodel_from_sources
from simms.skymodel.source_factory import (
    CatSource,
)
from simms.telescope.array_utilities import Array
from simms.utilities import ParameterError as SkymodelError


class InitTests:
    def __init__(self):
        # set up test inputs
        test_array = Array("meerkat")

        uv_coverage_data = test_array.uvgen(
            pointing_direction="J2000,0deg,-30deg".split(","),
            dtime=8,
            ntimes=75,
            start_freq="1293MHz",
            dfreq="206kHz",
            nchan=16,
        )

        self.uvw = uv_coverage_data.uvw
        self.freqs = uv_coverage_data.freqs
        self.times = uv_coverage_data.times

        self.ra0 = uv_coverage_data.ra0
        self.dec0 = uv_coverage_data.dec0

@pytest.fixture
def params():
    return InitTests()
   
def test_transient_visibility_shape(params):
    """
    Test that the visibility matrix follows (nrow, nchan, ncorr).
    Validates:
    - Output shape of visibilities
    """

    source = CatSource(
        name="test_source",
        ra=f"{params.ra0}rad",
        dec=f"{params.dec0}rad",
        stokes_i="1",
        transient_start="100",
        transient_absorb="0.5",
        transient_period="100",
        transient_ingress="20",
    )

    skymodel = skymodel_from_sources(sources=[source], chan_freqs=params.freqs, 
                                     unique_times=np.unique(params.times), full_stokes=True)

    ncorr = 4

    vis = compute_vis(
        sources=skymodel,
        uvw=params.uvw,
        freqs=params.freqs,
        times=params.times,
        ncorr=ncorr,
        polarisation=False,
        pol_basis='linear',
        ra0=params.ra0,
        dec0=params.dec0
    )

    nrow = params.uvw.shape[0]
    nchan = params.freqs.size
    assert vis.shape == (nrow, nchan, ncorr)

def test_transient_visibility_dip(params):
     """
    Test that the transient dip is present in visibilities.
    Validates:
    - a dip in flux is present in the time series
    - a dip occurs near the transient start time
    - flux recovers to original value after transient
    """
       
     source = CatSource(
        name="test_source",
        ra=f"{params.ra0}rad",
        dec=f"{params.dec0}rad",
        stokes_i="1",
        transient_start="100",
        transient_absorb="0.5",
        transient_period="100",
        transient_ingress="20",
    )

     skymodel = skymodel_from_sources(sources=[source], chan_freqs=params.freqs, 
                                      unique_times=np.unique(params.times), full_stokes=True)

     ncorr = 4

     vis = compute_vis(
        sources=skymodel,
        uvw=params.uvw,
        freqs=params.freqs,
        times=params.times,
        ncorr=ncorr,
        polarisation=False,
        pol_basis='linear',
        ra0=params.ra0,
        dec0=params.dec0
    )

     flux_time = np.mean(np.abs(vis[:,:,0]), axis=1)

     avg_flux = np.mean(flux_time)
     assert avg_flux < 1.0, "Transient should reduce flux below baseline (I=1)."

     times_rel = params.times - np.min(params.times)

     t_min = times_rel[np.argmin(flux_time)]

     transient_start = float(source.transient_start)
     tolerance = float(source.transient_ingress) * 1.5
     assert abs(t_min - transient_start) < tolerance, (
        f"Minimum flux ({t_min:.2f}) not near transient start ({transient_start})."
    )

     pre_dip = np.mean(flux_time[times_rel < transient_start - 20])
     post_dip = np.mean(flux_time[times_rel > transient_start + 80])
     assert abs(pre_dip - post_dip) < 0.1, "Flux should recover after transient."
    
def test_transient_missing_params(params):
    """
    Test that missing required transient parameters raise an error.
    Validates:
    - SkymodelError is raised for missing parameters
    """
    
    with pytest.raises(SkymodelError) as exception:
        source = CatSource(
            name="test_source",
            ra="0rad",
            dec="0rad",
            stokes_i="1",
            transient_start=None,    # Missing parameter
            transient_absorb="0.5",   
            transient_period="100",
            transient_ingress=None,  # Missing parameter
        )

        skymodel_from_sources(
            sources=[source], chan_freqs=params.freqs, unique_times=np.unique(params.times), full_stokes=True
            )

    assert exception.type is SkymodelError
    assert "missing required parameter(s)" in str(exception.value)
