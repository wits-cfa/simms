import pytest
import numpy as np


from simms.skymodel.mstools import compute_vis
from simms.skymodel.source_factory import (
    CatSource,
    StokesData,
    contspec,
)
from simms.skymodel.skymods import skymodel_from_sources
from simms.telescope.array_utilities import Array


class InitTests:
    def __init__(self):
        # set up test inputs
        test_array = Array('meerkat')
        
        
        uv_coverage_data = test_array.uvgen(
            pointing_direction = 'J2000,0deg,-30deg'.split(','),
            dtime = 8,
            ntimes = 75,
            start_freq = '1293MHz',
            dfreq = '206kHz',
            nchan = 16
        )
        
        self.uvw = uv_coverage_data.uvw
        self.freqs = uv_coverage_data.freqs
        
        self.ra0 = uv_coverage_data.ra0
        self.dec0 = uv_coverage_data.dec0
        
@pytest.fixture
def params():
    return InitTests()


def test_compute_vis_stokes_I(params):
    """
    Test that it stills works when only Stokes I is provided.
    Validates:
    - Output shape of visibilities
    - XX = I
    - XY = 0 (if ncorr == 4)
    - YX = 0 (if ncorr == 4)
    - YY = I
    """
    
    sources = [
        CatSource(
            name = 'test_source',
            ra = f'{params.ra0}rad',
            dec = f'{params.dec0}rad',
            stokes_i = "1",
            )]

    skymodel = skymodel_from_sources(sources, params.freqs, full_stokes=True)
    
    ncorr = 2
    vis = compute_vis(sources=sources, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=False,
                    pol_basis='linear', ra0=params.ra0, dec0=params.dec0)
    
    nrow = params.uvw.shape[0]
    nchan = params.freqs.size
    
    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
    np.testing.assert_allclose(vis[:, :, 1], 1.0, atol=1e-6) # check that YY = I = 1

    ncorr = 4
    vis = compute_vis(sources=sources, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=False,
                    pol_basis='linear', ra0=params.ra0, dec0=params.dec0)
    
    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
    np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
    np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
    np.testing.assert_allclose(vis[:, :, 3], 1.0, atol=1e-6) # check that YY = I = 1
        
    
def test_compute_vis_IQ(params):
    """
    Test compute_vis with only Stokes I and Q provided.
    Validates:
    - Output shape of visibilities
    - XX = I + Q
    - YY = I - Q
    """
    
    sources = [
        CatSource(
            name = 'test_source',
            ra = f'{params.ra0}rad',
            dec = f'{params.dec0}rad',
            stokes_i = "1",
            stokes_q = "1",
            )]

    skymodel = skymodel_from_sources(sources, params.freqs, full_stokes=True)

    ncorr = 2
    vis = compute_vis(skymodel, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=True,
                    pol_basis='linear', ra0=params.ra0, dec0=params.dec0)

    nrow = params.uvw.shape[0]
    nchan = params.freqs.size

    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
    np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that YY = I - Q = 0
    
    ncorr = 4
    vis = compute_vis(skymodel, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=True,
                    pol_basis='linear', ra0=params.ra0, dec0=params.dec0)

    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
    np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
    np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
    np.testing.assert_allclose(vis[:, :, 3], 0.0, atol=1e-6) # check that YY = I - Q = 0


def test_compute_vis_all_stokes_linear_basis(params):
    """
    Test compute_vis with ncorr == 4 and Stokes I, Q, U and V provided.
    Validates:
    - Output shape of visibilities
    - XX = I + Q
    - XY = U + iV
    - YX = U - iV
    - YY = I - Q
    """
    # the numbers below are unphysical—they are just for testing the computation
    I = 1.0
    Q = 2.0
    U = 3.0
    V = 4.0
    sources = [
        CatSource(
            name = 'test_source',
            ra = f'{params.ra0}rad',
            dec = f'{params.dec0}rad',
            stokes_i = str(I),
            stokes_q = str(Q),
            stokes_u = str(U),
            stokes_v = str(V),
            )]

    skymodel = skymodel_from_sources(sources, params.freqs, full_stokes=True)
    
    ncorr = 4
    vis = compute_vis(skymodel, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=True,
                    pol_basis='linear', ra0=params.ra0, dec0=params.dec0)
    
    nrow = params.uvw.shape[0]
    nchan = params.freqs.size
    
    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], I + Q, atol=1e-6) # check that XX = I + Q
    np.testing.assert_allclose(vis[:, :, 1], U + 1j*V, atol=1e-6) # check that XY = U + iV
    np.testing.assert_allclose(vis[:, :, 2], U - 1j*V, atol=1e-6) # check that YX = U - iV
    np.testing.assert_allclose(vis[:, :, 3], I - Q, atol=1e-6) # check that YY = I - Q
    

def test_compute_vis_all_stokes_circular_basis(params):
    """
    Test compute_vis with ncorr == 4 and Stokes I, Q, U and V provided.
    Validates:
    - Output shape of visibilities
    - RR = I + V
    - RL = Q + iU
    - LR = Q - iU
    - LL = I - V
    """
    ncorr = 4
    # the numbers below are unphysical—they are just for testing the computation
    I = 1.0
    Q = 2.0
    U = 3.0
    V = 4.0
    sources = [
        CatSource(
            name = 'test_source',
            ra = f'{params.ra0}rad',
            dec = f'{params.dec0}rad',
            stokes_i = str(I),
            stokes_q = str(Q),
            stokes_u = str(U),
            stokes_v = str(V),
            )]

    skymodel = skymodel_from_sources(sources, params.freqs, full_stokes=True)
    
    ncorr = 4
    vis = compute_vis(skymodel, uvw=params.uvw, freqs=params.freqs, ncorr=ncorr, polarisation=True,
                    pol_basis='cicular', ra0=params.ra0, dec0=params.dec0)
    
    nrow = params.uvw.shape[0]
    nchan = params.freqs.size
    
    assert vis.shape == (nrow, nchan, ncorr)
    np.testing.assert_allclose(vis[:, :, 0], I + V, atol=1e-6) # check that RR = I + V
    np.testing.assert_allclose(vis[:, :, 1], Q + 1j*U, atol=1e-6) # check that RL = Q + iU
    np.testing.assert_allclose(vis[:, :, 2], Q - 1j*U, atol=1e-6) # check that LR = Q - iU
    np.testing.assert_allclose(vis[:, :, 3], I - V, atol=1e-6) # check that LL = I - V
