import unittest
from simms.skymodel.mstools import compute_vis
from simms.skymodel.source_factory import (
    CatSource,
    StokesData,
    contspec,
)
from simms.skymodel.skymods import skymodel_from_sources
from simms.telescope.array_utilities import Array
import numpy as np

class TestComputeVis(unittest.TestCase):
    
    def setUp(self):
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
        
        self.source = CatSource(
            name = 'test_source',
            ra = f'{self.ra0}rad',
            dec = f'{self.dec0}rad',
            emaj = None,
            emin = None,
            pa = None,
            stokes_i = "1",
            stokes_q = "0.00",
            stokes_u = "0.00",
            stokes_v = "0",
            line_peak = "1.4e9",
            line_width = "26kHz",
            line_restfreq = "1.4GHz",
            cont_coeff_1 = None,
            cont_coeff_2 = None,
            cont_reffreq = None,
            transient_start = None,
            transient_absorb = None,
            transient_period = None,
            transient_ingress = None,
        )
    

    def test_compute_vis_stokes_I_only(self):
        """
        Test that it stills works when only Stokes I is provided.
        Validates:
        - Output shape of visibilities
        - XX = I
        - XY = 0 (if ncorr == 4)
        - YX = 0 (if ncorr == 4)
        - YY = I
        """
        
        I = 1.0
    
        stokes = StokesData([I,0,0,0])
        
        stokes.set_spectrum(self.freqs, specfunc=contspec, full_pol=True,
                    **dict(coeff=0.0, nu_ref= 1.4e9))
                    
        self.source.stokes = stokes
        sources = [self.source]
        
        ncorr = 2

        vis = compute_vis(sources=sources, uvw=self.uvw, freqs=self.freqs, ncorr=ncorr, polarisation=False, pol_basis='linear', ra0=self.ra0, dec0=self.dec0)
        
        nrow = self.uvw.shape[0]
        nchan = self.freqs.size
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
        np.testing.assert_allclose(vis[:, :, 1], 1.0, atol=1e-6) # check that YY = I = 1
    
        ncorr = 4
        vis = compute_vis(sources=sources, uvw=self.uvw, freqs=self.freqs, ncorr=ncorr, polarisation=False, pol_basis='linear', ra0=self.ra0, dec0=self.dec0)
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
        np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
        np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
        np.testing.assert_allclose(vis[:, :, 3], 1.0, atol=1e-6) # check that YY = I = 1
        
    
#TODO(Mika, Senkhosi, Zaryn) Adapt rest of the tests to the new API
    # def test_compute_vis_I_and_Q_2_corrs(self):
    #     """
    #     Test compute_vis with ncorr == 2 and only Stokes I and Q provided.
    #     Validates:
    #     - Output shape of visibilities
    #     - XX = I + Q
    #     - YY = I - Q
    #     """
    #     ncorr = 2
        
    #     Q = 1.0
    #     self.source.stokes_q = Q
    #     sources = [self.source]
    #     skymodel = skymodel_from_sources(sources, self.freqs)
        
    #     vis = compute_vis(skymodel, self.uvw, self.freqs, ncorr, True, 'linear', self.ra0, self.dec0)
        
    #     nrow = self.uvw.shape[0]
    #     nchan = self.freqs.size
        
    #     self.assertEqual(vis.shape, (nrow, nchan, ncorr))
    #     np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
    #     np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that YY = I - Q = 0

#   
#   def test_compute_vis_I_and_Q_4_corrs(self):
#       """
#       Test compute_vis with ncorr == 4 and only Stokes I and Q provided.
#       Validates:
#       - Output shape of visibilities
#       - XX = I + Q
#       - XY = 0
#       - YX = 0
#       - YY = I - Q
#       """
#       ncorr = 4
#       I = 1.0
#       Q = 1.0
#       spectrum = Spectrum(
#           stokes_i = str(I),
#           stokes_q = str(Q),
#           stokes_u = None,
#           stokes_v = None,
#           cont_reffreq = None,
#           line_peak = None,
#           line_width = None,
#           line_restfreq = None,
#           cont_coeff_1 = None,
#           cont_coeff_2 = None
#       )
#       
#       self.source.spectrum = spectrum.make_spectrum(self.freqs)
#       sources = [self.source]
#       
#       vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'linear', None, None)
#       
#       nrow = self.uvw.shape[0]
#       nchan = self.freqs.size
#       
#       self.assertEqual(vis.shape, (nrow, nchan, ncorr))
#       np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
#       np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
#       np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
#       np.testing.assert_allclose(vis[:, :, 3], 0.0, atol=1e-6) # check that YY = I - Q = 0
#       
#   
#   def test_compute_vis_all_stokes_4_corrs_linear_basis(self):
#       """
#       Test compute_vis with ncorr == 4 and Stokes I, Q, U and V provided.
#       Validates:
#       - Output shape of visibilities
#       - XX = I + Q
#       - XY = U + iV
#       - YX = U - iV
#       - YY = I - Q
#       """
#       ncorr = 4
#       # the numbers below are unphysical—they are just for testing the computation
#       I = 1.0
#       Q = 2.0
#       U = 3.0
#       V = 4.0
#       spectrum = Spectrum(
#           stokes_i = str(I),
#           stokes_q = str(Q),
#           stokes_u = str(U),
#           stokes_v = str(V),
#           cont_reffreq = None,
#           line_peak = None,
#           line_width = None,
#           line_restfreq = None,
#           cont_coeff_1 = None,
#           cont_coeff_2 = None
#       )
#       
#       self.source.spectrum = spectrum.make_spectrum(self.freqs)
#       sources = [self.source]
#       
#       vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'linear', None, None)
#       
#       nrow = self.uvw.shape[0]
#       nchan = self.freqs.size
#       
#       self.assertEqual(vis.shape, (nrow, nchan, ncorr))
#       np.testing.assert_allclose(vis[:, :, 0], I + Q, atol=1e-6) # check that XX = I + Q
#       np.testing.assert_allclose(vis[:, :, 1], U + 1j*V, atol=1e-6) # check that XY = U + iV
#       np.testing.assert_allclose(vis[:, :, 2], U - 1j*V, atol=1e-6) # check that YX = U - iV
#       np.testing.assert_allclose(vis[:, :, 3], I - Q, atol=1e-6) # check that YY = I - Q
#   
#   
#   def test_compute_vis_all_stokes_4_corrs_circular_basis(self):
#       """
#       Test compute_vis with ncorr == 4 and Stokes I, Q, U and V provided.
#       Validates:
#       - Output shape of visibilities
#       - RR = I + V
#       - RL = Q + iU
#       - LR = Q - iU
#       - LL = I - V
#       """
#       ncorr = 4
#       # the numbers below are unphysical—they are just for testing the computation
#       I = 1.0
#       Q = 2.0
#       U = 3.0
#       V = 4.0
#       spectrum = Spectrum(
#           stokes_i = str(I),
#           stokes_q = str(Q),
#           stokes_u = str(U),
#           stokes_v = str(V),
#           cont_reffreq = None,
#           line_peak = None,
#           line_width = None,
#           line_restfreq = None,
#           cont_coeff_1 = None,
#           cont_coeff_2 = None
#       )
#       
#       self.source.spectrum = spectrum.make_spectrum(self.freqs)
#       sources = [self.source]
#       
#       vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'circular', None, None)
#       
#       nrow = self.uvw.shape[0]
#       nchan = self.freqs.size
#       
#       self.assertEqual(vis.shape, (nrow, nchan, ncorr))
#       np.testing.assert_allclose(vis[:, :, 0], I + V, atol=1e-6) # check that RR = I + V
#       np.testing.assert_allclose(vis[:, :, 1], Q + 1j*U, atol=1e-6) # check that RL = Q + iU
#       np.testing.assert_allclose(vis[:, :, 2], Q - 1j*U, atol=1e-6) # check that LR = Q - iU
#       np.testing.assert_allclose(vis[:, :, 3], I - V, atol=1e-6) # check that LL = I - V
