import unittest
import os
from simms.skymodel.skymods import Source, Spectrum, compute_vis 
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
        
        self.source = Source(
            name = 'test_source',
            ra = '0h0m0s',
            dec = '-30d0m0s',
            emaj = None,
            emin = None,
            pa = None,
        )
        self.source.set_lm(self.source.ra, self.source.dec) # assuming the source is at phase centre
    
    
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
        spectrum = Spectrum(
            stokes_i = str(I),
            stokes_q = None,
            stokes_u = None,
            stokes_v = None,
            cont_reffreq = None,
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None
        )
        
        self.source.spectrum = spectrum.make_spectrum(self.freqs)
        sources = [self.source]
        
        ncorr = 2
        vis = compute_vis(sources, self.uvw, self.freqs, ncorr, False, 'linear', None, None)
        
        nrow = self.uvw.shape[0]
        nchan = self.freqs.size
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
        np.testing.assert_allclose(vis[:, :, 1], 1.0, atol=1e-6) # check that YY = I = 1
    
        ncorr = 4
        vis = compute_vis(sources, self.uvw, self.freqs, ncorr, False, 'linear', None, None)
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 1.0, atol=1e-6) # check that XX = I = 1
        np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
        np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
        np.testing.assert_allclose(vis[:, :, 3], 1.0, atol=1e-6) # check that YY = I = 1
        
    
    def test_compute_vis_I_and_Q_2_corrs(self):
        """
        Test compute_vis with ncorr == 2 and only Stokes I and Q provided.
        Validates:
        - Output shape of visibilities
        - XX = I + Q
        - YY = I - Q
        """
        ncorr = 2
        I = 1.0
        Q = 1.0
        spectrum = Spectrum(
            stokes_i = str(I),
            stokes_q = str(Q),
            stokes_u = None,
            stokes_v = None,
            cont_reffreq = None,
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None
        )
        
        self.source.spectrum = spectrum.make_spectrum(self.freqs)
        sources = [self.source]
        
        vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'linear', None, None)
        
        nrow = self.uvw.shape[0]
        nchan = self.freqs.size
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
        np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that YY = I - Q = 0

    
    def test_compute_vis_I_and_Q_4_corrs(self):
        """
        Test compute_vis with ncorr == 4 and only Stokes I and Q provided.
        Validates:
        - Output shape of visibilities
        - XX = I + Q
        - XY = 0
        - YX = 0
        - YY = I - Q
        """
        ncorr = 4
        I = 1.0
        Q = 1.0
        spectrum = Spectrum(
            stokes_i = str(I),
            stokes_q = str(Q),
            stokes_u = None,
            stokes_v = None,
            cont_reffreq = None,
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None
        )
        
        self.source.spectrum = spectrum.make_spectrum(self.freqs)
        sources = [self.source]
        
        vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'linear', None, None)
        
        nrow = self.uvw.shape[0]
        nchan = self.freqs.size
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
        np.testing.assert_allclose(vis[:, :, 1], 0.0, atol=1e-6) # check that XY = 0
        np.testing.assert_allclose(vis[:, :, 2], 0.0, atol=1e-6) # check that YX = 0
        np.testing.assert_allclose(vis[:, :, 3], 0.0, atol=1e-6) # check that YY = I - Q = 0
        
    
    def test_compute_vis_all_stokes_4_corrs(self):
        """
        Test compute_vis with ncorr == 4 and Stokes I, Q, U and V provided.
        Validates:
        - Output shape of visibilities
        - XX = I + Q
        - XY = U + iV
        - YX = U - iV
        - YY = I - Q
        """
        ncorr = 4
        I = 1.0
        Q = 1.0
        U = 1.0
        V = 1.0
        spectrum = Spectrum(
            stokes_i = str(I),
            stokes_q = str(Q),
            stokes_u = str(U),
            stokes_v = str(V),
            cont_reffreq = None,
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None
        )
        
        self.source.spectrum = spectrum.make_spectrum(self.freqs)
        sources = [self.source]
        
        vis = compute_vis(sources, self.uvw, self.freqs, ncorr, True, 'linear', None, None)
        
        nrow = self.uvw.shape[0]
        nchan = self.freqs.size
        
        self.assertEqual(vis.shape, (nrow, nchan, ncorr))
        np.testing.assert_allclose(vis[:, :, 0], 2.0, atol=1e-6) # check that XX = I + Q = 2
        np.testing.assert_allclose(np.abs(vis[:, :, 1]), np.sqrt(2.0), atol=1e-6) # check that XY = U + iV = sqrt(2)
        np.testing.assert_allclose(np.abs(vis[:, :, 2]), np.sqrt(2.0), atol=1e-6) # check that YX = U - iV = sqrt(2)
        np.testing.assert_allclose(vis[:, :, 3], 0.0, atol=1e-6) # check that YY = I - Q = 0