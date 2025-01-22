import unittest
import os
from simms.skymodel.skymods import Source, Spectrum, computevis 
from simms.skymodel.converters import (
    convertdec2rad, 
    convertra2rad,
)
from simms.telescope.array_utilities import Array
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
import dask.array as da

class TestSkySim(unittest.TestCase):
    
    # set up inputs for the test
    test_array = Array('meerkat')
    uv_coverage_data = test_array.uvgen(
        pointing_direction = 'J2000,0deg,-30deg',
        dtime = 8,
        ntimes = 75,
        startfreq = 1293,
        dfreq = '206kHz',
        nchan = 16
    )
    uvw = uv_coverage_data['uvw']
    freqs = uv_coverage_data['freqs']
    
    source = Source(
        name = 'test_source',
        ra = convertra2rad('0h0m0s'),
        dec = convertdec2rad('-30d0m0s'),
        emaj = None,
        emin = None,
        pa = None,
    )
    source.l, source.m = source.radec2lm(source.ra, source.dec) # assuming the source is at phase centre
    
    # test ncorr == 2 and I and Q are provided
    def test_computevis_1(self):
        ncorr = 2
        I = 1.0
        Q = 1.0
        spectrum = Spectrum(
            stokes_i = I,
            stokes_q = Q,
            stokes_u = None,
            stokes_v = None,
            cont_reffreq = None,
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None
        )
        
        source.spectrum = spectrum.set_spectrum(self.freqs)
        sources = [source]
        
        vis = computevis(sources, self.uvw, self.freqs, ncorr, True)
        
        self.assertEqual(vis.shape, (75, 16, 2))
        self.assertTrue(np.all(vis[:, :, 0] == 2.0))
        self.assertTrue(np.all(vis[:, :, 1] == 2.0))
        