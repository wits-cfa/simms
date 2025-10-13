import unittest
from simms.skymodel.mstools import compute_vis
from simms.skymodel.source_factory import (
    CatSource,
    StokesData,
    exoplanet_transient_logistic,
)
from simms.telescope.array_utilities import Array
import numpy as np

class TestComputeVis(unittest.TestCase):
    
    def setUp(self):
        # set up test inputs
        test_array = Array('meerkat')
        
        
        uv_coverage_data = test_array.uvgen(
            pointing_direction = 'J2000,0deg,-30deg'.split(','),
            dtime = 8,
            ntimes = 100,
            start_freq = '1293MHz',
            dfreq = '206kHz',
            nchan = 16
        )
        self.uvw = uv_coverage_data.uvw
        self.freqs = uv_coverage_data.freqs
        self.ntimes = uv_coverage_data.ntimes
        
        self.ra0 = uv_coverage_data.ra0
        self.dec0 = uv_coverage_data.dec0
        
        self.source = CatSource(
            name = 'test_source',
            ra = f'{self.ra0}rad',
            dec = f'{self.dec0}rad',
            emaj = None,
            emin = None,
            pa = None,
        )
        
        self.source.add_stokes(
            stokes_i = "1",
            stokes_q = "0.00",
            stokes_u = "0.00",
            stokes_v = "0",
        )
        
        self.source.add_spectral(
            line_peak = None,
            line_width = None,
            line_restfreq = None,
            cont_coeff_1 = None,
            cont_coeff_2 = None,
            cont_reffreq = None,
        )

        self.source.add_lightcurve(
            transient_start = "100",
            transient_absorb = "0.5",
            transient_period = "100",
            transient_ingress = "20",
        )
    
    def test_transient_event(self):\
        I = 1.0
        t0 = 
        unique_times_rel = unique_times - t0
    
        stokes = StokesData([I,0,0,0])

        stokes.set_lightcurve(lightcurve_func=exoplanet_transient_logistic, **dict(start_time=)

        
            unique_times_rel = unique_times - t0
            kwargs = {
                "start_time": unique_times_rel.min(),
                "end_time": unique_times_rel.max(),
                "ntimes": unique_times_rel.shape[0],
                "transient_start": src.transient_start,  
                "transient_period": src.transient_period,
                "transient_ingress": src.transient_ingress,
                "transient_absorb": src.transient_absorb
            }