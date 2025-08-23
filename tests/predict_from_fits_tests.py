import unittest
import uuid
import os
import logging
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from africanus.dft import im_to_vis
from simms import BIN, get_logger
from simms.telescope.array_utilities import Array
from simms.skymodel.skymods import skymodel_from_fits


log = get_logger(BIN.skysim)


class TestPredictFromFITS(unittest.TestCase):
    
    def setUp(self):
        """"Set up test inputs."""
        self.nchan = 16
        self.n_times = 75
        self.n_baselines = 64*63/2
        
        test_array = Array('meerkat')
        uv_coverage_data = test_array.uvgen(
            pointing_direction = 'J2000,0deg,-30deg'.split(','),
            dtime = 8,
            ntimes = self.n_times,
            start_freq = '1293MHz',
            dfreq = '206kHz',
            nchan = self.nchan
        )
        self.uvw = uv_coverage_data.uvw
        self.freqs = uv_coverage_data.freqs
        self.ra0 = 0.0
        self.dec0 = np.deg2rad(-30.0)
        self.ncorr = 2

        # image parameters
        self.img_size = 256
        self.cell_size = 3e-6  # arcsec
        
        # Store temporary files to be cleaned up
        self.test_files = []
        
        # Set up logging level
        self.original_log_level = log.level
        
    
    def tearDown(self):
        """Clean up after each test method runs."""
        # Remove any temporary files created
        for file in self.test_files:
            if os.path.exists(file):
                os.remove(file)
        
        # Reset logging level
        log.setLevel(self.original_log_level)
        
    
    def test_fits_predict_stokes_I(self):
        """
        Test visibility prediction from only a FITS sky model, ncorr = 2
        Validates:
            - Output shape of visibilities
            - XX = I
            - YY = I
        """
        I = 1.0
        
        # create a FITS sky model
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2] # reference pixel
        wcs.wcs.crval = [np.rad2deg(self.ra0), np.rad2deg(self.dec0)] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.img_size, self.img_size))
        image[self.img_size//2, self.img_size//2] = I
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        brightness_matrix, lm, _, _, _, _ = skymodel_from_fits(test_filename, self.ra0, self.dec0, self.freqs, self.freqs[1]-self.freqs[0], self.ncorr, 'linear')
        
        # predict visibilities
        vis = im_to_vis(brightness_matrix, self.uvw, lm, self.freqs)
        
        # check the output
        assert vis.shape == (self.n_baselines * self.n_times, self.nchan, self.ncorr)
        assert np.allclose(vis[:, :, 0], I, atol=1e-6)
        assert np.allclose(vis[:, :, 1], I, atol=1e-6)
        
    
    def test_fits_predict_stokes_I_with_spectral_axis(self):
        """
        Test visibility prediction from only a FITS sky model with a spectral axis, ncorr = 2
        Validates:
            - Output shape of visibilities
            - XX = I
            - YY = I
        """
        I = 1.0
        
        # create a FITS sky model
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.freqs[1]-self.freqs[0]]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [np.rad2deg(self.ra0), np.rad2deg(self.dec0), self.freqs[0]] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.nchan, self.img_size, self.img_size))
        image[:, self.img_size//2, self.img_size//2] = I
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        log.setLevel(logging.ERROR)
        brightness_matrix, lm, _, _, _, _ = skymodel_from_fits(test_filename, self.ra0, self.dec0, self.freqs, self.freqs[1]-self.freqs[0], self.ncorr, 'linear')
        
        # predict visibilities
        vis = im_to_vis(brightness_matrix, self.uvw, lm, self.freqs)
        
        # check the output
        assert vis.shape == (self.n_baselines * self.n_times, self.nchan, self.ncorr)
        assert np.allclose(vis[:, :, 0], I, atol=1e-6)
        assert np.allclose(vis[:, :, 1], I, atol=1e-6)
    
    
    def test_fits_predicting_all_stokes_linear_basis(self):
        """
        Test visibility prediction from FITS images of all Stokes parameters, ncorr = 4
        Validates:
            - Output shape of visibilities
            - XX = I + Q
            - XY = U + iV
            - YX = U - iV
            - YY = I - Q
        """
        self.ncorr = 4
        # the numbers below are unphysical—they are just for testing the computation
        stokes_params = [('I', 1.0), ('Q', 2.0), ('U', 3.0), ('V', 4.0)]

        test_skymodels = []
        for stokes in stokes_params:
            wcs = WCS(naxis=3)
            wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
            wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.freqs[1]-self.freqs[0]]) # pixel scale in deg
            wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
            wcs.wcs.crval = [np.rad2deg(self.ra0), np.rad2deg(self.dec0), self.freqs[0]] # reference pixel RA and Dec in deg
        
            # make header
            header = wcs.to_header()
            header['BUNIT'] = 'Jy'
            
            # make image
            image = np.zeros((self.nchan, self.img_size, self.img_size))
            image[:, self.img_size//2, self.img_size//2] = stokes[1] # put a point source at the center
            
            # write to FITS file
            hdu = fits.PrimaryHDU(image, header=header)
            test_filename = f'test_{uuid.uuid4()}_{stokes[0]}.fits'
            self.test_files.append(test_filename)
            hdu.writeto(test_filename, overwrite=True)
        
            test_skymodels.append(test_filename)
            
        # process the FITS files
        brightness_matrix, lm, _, _, _, _ = skymodel_from_fits(test_skymodels, self.ra0, self.dec0, self.freqs, self.freqs[1]-self.freqs[0], self.ncorr, 'linear')
        
        # predict visibilities
        vis = im_to_vis(brightness_matrix, self.uvw, lm, self.freqs)
        
        # check the output
        assert vis.shape == (self.n_baselines * self.n_times, self.nchan, self.ncorr)
        assert np.allclose(vis[:, :, 0], stokes_params[0][1] + stokes_params[1][1], atol=1e-6)      # I + Q
        assert np.allclose(vis[:, :, 1], stokes_params[2][1] + 1j*stokes_params[3][1], atol=1e-6)   # U + iV
        assert np.allclose(vis[:, :, 2], stokes_params[2][1] - 1j*stokes_params[3][1], atol=1e-6)   # U - iV
        assert np.allclose(vis[:, :, 3], stokes_params[0][1] - stokes_params[1][1], atol=1e-6)      # I - Q

    def test_fits_predicting_all_stokes_circular_basis(self):
        """
        Test visibility prediction from FITS images of all Stokes parameters, ncorr = 4
        Validates:
            - Output shape of visibilities
            - RR = I + V
            - RL = Q + iU
            - LR = Q - iU
            - LL = I - V
        """
        self.ncorr = 4
        # the numbers below are unphysical—they are just for testing the computation
        stokes_params = [('I', 1.0), ('Q', 2.0), ('U', 3.0), ('V', 4.0)]

        test_skymodels = []
        for stokes in stokes_params:
            wcs = WCS(naxis=3)
            wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
            wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.freqs[1]-self.freqs[0]]) # pixel scale in deg
            wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
            wcs.wcs.crval = [np.rad2deg(self.ra0), np.rad2deg(self.dec0), self.freqs[0]] # reference pixel RA and Dec in deg
        
            # make header
            header = wcs.to_header()
            header['BUNIT'] = 'Jy'
            
            # make image
            image = np.zeros((self.nchan, self.img_size, self.img_size))
            image[:, self.img_size//2, self.img_size//2] = stokes[1] # put a point source at the center
            
            # write to FITS file
            hdu = fits.PrimaryHDU(image, header=header)
            test_filename = f'test_{uuid.uuid4()}_{stokes[0]}.fits'
            self.test_files.append(test_filename)
            hdu.writeto(test_filename, overwrite=True)
        
            test_skymodels.append(test_filename)
            
        # process the FITS files
        brightness_matrix, lm, _, _, _, _ = skymodel_from_fits(test_skymodels, self.ra0, self.dec0, self.freqs, self.freqs[1]-self.freqs[0], self.ncorr, 'circular')
        
        # predict visibilities
        vis = im_to_vis(brightness_matrix, self.uvw, lm, self.freqs)
        
        # check the output
        assert vis.shape == (self.n_baselines * self.n_times, self.nchan, self.ncorr)
        assert np.allclose(vis[:, :, 0], stokes_params[0][1] + stokes_params[3][1], atol=1e-6)      # I + V
        assert np.allclose(vis[:, :, 1], stokes_params[1][1] + 1j*stokes_params[2][1], atol=1e-6)   # Q + iU
        assert np.allclose(vis[:, :, 2], stokes_params[1][1] - 1j*stokes_params[2][1], atol=1e-6)   # Q - iU
        assert np.allclose(vis[:, :, 3], stokes_params[0][1] - stokes_params[3][1], atol=1e-6)      # I - V
