import unittest
import os
import logging
import uuid
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from simms import BIN, get_logger
from simms.skymodel.skymods import process_fits_skymodel
from simms.utilities import FITSSkymodelError as SkymodelError


log = get_logger(BIN.skysim)


class TestFITSProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up common test parameters before each test method runs."""
        # Common parameters used across tests
        self.img_size = 256
        self.cell_size = 3e-6  # arcsec
        self.chan_freqs = np.array([1e9, 2e9, 3e9])
        self.nchan = len(self.chan_freqs)
        self.ncorr = 2
        self.ms_delta_nu = self.chan_freqs[1] - self.chan_freqs[0]
        self.basis = 'linear'
        self.tol = 1e-9
        
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
    
    
    def test_stokes_I_fits_processing(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, no spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2] # reference pixel
        wcs.wcs.crval = [0, 0] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.img_size, self.img_size))
        image[self.img_size//2, self.img_size//2] = 1.0 # put a point source at the center
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
    
    
    def test_off_centre_stokes_I_processing(self):
        """
        Tests if Stokes I only FITS file is processed correctly (off-centre point source, no spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
            
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2] # reference pixel
        wcs.wcs.crval = [0, 0] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.img_size, self.img_size))
        
        # place a point source in a random pixel
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        rand_pix = np.random.randint(0, self.img_size)
        image[rand_pix, rand_pix] = 1.0
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[rand_pix, rand_pix, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
    
    def test_stokes_I_with_spectral_axis_processing(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
            
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0]]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0, 0, self.chan_freqs[0]] # reference pixel RA and Dec in deg
    
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.nchan, self.img_size, self.img_size))
        image[:, self.img_size//2, self.img_size//2] = 1.0 # put a point source at the center
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)

             
    def test_stokes_I_with_freq_interp_processing(self):
        """
        Tests if Stokes I only FITS file is processed correctly with frequencies not matching MS channel frequencies
        Validates:
            - output intensities shape
            - output intensities values
        """

        # we make the FITS frequencies [0.5e9, 1.5e9, 2.5e9, 3.5e9]
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, 1e9]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0, 0, self.chan_freqs[0] - 0.5e9] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((4, self.img_size, self.img_size))
        image[:, self.img_size//2, self.img_size//2] = 1.0
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        log.setLevel(logging.ERROR) # suppress warning messages
        # process the FITS file
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        
    def test_stokes_I_processing_with_interp_bounds_error(self):
        """
        Test that the frequency interpolation raises an error when the FITS frequency axis doesn't
        cover the full range of the MS frequency axis.
        Validates:
            - error message
        """
        # we make the FITS frequencies [1e9, 2e9]
        
         # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0]]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0.0, -30.0, self.chan_freqs[0]] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.nchan - 1, self.img_size, self.img_size))
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        log.setLevel(logging.ERROR)
        # process the FITS file
        with self.assertRaises(SkymodelError):
            process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
    
    
    def test_full_stokes_fits_list_processing(self):
        """
        Tests if list FITS files each containing one Stokes parameter is processed correctly
        Validates:
            - output intensities shape
            - output intensities values
        """
        
        self.ncorr = 4
        stokes_params = [('I', 1.0), ('Q', 1.0), ('U', 1.0), ('V', 1.0)]

        test_skymodels = []
        for stokes in stokes_params:
            wcs = WCS(naxis=3)
            wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
            wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0]]) # pixel scale in deg
            wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
            wcs.wcs.crval = [0, 0, self.chan_freqs[0]] # reference pixel RA and Dec in deg
        
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
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_skymodels, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr), dtype=np.complex128)
        expected_intensities[self.img_size//2, self.img_size//2, :, 0] = stokes_params[0][1] + stokes_params[1][1]        # I + Q
        expected_intensities[self.img_size//2, self.img_size//2, :, 1] = stokes_params[2][1] + 1j*stokes_params[3][1]    # U + iV
        expected_intensities[self.img_size//2, self.img_size//2, :, 2] = stokes_params[2][1] - 1j*stokes_params[3][1]     # U - iV
        expected_intensities[self.img_size//2, self.img_size//2, :, 3] = stokes_params[0][1] - stokes_params[1][1]       # I - Q
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        
    # TODO: Modify test below to check use of second element of Stokes axis
    # def test_stokes_axis_in_fits_processing(self):
    #     """
    #     Tests that the code raises an error when a FITS file contains a Stokes axis
    #     Validates:
    #         - error message
    #     """
    #     # create a FITS file with Stokes ndim > 1
    #     wcs = WCS(naxis=4)
    #     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    #     wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0], 1.0]) # pixel scale in deg
    #     wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1, 1] # reference pixel
    #     wcs.wcs.crval = [0.0, -30.0, self.chan_freqs[0], 1] # reference pixel RA and Dec in deg
        
    #     # make header
    #     header = wcs.to_header()
    #     header['BUNIT'] = 'Jy'
        
    #     # make image
    #     image = np.ones((4, self.nchan, self.img_size, self.img_size))
        
    #     # write to FITS file
    #     hdu = fits.PrimaryHDU(image, header=header)
    #     test_filename = f'test_{uuid.uuid4()}.fits'
    #     self.test_files.append(test_filename)
    #     hdu.writeto(test_filename, overwrite=True)
        
    #     # process the FITS file
    #     with self.assertRaises(SkymodelError):
    #         process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
    
    # TODO: Modify test below to check use of only first element of temporal axis
    # def test_time_axis_fits_processing(self):
    #     """
    #     Tests that the code raises an error when a FITS file contains a temporal axis
    #     Validates:
    #         - error message
    #     """
    #     # create a FITS file with time axis
    #     wcs = WCS(naxis=4)
    #     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'TIME']
    #     wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0], 1.0]) # pixel scale in deg
    #     wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1, 1] # reference pixel
    #     wcs.wcs.crval = [0.0, -30.0, self.chan_freqs[0], 1] # reference pixel RA and Dec in deg
        
    #     # make header
    #     header = wcs.to_header()
    #     header['BUNIT'] = 'Jy'
        
    #     # make image
    #     image = np.ones((2, self.nchan, self.img_size, self.img_size))
        
    #     # write to FITS file
    #     hdu = fits.PrimaryHDU(image, header=header)
    #     test_filename = f'test_{uuid.uuid4()}.fits'
    #     self.test_files.append(test_filename)
    #     hdu.writeto(test_filename, overwrite=True)
        
    #     # process the FITS file
    #     with self.assertRaises(SkymodelError):
    #         process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
    
        
    def test_stokes_I_processing_with_heinous_axis_ordering(self):
        """
        Tests if Stokes I FITS file with spectral axis pocessing.
        """
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'FREQ' , 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0], self.cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, 1, self.img_size/2] # reference pixel
        wcs.wcs.crval = [0, self.chan_freqs[0], 0] # reference pixel RA and Dec in deg
    
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
    
        # make image
        image = np.zeros((self.img_size, self.nchan, self.img_size))
        image[self.img_size//2, :, self.img_size//2] = 1.0 # put a point source at the center
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        intensities, _, _, _, _, _, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        non_zero_mask = np.any(expected_intensities > self.tol, axis=(1, 2))
        expected_intensities = expected_intensities[non_zero_mask]
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
    
    def test_lm_grid_creation_with_stokes_I_only(self):
        """
        Tests if the l-m grid is created correctly for Stokes I only FITS file (no spectral axis in FITS file)
        Validates:
            - output l-m grid shape
            - output l-m grid values
        """
    
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2] # reference pixel
        wcs.wcs.crval = [0.0, -30.0] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.ones((self.img_size, self.img_size))
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        _, lm, _, _, _, _, _ = process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # created expected l-m grid
        delt = np.deg2rad(self.cell_size/3600)
        l = np.sort(np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt)
        m = np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt
        ll, mm = np.meshgrid(l, m)
        expected_lm = np.stack([ll, mm], axis=-1)
        
        # validate the l-m grid
        assert lm.shape == expected_lm.shape
        assert np.allclose(lm, expected_lm)
        
        # clean up
        os.remove(test_filename)
        
        
    def test_lm_grid_creation_with_stokes_I_and_spectral_axis(self):
        """
        Tests if the l-m grid is created correctly for Stokes I only FITS file with spectral axis
        Validates:
            - output l-m grid shape
            - output l-m grid values
        """
    
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, self.chan_freqs[1]-self.chan_freqs[0]]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0.0, -30.0, self.chan_freqs[0]] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.ones((self.nchan, self.img_size, self.img_size))
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        _, lm, _, _, _, _, _ = process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ms_delta_nu, self.ncorr, self.basis)
        
        # created expected l-m grid
        delt = np.deg2rad(self.cell_size/3600)
        l = np.sort(np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt)
        m = np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt
        ll, mm = np.meshgrid(l, m)
        expected_lm = np.stack([ll, mm], axis=-1)
        
        # validate the l-m grid
        assert lm.shape == expected_lm.shape
        assert np.allclose(lm, expected_lm)
        
        # clean up
        os.remove(test_filename)