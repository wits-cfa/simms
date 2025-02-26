import unittest
import os
import logging
import uuid
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from simms import BIN, get_logger
from simms.skymodel.skymods import process_fits_skymodel


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
        intensities, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ncorr)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        
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
        intensities, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ncorr)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[rand_pix, rand_pix, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        
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
        intensities, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ncorr)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        
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

        self.chan_freqs = np.array([1.0e9, 1.5e9, 2.0e9])
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-self.cell_size/3600, self.cell_size/3600, 0.25e9]) # pixel scale in deg
        wcs.wcs.crpix = [self.img_size/2, self.img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0, 0, self.chan_freqs[0]] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((self.nchan, self.img_size, self.img_size))
        image[:, self.img_size//2, self.img_size//2] = 1.0
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        self.test_files.append(test_filename)
        hdu.writeto(test_filename, overwrite=True)
        
        log.setLevel(logging.ERROR) # suppress warning messages
        # process the FITS file
        intensities, _ = process_fits_skymodel(test_filename, 0, 0, self.chan_freqs, self.ncorr)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr))
        expected_intensities[self.img_size//2, self.img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
    
    
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
        intensities, _ = process_fits_skymodel(test_skymodels, 0, 0, self.chan_freqs, self.ncorr)
        
        
        # create expected intensities
        expected_intensities = np.zeros((self.img_size, self.img_size, self.chan_freqs.size, self.ncorr), dtype=np.complex128)
        expected_intensities[self.img_size//2, self.img_size//2, :, 0] = stokes_params[0][1] + stokes_params[1][1]        # I + Q
        expected_intensities[self.img_size//2, self.img_size//2, :, 1] = stokes_params[2][1] + 1j*stokes_params[3][1]    # U + iV
        expected_intensities[self.img_size//2, self.img_size//2, :, 2] = stokes_params[2][1] - 1j*stokes_params[3][1]     # U - iV
        expected_intensities[self.img_size//2, self.img_size//2, :, 3] = stokes_params[0][1] - stokes_params[1][1]       # I - Q
        expected_intensities = expected_intensities.reshape(self.img_size * self.img_size, self.chan_freqs.size, self.ncorr)
        
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
        image = np.zeros((self.img_size, self.img_size))
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = f'test_{uuid.uuid4()}.fits'
        hdu.writeto(test_filename, overwrite=True)
        
        # process the FITS file
        _, lm = process_fits_skymodel(test_filename, 0.0, np.deg2rad(-30.0), self.chan_freqs, self.ncorr)
        
        # created expected l-m grid
        delt = np.deg2rad(self.cell_size/3600)
        l = np.sort(np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt)
        m = np.arange(1-self.img_size/2, 1-self.img_size/2+self.img_size) * delt
        ll, mm = np.meshgrid(l, m)
        expected_lm = np.vstack((ll.flatten(), mm.flatten())).T
        
        # validate the l-m grid
        assert lm.shape == expected_lm.shape
        assert np.allclose(lm, expected_lm)
        
        # clean up
        os.remove(test_filename)