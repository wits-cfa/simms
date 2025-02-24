import unittest
import os
import logging
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from simms import BIN, get_logger
from simms.skymodel.skymods import process_fits_skymodel


log = get_logger(BIN.skysim)


class TestPredictFromFITS(unittest.TestCase):

    def test_fits_image_processing_1(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, no spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
        img_size = 256
        cell_size = 3e-6 # arcsec
        chan_freqs = np.array([1e9, 2e9, 3e9])
        ncorr = 2
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [img_size/2, img_size/2] # reference pixel
        wcs.wcs.crval = [0, 0] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((img_size, img_size))
        image[img_size//2, img_size//2] = 1.0 # put a point source at the center
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto('test.fits', overwrite=True)
        
        # process the FITS file
        intensities, _ = process_fits_skymodel('test.fits', 0, 0, chan_freqs, ncorr)
        
        # create expected intensities
        expected_intensities = np.empty((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[img_size//2, img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
    
    def test_fits_image_processing_2(self):
        """
        Tests if Stokes I only FITS file is processed correctly (off-centre point source, no spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
        img_size = 256
        cell_size = 3e-6 # arcsec
        chan_freqs = np.array([1e9, 2e9, 3e9])
        ncorr = 2
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600]) # pixel scale in deg
        wcs.wcs.crpix = [img_size/2, img_size/2] # reference pixel
        wcs.wcs.crval = [0, 0] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((img_size, img_size))
        
        # place a point source in a random pixel
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        rand_pix = np.random.randint(0, img_size)
        image[rand_pix, rand_pix] = 1.0
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto('test.fits', overwrite=True)
        
        # process the FITS file
        intensities, _ = process_fits_skymodel('test.fits', 0, 0, chan_freqs, ncorr)
        
        # create expected intensities
        expected_intensities = np.empty((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[rand_pix, rand_pix, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
        
    
    def test_fits_image_processing_3(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, spectral axis in FITS file)
        Validates:
            - output intensities shape
            - output intensities values
        """
        img_size = 256
        cell_size = 3e-6 # arcsec
        chan_freqs = np.array([1e9, 2e9, 3e9])
        nchan = chan_freqs.size
        ncorr = 2
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600, chan_freqs[1]-chan_freqs[0]]) # pixel scale in deg
        wcs.wcs.crpix = [img_size/2, img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0, 0, chan_freqs[0]] # reference pixel RA and Dec in deg
     
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((nchan, img_size, img_size))
        image[:, img_size//2, img_size//2] = 1.0 # put a point source at the center
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto('test.fits', overwrite=True)
        
        # process the FITS file
        intensities, _ = process_fits_skymodel('test.fits', 0, 0, chan_freqs, ncorr)
        
        # create expected intensities
        expected_intensities = np.empty((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[img_size//2, img_size//2, :, :] = 1.0
        # expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        intensities = intensities.reshape(img_size, img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
        
        
    def test_fits_image_processing_4(self):
        """
        Tests if Stokes I only FITS file is processed correctly with frequencies not matching MS channel frequencies
        Validates:
            - output intensities shape
            - output intensities values
        """
        img_size = 256
        cell_size = 3e-6 # arcsec
        chan_freqs = np.array([1.0e9, 1.5e9, 2.0e9])
        nchan = chan_freqs.size
        ncorr = 2
        
        # create a FITS file with Stokes I only
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
        wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600, 0.25e9]) # pixel scale in deg
        wcs.wcs.crpix = [img_size/2, img_size/2, 1] # reference pixel
        wcs.wcs.crval = [0, 0, chan_freqs[0]] # reference pixel RA and Dec in deg
        
        # make header
        header = wcs.to_header()
        header['BUNIT'] = 'Jy'
        
        # make image
        image = np.zeros((nchan, img_size, img_size))
        image[:, img_size//2, img_size//2] = 1.0
        
        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto('test.fits', overwrite=True)
        
        log.setLevel(logging.ERROR) # suppress warning messages
        # process the FITS file
        intensities, _ = process_fits_skymodel('test.fits', 0, 0, chan_freqs, ncorr)
        log.setLevel(logging.INFO) # reset logging level
        
        # create expected intensities
        expected_intensities = np.empty((img_size, img_size, chan_freqs.size, ncorr), dtype=np.complex128)
        expected_intensities[img_size//2, img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
        
    
    # TODO: Add more tests for all Stokes parameters and spectral axis in FITS file
    
    
    def test_fits_image_processing_5(self):
        """
        Tests if list FITS files each containing one Stokes parameter is processed correctly
        Validates:
            - output intensities shape
            - output intensities values
        """
        img_size = 256
        cell_size = 3e-6 # arcsec
        chan_freqs = np.array([1.0e9, 1.5e9, 2.0e9])
        nchan = chan_freqs.size
        ncorr = 4
        # I, Q, U, V = 1.0, 1.0, 1.0, 1.0
        stokes_params = [('I', 1.0), ('Q', 1.0), ('U', 1.0), ('V', 1.0)]

        test_fits_files = []
        for stokes in stokes_params:
            wcs = WCS(naxis=3)
            wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
            wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600, chan_freqs[1]-chan_freqs[0]]) # pixel scale in deg
            wcs.wcs.crpix = [img_size/2, img_size/2, 1] # reference pixel
            wcs.wcs.crval = [0, 0, chan_freqs[0]] # reference pixel RA and Dec in deg
        
            # make header
            header = wcs.to_header()
            header['BUNIT'] = 'Jy'
            
            # make image
            image = np.zeros((nchan, img_size, img_size))
            image[:, img_size//2, img_size//2] = stokes[1] # put a point source at the center
            
            # write to FITS file
            hdu = fits.PrimaryHDU(image, header=header)
            hdu.writeto(f'test_{stokes[0]}.fits', overwrite=True)
        
            test_fits_files.append(f'test_{stokes[0]}.fits')
            
        # process the FITS files
        intensities, _ = process_fits_skymodel(test_fits_files, 0, 0, chan_freqs, ncorr)
        
        # create expected intensities
        expected_intensities = np.empty((img_size, img_size, chan_freqs.size, ncorr), dtype=np.complex128)
        expected_intensities[img_size//2, img_size//2, :, 0] = stokes_params[0][1] + stokes_params[1][1]        # I + Q
        expected_intensities[img_size//2, img_size//2, :, 1] = stokes_params[2][1] + 1j*stokes_params[3][1]    # U + iV
        expected_intensities[img_size//2, img_size//2, :, 2] = stokes_params[2][1] - 1j*stokes_params[3][1]     # U - iV
        expected_intensities[img_size//2, img_size//2, :, 3] = stokes_params[0][1] - stokes_params[1][1]       # I - Q
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == expected_intensities.shape
        assert np.allclose(intensities, expected_intensities)

        # clean up
        for file in test_fits_files:
            os.remove(file)