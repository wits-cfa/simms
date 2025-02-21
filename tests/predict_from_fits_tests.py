import unittest
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from simms.skymodel.skymods import process_fits_skymodel

class TestPredictFromFITS(unittest.TestCase):

    def test_fits_image_processing_1(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, no spectral axis in FITS file)
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
        expected_intensities = np.zeros((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[img_size//2, img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == (img_size * img_size, chan_freqs.size, ncorr)
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
    
    def test_fits_image_processing_2(self):
        """
        Tests if Stokes I only FITS file is processed correctly (off-centre point source, no spectral axis in FITS file)
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
        expected_intensities = np.zeros((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[rand_pix, rand_pix, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == (img_size * img_size, chan_freqs.size, ncorr)
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
        
        
    # TODO: Add more tests for all Stokes parameters and spectral axis in FITS file
    
    def test_fits_image_processing_3(self):
        """
        Tests if Stokes I only FITS file is processed correctly (centred point source, spectral axis in FITS file)
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
        expected_intensities = np.zeros((img_size, img_size, chan_freqs.size, ncorr))
        expected_intensities[img_size//2, img_size//2, :, :] = 1.0
        expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
        # compare the intensities with the original image
        assert intensities.shape == (img_size * img_size, chan_freqs.size, ncorr)
        assert np.allclose(intensities, expected_intensities)
        
        # clean up
        os.remove('test.fits')
        
        
    # def test_fits_image_processing_4(self):
    #     """
    #     Tests if Stokes I only FITS file is processed correctly with frequencies not matching MS channel frequencies
    #     """
    #     img_size = 256
    #     cell_size = 3e-6 # arcsec
    #     chan_freqs = np.array([1.0e9, 1.5e9, 2.0e9])
    #     nchan = chan_freqs.size
    #     ncorr = 2
        
    #     # create a FITS file with Stokes I only
    #     wcs = WCS(naxis=3)
    #     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
    #     wcs.wcs.cdelt = np.array([-cell_size/3600, cell_size/3600, 0.25e9]) # pixel scale in deg
    #     wcs.wcs.crpix = [img_size/2, img_size/2, 1] # reference pixel
    #     wcs.wcs.crval = [0, 0, chan_freqs[0]] # reference pixel RA and Dec in deg
        
    #     # make header
    #     header = wcs.to_header()
    #     header['BUNIT'] = 'Jy'
        
    #     # make image
    #     image = np.zeros((nchan, img_size, img_size))
    #     image[:, img_size//2, img_size//2] = 1.0
        
    #     # write to FITS file
    #     hdu = fits.PrimaryHDU(image, header=header)
    #     hdu.writeto('test.fits', overwrite=True)
        
    #     # process the FITS file
    #     intensities, _ = process_fits_skymodel('test.fits', 0, 0, chan_freqs, ncorr)
        
    #     # create expected intensities
    #     expected_intensities = np.zeros((img_size * img_size, chan_freqs.size, ncorr))
    #     expected_intensities[img_size//2, img_size//2, :, :] = 1.0
    #     expected_intensities = expected_intensities.reshape(img_size * img_size, chan_freqs.size, ncorr)
        
    #     # compare the intensities with the original image
    #     assert intensities.shape == (img_size * img_size, chan_freqs.size, ncorr)
    #     assert np.allclose(intensities, expected_intensities)
        
    #     # clean up
    #     os.remove('test.fits')