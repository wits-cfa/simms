import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from simms.exceptions import FITSSkymodelError as SkymodelError
from simms.skymodel.fits_skies import skymodel_from_fits

from . import InitTest


class InitThisTest(InitTest):
    def __init__(self):
        """Set up common test parameters before each test method runs."""
        # Common parameters used across tests
        self.img_size = 256
        self.cell_size = 2  # arcsec
        self.chan_freqs = np.array([0.9e9, 1e9, 1.1e9])
        self.nchan = len(self.chan_freqs)
        self.ncorr = 2
        self.ms_delta_nu = self.chan_freqs[1] - self.chan_freqs[0]
        self.linear_basis = True
        self.tol = 1e-7

        # Store temporary files to be cleaned up
        self.test_files = []


@pytest.fixture
def params():
    return InitThisTest()


def test_stokes_I_fits_processing(params):
    """
    Tests if Stokes I only FITS file is processed correctly (centred point source, no spectral axis in FITS file)
    Validates:
        - output intensities shape
        - output intensities values
    """
    # create a FITS file with Stokes I only
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array([-params.cell_size / 3600, params.cell_size / 3600, 1e9, 1.0])
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[1], 1.0]

    # make header
    header = wcs.to_header()
    header["BUNIT"] = "Jy"

    # make image
    image = np.zeros((1, 1, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0  # put a point source at the center

    # write to FITS file
    hdu = fits.PrimaryHDU(image, header=header)
    test_filename = params.random_named_file(suffix=".fits")
    hdu.writeto(test_filename, overwrite=True)

    # process the FITS file
    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image

    # create expected intensities
    expected_intensities = np.zeros((params.img_size, params.img_size, 1, params.ncorr))
    expected_intensities[params.img_size // 2, params.img_size // 2, 0, :] = 1.0
    expected_intensities = expected_intensities.reshape(params.img_size * params.img_size, 1, params.ncorr)
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    # compare the intensities with the original image
    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities)


def test_off_centre_stokes_I_processing(params):
    """
    Tests if Stokes I only FITS file is processed correctly (off-centre point source, no spectral axis in FITS file)
    Validates:
        - output intensities shape
        - output intensities values
    """

    # create a FITS file with Stokes I only
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    # make header
    header = wcs.to_header()
    header["BUNIT"] = "Jy"

    # make image
    image = np.zeros((1, params.nchan, params.img_size, params.img_size))

    # place a point source in a random pixel
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    rand_pix = np.random.randint(0, params.img_size)
    image[:, :, rand_pix, rand_pix] = 1.0

    # write to FITS file
    hdu = fits.PrimaryHDU(image, header=header)
    test_filename = params.random_named_file(suffix=".fits")
    hdu.writeto(test_filename, overwrite=True)

    # process the FITS file
    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image

    # create expected intensities
    expected_intensities = np.zeros((params.img_size, params.img_size, params.chan_freqs.size, params.ncorr))
    expected_intensities[rand_pix, rand_pix, :, :] = 1.0
    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    # compare the intensities with the original image
    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities)


def test_stokes_I_with_spectral_axis_processing(params):
    """
    Tests if Stokes I only FITS file is processed correctly (centred point source, spectral axis in FITS file)
    Validates:
        - output intensities shape
        - output intensities values
    """

    # create a FITS file with Stokes I only
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    # make header
    header = wcs.to_header()
    header["BUNIT"] = "Jy"

    # make image
    image = np.zeros((1, params.nchan, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0  # put a point source at the center

    # write to FITS file
    hdu = fits.PrimaryHDU(image, header=header)
    test_filename = params.random_named_file(suffix=".fits")
    hdu.writeto(test_filename, overwrite=True)

    # process the FITS file
    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image

    # create expected intensities
    expected_intensities = np.zeros((params.img_size, params.img_size, params.chan_freqs.size, params.ncorr))
    expected_intensities[params.img_size // 2, params.img_size // 2, :, :] = 1.0
    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    # compare the intensities with the original image
    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities)


def test_stokes_I_with_freq_interp_processing(params):
    """
    Tests if Stokes I only FITS file is processed correctly with frequencies not matching MS channel frequencies
    Validates:
        - output intensities shape
        - output intensities values
    """

    # we make the FITS frequencies [0.5e9, 1.5e9, 2.5e9, 3.5e9]

    # create a FITS file with Stokes I only
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array([-params.cell_size / 3600, params.cell_size / 3600, 1e9, 1.0])  # pixel scale in deg
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]  # reference pixel
    wcs.wcs.crval = [0, 0, params.chan_freqs[0] + 0.1e9, 1.0]  # reference pixel RA and Dec in deg

    # make header
    header = wcs.to_header()
    header["BUNIT"] = "Jy"

    # make image
    image = np.zeros((1, 4, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0

    # write to FITS file
    hdu = fits.PrimaryHDU(image, header=header)
    test_filename = params.random_named_file(suffix=".fits")
    hdu.writeto(test_filename, overwrite=True)

    predict = skymodel_from_fits(
        test_filename,
        0,
        0,
        params.chan_freqs,
        params.ms_delta_nu,
        params.ncorr,
        linear_basis=params.linear_basis,
        interpolation="linear",
    )
    intensities = predict.image

    # create expected intensities
    expected_intensities = np.zeros((params.img_size, params.img_size, params.chan_freqs.size, params.ncorr))
    expected_intensities[params.img_size // 2, params.img_size // 2, :, :] = 1.0
    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    # compare the intensities with the original image
    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities)


def test_stokes_I_processing_with_interp_bounds_error(params):
    """
    Test that the frequency interpolation raises an error when the FITS frequency axis doesn't
    cover the full range of the MS frequency axis.
    Validates:
        - error message
    Note:
    This test triggers a SkymodelError before the FITS dataset is closed,
    which causes a ResourceWarning about an unclosed file. This is expected
    and harmless for this test.
    """
    # we make the FITS frequencies [1e9, 2e9]

    # create a FITS file with Stokes I only
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    # make header
    header = wcs.to_header()
    header["BUNIT"] = "Jy"

    # make image
    image = np.zeros((1, params.nchan - 1, params.img_size, params.img_size))

    # write to FITS file
    hdu = fits.PrimaryHDU(image, header=header)
    test_filename = params.random_named_file(suffix=".fits")
    print(f"test_stokes_I_processing_with_interp_bounds_error created the file: {test_filename}")
    hdu.writeto(test_filename, overwrite=True)

    # process the FITS file
    with pytest.raises(SkymodelError) as exception:
        skymodel_from_fits(
            test_filename,
            0.0,
            np.deg2rad(-30.0),
            params.chan_freqs,
            params.ms_delta_nu,
            params.ncorr,
            linear_basis=params.linear_basis,
        )

    assert exception.type is SkymodelError


def test_full_stokes_fits_list_processing(params):
    """
    Tests if list FITS files each containing one Stokes parameter is processed correctly
    Validates:
        - output intensities shape
        - output intensities values
    """

    params.ncorr = 4
    stokes_params = [("I", 1.0), ("Q", 1.0), ("U", 1.0), ("V", 1.0)]

    test_skymodels = []
    for stokes in stokes_params:
        wcs = WCS(naxis=4)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
        wcs.wcs.cdelt = np.array(
            [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
        )
        wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
        wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

        # make header
        header = wcs.to_header()
        header["BUNIT"] = "Jy"

        # make image
        image = np.zeros((1, params.nchan, params.img_size, params.img_size))
        image[:, :, params.img_size // 2, params.img_size // 2] = stokes[1]  # put a point source at the center

        # write to FITS file
        hdu = fits.PrimaryHDU(image, header=header)
        test_filename = params.random_named_file(suffix=".fits")
        hdu.writeto(test_filename, overwrite=True)

        test_skymodels.append(test_filename)

    # process the FITS files
    intensities = skymodel_from_fits(
        test_skymodels, 0.0, 0.0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    ).image

    # create expected intensities
    expected_intensities = np.zeros(
        (params.img_size, params.img_size, params.chan_freqs.size, params.ncorr), dtype=np.complex128
    )
    expected_intensities[params.img_size // 2, params.img_size // 2, :, 0] = (
        stokes_params[0][1] + stokes_params[1][1]
    )  # I + Q
    expected_intensities[params.img_size // 2, params.img_size // 2, :, 1] = (
        stokes_params[2][1] + 1j * stokes_params[3][1]
    )  # U + iV
    expected_intensities[params.img_size // 2, params.img_size // 2, :, 2] = (
        stokes_params[2][1] - 1j * stokes_params[3][1]
    )  # U - iV
    expected_intensities[params.img_size // 2, params.img_size // 2, :, 3] = (
        stokes_params[0][1] - stokes_params[1][1]
    )  # I - Q
    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
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
#     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES'
#     wcs.wcs.cdelt = np.array([-params.cell_size/3600, params.cell_size/3600,
#                                params.chan_freqs[1]-params.chan_freqs[0], 1.0])
#     wcs.wcs.crpix = [params.img_size/2, params.img_size/2, 1, 1] # reference pixel
#     wcs.wcs.crval = [0.0, -30.0, params.chan_freqs[0], 1] # reference pixel RA and Dec in deg

#     # make header
#     header = wcs.to_header()
#     header['BUNIT'] = 'Jy'

#     # make image
#     image = np.ones((4, params.nchan, params.img_size, params.img_size))

#     # write to FITS file
#     hdu = fits.PrimaryHDU(image, header=header)
#     test_filename = f'test_{uuid.uuid4()}.fits'
#     params.test_files.append(test_filename)
#     hdu.writeto(test_filename, overwrite=True)

#     # process the FITS file
#     with params.assertRaises(SkymodelError):
#         skymodel_from_fits(test_filename, 0.0, np.deg2rad(-30.0), params.chan_freqs, params.ms_delta_nu,
#                            params.ncorr, params.basis)

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
#     wcs.wcs.cdelt = np.array([-params.cell_size/3600, params.cell_size/3600,
#                               params.chan_freqs[1]-params.chan_freqs[0], 1.0])
#     wcs.wcs.crpix = [params.img_size/2, params.img_size/2, 1, 1] # reference pixel
#     wcs.wcs.crval = [0.0, -30.0, params.chan_freqs[0], 1] # reference pixel RA and Dec in deg

#     # make header
#     header = wcs.to_header()
#     header['BUNIT'] = 'Jy'

#     # make image
#     image = np.ones((2, params.nchan, params.img_size, params.img_size))

#     # write to FITS file
#     hdu = fits.PrimaryHDU(image, header=header)
#     test_filename = f'test_{uuid.uuid4()}.fits'
#     params.test_files.append(test_filename)
#     hdu.writeto(test_filename, overwrite=True)

#     # process the FITS file
#     with params.assertRaises(SkymodelError):
#         skymodel_from_fits(test_filename, 0.0, np.deg2rad(-30.0), params.chan_freqs, params.ms_delta_nu,
#                            params.ncorr, params.basis)

# FIXME: Incorporate the two tests below into the ones above as lm-grid is no longer created
# when FFT is used for visibility prediction.
# def test_lm_grid_creation_with_stokes_I_only(self):
#     """
#     Tests if the l-m grid is created correctly for Stokes I only FITS file (no spectral axis in FITS file)
#     Validates:
#         - output l-m grid shape
#         - output l-m grid values
#     """

#     # create a FITS file with Stokes I only
#     wcs = WCS(naxis=2)
#     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
#     wcs.wcs.cdelt = np.array([-params.cell_size/3600, params.cell_size/3600]) # pixel scale in deg
#     wcs.wcs.crpix = [params.img_size/2, params.img_size/2] # reference pixel
#     wcs.wcs.crval = [0.0, -30.0] # reference pixel RA and Dec in deg

#     # make header
#     header = wcs.to_header()
#     header['BUNIT'] = 'Jy'

#     # make image
#     image = np.ones((params.img_size, params.img_size))

#     # write to FITS file
#     hdu = fits.PrimaryHDU(image, header=header)
#     test_filename = f'test_{uuid.uuid4()}.fits'
#     params.test_files.append(test_filename)
#     hdu.writeto(test_filename, overwrite=True)

#     # process the FITS file
#     _, lm, _, _, _, _ = skymodel_from_fits(test_filename, 0.0, np.deg2rad(-30.0), params.chan_freqs,
#                                            params.ms_delta_nu, params.ncorr, params.basis)

#     # created expected l-m grid
#     delt = np.deg2rad(params.cell_size/3600)
#     l = np.sort(np.arange(1-params.img_size/2, 1-params.img_size/2+params.img_size) * delt)
#     m = np.arange(1-params.img_size/2, 1-params.img_size/2+params.img_size) * delt
#     ll, mm = np.meshgrid(l, m)
#     expected_lm = np.stack([ll, mm], axis=-1)

#     # validate the l-m grid
#     assert lm.shape == expected_lm.shape
#     assert np.allclose(lm, expected_lm)

#     # clean up
#     os.remove(test_filename)

# def test_lm_grid_creation_with_stokes_I_and_spectral_axis(self):
#     """
#     Tests if the l-m grid is created correctly for Stokes I only FITS file with spectral axis
#     Validates:
#         - output l-m grid shape
#         - output l-m grid values
#     """

#     # create a FITS file with Stokes I only
#     wcs = WCS(naxis=3)
#     wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
#     wcs.wcs.cdelt = np.array([-params.cell_size/3600, params.cell_size/3600,
#                              params.chan_freqs[1]-params.chan_freqs[0]])
#     wcs.wcs.crpix = [params.img_size/2, params.img_size/2, 1] # reference pixel
#     wcs.wcs.crval = [0.0, -30.0, params.chan_freqs[0]] # reference pixel RA and Dec in deg

#     # make header
#     header = wcs.to_header()
#     header['BUNIT'] = 'Jy'

#     # make image
#     image = np.ones((params.nchan, params.img_size, params.img_size))

#     # write to FITS file
#     hdu = fits.PrimaryHDU(image, header=header)
#     test_filename = f'test_{uuid.uuid4()}.fits'
#     params.test_files.append(test_filename)
#     hdu.writeto(test_filename, overwrite=True)

#     # process the FITS file
#     _, lm, _, _, _, _ = skymodel_from_fits(test_filename, 0.0, np.deg2rad(-30.0), params.chan_freqs,
#                                            params.ms_delta_nu, params.ncorr, params.basis)

#     # created expected l-m grid
#     delt = np.deg2rad(params.cell_size/3600)
#     l = np.sort(np.arange(1-params.img_size/2, 1-params.img_size/2+params.img_size) * delt)
#     m = np.arange(1-params.img_size/2, 1-params.img_size/2+params.img_size) * delt
#     ll, mm = np.meshgrid(l, m)
#     expected_lm = np.stack([ll, mm], axis=-1)

#     # validate the l-m grid
#     assert lm.shape == expected_lm.shape
#     assert np.allclose(lm, expected_lm)

#     # clean up
#     os.remove(test_filename)


def test_bmaj_bmin_header_scaling(params):
    """
    Tests flux scaling for FITS with BMAJ/BMIN in header (single beam for all channels).
    """
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = params.cell_size * 7 / 3600  # degrees
    header["BMIN"] = params.cell_size * 5 / 3600  # degrees
    header["BPA"] = 0.0
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Hz"
    header["CUNIT4"] = ""

    image = np.zeros((1, params.nchan, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0
    hdu = fits.PrimaryHDU(image, header=header)

    test_filename = params.random_named_file(suffix="bmajmin.fits")
    hdu.writeto(test_filename, overwrite=True)

    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image

    # Calculate expected scaling
    freq_scale = params.chan_freqs[0] / params.chan_freqs
    bmaj = header["BMAJ"] * freq_scale
    bmin = header["BMIN"] * freq_scale
    # if image has multiple frequencies, then beam params need to scale with frequency
    pixel_area = (params.cell_size / 3600) ** 2
    beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2))
    pixels_per_beam = beam_area / pixel_area

    expected_intensities = np.ones((1, params.chan_freqs.size, params.ncorr))
    expected_intensities /= pixels_per_beam[np.newaxis, :, np.newaxis]
    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities, atol=1e-6)


def test_bmaj1_bmin1_cube_scaling(params):
    """
    Tests flux scaling for FITS cube with BMAJ1/BMIN1 in header (per-channel beam).
    """
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Hz"
    header["CUNIT4"] = ""

    for i in range(params.nchan):
        header[f"BMAJ{i + 1}"] = 0.01 + 0.001 * i
        header[f"BMIN{i + 1}"] = 0.005 + 0.001 * i
        header[f"BPA{i + 1}"] = 0.0  # Position angle, not used in scaling

    image = np.zeros((1, params.nchan, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0
    hdu = fits.PrimaryHDU(image, header=header)

    test_filename = params.random_named_file(suffix="bmajmin.fits")
    hdu.writeto(test_filename, overwrite=True)

    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image.real
    pixel_area = np.abs(np.deg2rad(header["CDELT1"])) * np.abs(np.deg2rad(header["CDELT2"]))

    expected_intensities = np.zeros((params.img_size, params.img_size, params.chan_freqs.size, params.ncorr))
    for i in range(params.nchan):
        bmaj_rad = np.deg2rad(header[f"BMAJ{i + 1}"])
        bmin_rad = np.deg2rad(header[f"BMIN{i + 1}"])
        beam_area = (np.pi * bmaj_rad * bmin_rad) / (4 * np.log(2))
        scale = 1.0 / (beam_area / pixel_area)
        expected_intensities[params.img_size // 2, params.img_size // 2, i, :] = scale

    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )
    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities, atol=1e-6)


def test_beam_table_scaling(params):
    """
    Tests flux scaling for FITS with beam table (per-channel beam from table).
    """
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.cdelt = np.array(
        [-params.cell_size / 3600, params.cell_size / 3600, params.chan_freqs[1] - params.chan_freqs[0], 1.0]
    )
    wcs.wcs.crpix = [params.img_size / 2, params.img_size / 2, 1, 1.0]
    wcs.wcs.crval = [0, 0, params.chan_freqs[0], 1.0]

    header = wcs.to_header()
    header["BUNIT"] = "Jy/beam"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Hz"
    header["CUNIT4"] = ""

    image = np.zeros((1, params.nchan, params.img_size, params.img_size))
    image[:, :, params.img_size // 2, params.img_size // 2] = 1.0

    # Create beam table
    beam_table = Table()
    beam_table["BMAJ"] = [15 + 0.1 * i for i in range(params.nchan)]
    beam_table["BMIN"] = [16 + 0.1 * i for i in range(params.nchan)]
    beam_table["BPA"] = [0.0] * params.nchan
    beam_table["BMAJ"].unit = "arcsec"
    beam_table["BMIN"].unit = "arcsec"
    beam_table["BPA"].unit = "deg"
    beam_table.write("beam_table.fits", overwrite=True)
    params.test_files.append("beam_table.fits")

    # Write image and beam table to same FITS file (multi-extension)
    hdu = fits.PrimaryHDU(image, header=header)
    hdul = fits.HDUList([hdu, fits.BinTableHDU(beam_table)])

    test_filename = params.random_named_file(suffix=".fits")
    hdul.writeto(test_filename, overwrite=True)

    predict = skymodel_from_fits(
        test_filename, 0, 0, params.chan_freqs, params.ms_delta_nu, params.ncorr, linear_basis=params.linear_basis
    )
    intensities = predict.image

    pixel_area = np.abs(np.deg2rad(header["CDELT1"])) * np.abs(np.deg2rad(header["CDELT2"]))

    expected_intensities = np.zeros((params.img_size, params.img_size, params.chan_freqs.size, params.ncorr))
    for i in range(params.nchan):
        bmaj_rad = beam_table["BMAJ"][i] * np.pi / (180 * 3600)
        bmin_rad = beam_table["BMIN"][i] * np.pi / (180 * 3600)
        beam_area = (np.pi * bmaj_rad * bmin_rad) / (4 * np.log(2))
        scale = pixel_area / beam_area
        expected_intensities[params.img_size // 2, params.img_size // 2, i, :] = scale

    expected_intensities = expected_intensities.reshape(
        params.img_size * params.img_size, params.chan_freqs.size, params.ncorr
    )

    non_zero_mask = np.any(expected_intensities > params.tol, axis=(1, 2))
    expected_intensities = expected_intensities[non_zero_mask]

    assert intensities.shape == expected_intensities.shape
    assert np.allclose(intensities, expected_intensities)
