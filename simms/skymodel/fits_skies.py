import logging
from typing import Dict, List, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from astropy import units
from fitstoolz.reader import FitsData
from scabha.basetypes import File

from simms import BIN
from simms.exceptions import FITSSkymodelError
from simms.skymodel.source_factory import StokesDataFits
from simms.utilities import ObjDict, is_range_in_range, pix_radec2lm


def compute_lm_coords(
    phase_centre: np.ndarray,
    n_ra: float,
    n_dec: float,
    ra_coords: np.ndarray = None,
    dec_coords: np.ndarray = None,
    tol_mask: np.ndarray = None,
):
    """
    Compute direction-cosine coordinates (l, m) for an image grid.

    Parameters
    ----------
    phase_centre : numpy.ndarray
        Array-like of shape (2,) with [ra0, dec0] of the phase-tracking centre
        in radians.
    n_ra : int
        Number of pixels along right ascension (l-axis).
    n_dec : int
        Number of pixels along declination (m-axis).
    ra_coords : numpy.ndarray, optional
        1D array of right ascension pixel coordinates in radians.
    dec_coords : numpy.ndarray, optional
        1D array of declination pixel coordinates in radians.
    tol_mask : numpy.ndarray, optional
        Boolean mask of shape (n_ra * n_dec,) selecting pixels to keep.
        When provided, only the corresponding (l, m) rows are returned.

    Returns
    -------
    numpy.ndarray
        Array of direction cosines:
        - If `tol_mask` is None, an array with the grid of (l, m) values with
          shape (..., 2), where the last dimension stores (l, m).
        - If `tol_mask` is provided, a 2D array with shape (N, 2) containing
          only the selected (l, m) pairs.

    Notes
    -----
    The conversion from (ra, dec) to (l, m) is performed by
    `simms.utilities.pix_radec2lm` relative to the given phase centre.
    All angular quantities are in radians.
    """
    # calculate pixel (l, m) coordinates
    ra0, dec0 = phase_centre
    lm = pix_radec2lm(ra0, dec0, ra_coords, dec_coords)

    if isinstance(tol_mask, np.ndarray):
        # reshape lm for DFT
        reshaped_lm = lm.reshape(n_ra * n_dec, 2)
        non_zero_lm = reshaped_lm[tol_mask]
        return non_zero_lm

    return lm


def skymodel_from_fits(
    input_fitsimages: Union[File, List[File]],
    ra0: float,
    dec0: float,
    chan_freqs: np.ndarray,
    ms_delta_nu: float,
    ncorr: int,
    linear_basis: bool = True,
    tol: float = 1e-7,
    use_dft: Optional[bool] = None,
    stack_axis="STOKES",
    interpolation="nearest",
) -> tuple:
    """
    Convert one or more FITS images into a brightness array usable for visibility
    prediction, optionally computing sparse (l, m) coordinates for DFT. Handles
    unit conversion, frequency interpolation, Stokes stacking, and sparsity-based
    selection between DFT and FFT workflows.

    Parameters
    ----------
    input_fitsimages : scabha.basetypes.File or list of File
        A single FITS image or an ordered list of FITS images (e.g., per Stokes).
    ra0 : float
        Right ascension of the phase-tracking centre in radians.
    dec0 : float
        Declination of the phase-tracking centre in radians.
    chan_freqs : numpy.ndarray
        1D array of MS channel centre frequencies in Hz.
    ms_delta_nu : float
        MS channel width in Hz.
    ncorr : int
        Number of output correlations (e.g., 1, 2, or 4).
    linear_basis : bool, default True
        If True, output correlations are in the linear basis (XX, XY, YX, YY).
        If False, use the circular basis (RR, RL, LR, LL).
    tol : float, default 1e-7
        Pixel brightness threshold used to select non-zero pixels for DFT.
        Units follow the FITS BUNIT after conversion (typically Jy).
    use_dft : bool or None, optional
        Whether to force DFT (True) or FFT (False). If None, the choice is made
        based on the sparsity of the thresholded image (DFT if >= 80% zeros).
    stack_axis : str or dict, default "STOKES"
        Axis name to stack along when multiple input FITS images are provided.
        If a dict, it is passed to `fitstoolz.reader.FitsData.add_axis()`.
    interpolation : {"nearest", "linear", ...}, default "nearest"
        Interpolation method used when regridding the FITS spectral axis to the
        MS frequencies.

    Returns
    -------
    ObjDict
        Container with the following fields:
        - image : numpy.ndarray
            If DFT is selected or forced: shape (N_nonzero, N_freq, ncorr),
            containing only thresholded non-zero pixels.
            If FFT is selected: shape (n_pix_l, n_pix_m, N_freq, ncorr),
            the full image cube.
        - lm : numpy.ndarray or None
            If DFT is selected or forced: array of shape (N_nonzero, 2) with
            direction cosines (l, m) for the retained pixels. Otherwise None.
        - is_polarised : bool
            True if any of Q, U, V are present in the input.
        - expand_freq_dim : bool
            True if the FITS image had a single frequency and should be expanded
            along the MS frequency axis downstream.
        - use_dft : bool
            The final choice used for visibility prediction.
        - ra_pixel_size : float or None
            Pixel size along RA in radians for FFT workflows; None for DFT.
        - dec_pixel_size : float or None
            Pixel size along Dec in radians for FFT workflows; None for DFT.

    Raises
    -------
    TypeError
        If `stack_axis` is neither a string nor a dict.
    RuntimeError
        If the requested `stack_axis` does not exist and cannot be added.
    FITSSkymodelError
        If the MS frequency range lies outside the FITS frequency coverage and
        interpolation is not possible.

    Notes
    -----
    - FITS cubes with spectral coordinates in velocity (VRAD/VOPT) are converted
      to frequency using `astropy.units` and the Doppler convention.
    - If BUNIT is "Jy/beam", beam areas are used to convert to Jy/pixel.
    - When spectral grids differ, the FITS image is interpolated onto the MS
      frequency grid, which can be memory intensive for large cubes.
    """
    log = logging.getLogger(BIN.skysim)

    phase_centre = np.array([ra0, dec0])
    nchan = chan_freqs.size

    dummy_stokes = {
        "name": "STOKES",
        "idx": 0,
        "axis_grid": da.asarray([0]),
        "coord_type": "stokes",
        "attrs": dict(ref_pixel=0, units="Jy", size=1, dim="stokes", pixel_size=1),
    }

    if isinstance(input_fitsimages, List):
        fds = FitsData(input_fitsimages[0])
        if isinstance(stack_axis, Dict):
            stack_options = dict(stack_axis)
            stack_axis = stack_options["name"]
            fds.add_axis(**stack_axis)
        elif not isinstance(stack_axis, str):
            raise TypeError(f"Option 'stack_axis' must either be a string or a dictionary. Found {type(stack_axis)}")
        elif stack_axis not in fds.coord_names:
            if stack_axis == "STOKES":
                fds.add_axis(**dummy_stokes)
            else:
                raise RuntimeError(
                    f"Input skymodel FITS images cannot combined along the given axis '{stack_axis}'"
                    f"because it doesn't exist in the input images"
                )

        fds.expand_along_axis_from_files(stack_axis, input_fitsimages[1:])
    else:
        fds = FitsData(input_fitsimages)

    if "STOKES" not in fds.coord_names:
        fds.add_axis(**dummy_stokes)

    # computes edges of FITS and MS frequency axes
    ms_start_freq = chan_freqs[0] - 0.5 * (ms_delta_nu)
    ms_end_freq = chan_freqs[-1] + 0.5 * (ms_delta_nu)
    # If FITS file doesn't have a spectral axis, assume 1 frequency at MS-centre equal to MS bandwidth
    if not hasattr(fds, "spectral_coord"):
        crval = ms_start_freq + (ms_end_freq - ms_start_freq) / 2
        fds.add_axis(
            "FREQ",
            0,
            coord_type="spectral",
            axis_grid=da.asarray([crval]),
            attrs={
                "pixel_size": ms_end_freq - ms_start_freq,
                "size": 1,
                "units": "Hz",
                "ref_pixel": 0,
            },
        )

    if fds.spectral_coord == "VRAD":
        fits_freqs = fds.get_freq_from_vrad()

        dspec = fds.coords["VRAD"].pixel_size * getattr(units, fds.coords["RAD"].units)
        fits_d_nu = dspec.to(units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="radio").value

    elif fds.spectral_coord == "VOPT":
        fits_freqs = fds.get_freq_from_vopt()
        dspec = fds.coords["VOPT"].pixel_size * getattr(units, fds.coords["VOPT"].units)
        fits_d_nu = dspec.to(
            units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="optical"
        ).value
    else:
        fits_freqs = fds.coords["FREQ"].data
        fits_d_nu = fds.coords["FREQ"].pixel_size

    nchan_fits = len(fits_freqs)
    fits_start_freq = fits_freqs[0] - 0.5 * fits_d_nu
    fits_end_freq = fits_freqs[-1] + 0.5 * fits_d_nu

    ra_coords = fds.coords["RA"]
    dec_coords = fds.coords["DEC"]

    ra_grid = np.squeeze(ra_coords.data * getattr(units, ra_coords.units).to("rad"))
    dec_grid = np.squeeze(dec_coords.data * getattr(units, dec_coords.units).to("rad"))
    ra_pixel_size = ra_coords.pixel_size * getattr(units, ra_coords.units).to("rad")
    dec_pixel_size = dec_coords.pixel_size * getattr(units, dec_coords.units).to("rad")

    print(f"dec_pix:{np.rad2deg(dec_pixel_size)}, ra_pix:{np.rad2deg(ra_pixel_size)}")
    pixel_area = abs(ra_pixel_size * dec_pixel_size)

    ms_range = (ms_start_freq, ms_end_freq)
    fits_range = (fits_start_freq, fits_end_freq)
    # check if MS freqs are valid
    if nchan_fits == 1 and is_range_in_range(fits_range, ms_range):
        pass
    elif not is_range_in_range(ms_range, fits_range):
        raise FITSSkymodelError(
            f"MS frequencies [{ms_start_freq / 1e9:.6f} GHz, {ms_end_freq / 1e9:.6f} GHz] "
            f"are outside the FITS image frequencies[{fits_start_freq / 1e9:.6f} GHz, {fits_end_freq / 1e9:.6f} GHz]. "
            "Cannot interpolate FITS image onto MS frequency grid."
        )

    trgt_shape = ["STOKES", "RA", "DEC", "FREQ"]
    skymodel = fds.get_xds(transpose=trgt_shape).data

    fds.close()

    # get image shape
    n_pix_l = ra_coords.size
    n_pix_m = dec_coords.size

    # convert from intensity to Jy
    if fds.data_units == "Jy/beam":
        if fds.beam_table:
            bmaj = fds.beam_table["BMAJ"].to("rad").value
            bmin = fds.beam_table["BMIN"].to("rad").value
            beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2))  # this should also be an array
            pixels_per_beam = beam_area / pixel_area
            skymodel = skymodel / pixels_per_beam[np.newaxis, np.newaxis, np.newaxis, :]

            # check only the the value of the
        else:
            log.warning(
                f"FITS sky model units (BUNIT) are '{fds.data_units}', but no beam information found."
                "Assuming data are in Jy"
            )

    elif fds.data_units == "":
        log.warning("FITS sky model has no BUNIT specified. Assuming data are in Jy")

    elif fds.data_units not in ["Jy", "Jy/pixel"]:
        log.warning(f"FITS image sky model has unknown BUNIT='{fds.data_units}'. Assuming data are in Jy")

    if nchan_fits > 1:
        expand_freq_dim = False
        if nchan != nchan_fits or np.any(chan_freqs != fits_freqs):
            log.warning("Interpolating FITS sky model onto MS frequency grid.  This can use a lot of memory.")
            interp_stokes = []
            for stokes in range(fds.coords["STOKES"].size):
                data = xr.DataArray(
                    skymodel[stokes, ...],
                    coords={"ra": ra_grid, "dec": dec_grid, "freq": fits_freqs},
                    dims=["ra", "dec", "freq"],
                )
                interp_data = data.interp(freq=chan_freqs, method=interpolation)
                interp_data = interp_data.interpolate_na(dim="freq", method=interpolation, fill_value="extrapolate")
                interp_stokes.append(interp_data)

            # combine into new array with shape (n_stokes, n_pix_l, n_pix_m, len(chan_freqs))
            skymodel = da.stack(interp_stokes, axis=0)
    else:
        expand_freq_dim = nchan > 1

    skymodel = StokesDataFits(fds.coords["STOKES"], dim_idx=0, data=skymodel, linear_basis=linear_basis)
    # The stokes parameters in this class will be transposed to the correct basis.

    predict_image = skymodel.get_brightness_matrix(ncorr)
    predict_nchan = 1 if expand_freq_dim else nchan
    ref_freq = fits_freqs[:1] if expand_freq_dim else None

    # first transpose stokes axis to the end,
    predict_image = np.transpose(predict_image, (1, 2, 3, 0))

    # then reshape predict_image to im_to_vis expectations
    reshaped_predict_image = predict_image.reshape(n_pix_l * n_pix_m, predict_nchan, ncorr)

    # get only pixels with brightness > tol
    tol_mask = np.any(np.abs(reshaped_predict_image) > tol, axis=(1, 2))
    non_zero_predict_image = reshaped_predict_image[tol_mask]

    # decide whether image is sparse enough for DFT
    sparsity = 1 - (non_zero_predict_image.size / predict_image.size)

    if use_dft is None:
        if sparsity >= 0.8:
            log.info(
                f"More than 80% of pixels have intensity < {(tol * 1e6):.2f} μJy. "
                "DFT will be used for visibility prediction."
            )
            use_dft = True
            non_zero_lm = compute_lm_coords(
                phase_centre,
                n_pix_l,
                n_pix_m,
                ra_grid,
                dec_grid,
                tol_mask,
            )
            return ObjDict(
                {
                    "image": non_zero_predict_image,
                    "lm": non_zero_lm,
                    "is_polarised": skymodel.is_polarised,
                    "expand_freq_dim": expand_freq_dim,
                    "ref_freq": ref_freq,
                    "use_dft": use_dft,
                    "ra_pixel_size": None,
                    "dec_pixel_size": None,
                }
            )
        else:
            log.info(
                f"More than 20% of pixels have intensity > {(tol * 1e6):.2f} μJy. "
                "FFT will be used for visibility prediction."
            )
            use_dft = False

            return ObjDict(
                {
                    "image": predict_image,
                    "lm": None,
                    "is_polarised": skymodel.is_polarised,
                    "expand_freq_dim": expand_freq_dim,
                    "ref_freq": ref_freq,
                    "use_dft": use_dft,
                    "ra_pixel_size": ra_pixel_size,
                    "dec_pixel_size": dec_pixel_size,
                }
            )
    else:
        log.info(f"Filtered out {sparsity * 100:.2f}% of pixels using {(tol * 1e6):.2f}-μJy tolerance.")
        non_zero_lm = compute_lm_coords(phase_centre, n_pix_l, n_pix_m, ra_grid, dec_grid, tol_mask)

        return ObjDict(
            {
                "image": non_zero_predict_image,
                "lm": non_zero_lm,
                "is_polarised": skymodel.is_polarised,
                "expand_freq_dim": expand_freq_dim,
                "ref_freq": ref_freq,
                "use_dft": use_dft,
                "ra_pixel_size": None,
                "dec_pixel_size": None,
            }
        )
