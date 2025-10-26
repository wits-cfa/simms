import logging
from typing import Dict, List, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from astropy import units
from fitstoolz.reader import FitsData
from numba import njit, prange
from scabha.basetypes import File

from simms import BIN
from simms.skymodel.catalogue_reader import load_sources
from simms.skymodel.converters import radec2lm
from simms.skymodel.source_factory import (
    CatSource,
    StokesData,
    StokesDataFits,
    contspec,
    exoplanet_transient_logistic,
    gauss_1d,
)
from simms.utilities import (
    FITSSkymodelError as SkymodelError,
)
from simms.utilities import ObjDict, is_range_in_range


@njit(parallel=True)
def pix_radec2lm(ra0: float, dec0: float, ra_coords: np.ndarray, dec_coords: np.ndarray):
    """
    Calculates pixel (l, m) coordinates. Returns sth akin to a 2D meshgrid
    """
    n_pix_l = len(ra_coords)
    n_pix_m = len(dec_coords)
    lm = np.zeros((n_pix_l, n_pix_m, 2), dtype=np.float64)
    for i in prange(len(ra_coords)):
        for j in range(len(dec_coords)):
            l_coords, m_coords = radec2lm(ra0, dec0, ra_coords[i], dec_coords[j])
            lm[i, j, 0] = l_coords
            lm[i, j, 1] = m_coords

    return lm


# TODO: consider assuming degrees for RA and Dec if no units are given
def compute_lm_coords(
    phase_centre: np.ndarray,
    n_ra: float,
    n_dec: float,
    ra_coords: Optional[np.ndarray] = None,
    dec_coords: Optional[np.ndarray] = None,
    tol_mask: Optional[np.ndarray] = None,
):
    """
    Calculates pixel (l, m) coordinates
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


def skymodel_from_sources(
    sources: List[CatSource], chan_freqs: np.ndarray, unique_times: np.ndarray = None, full_stokes: bool = True
):
    mod_sources = []
    for src in sources:
        stokes = StokesData([src.stokes_i, src.stokes_q, src.stokes_u, src.stokes_v])
        if src.line_peak:
            specfunc = gauss_1d
            kwargs = {
                "x0": src.line_peak,
                "width": src.line_width,
            }
        else:
            specfunc = contspec
            kwargs = {
                "coeff": [src.cont_coeff_1, src.cont_coeff_2],
                "nu_ref": src.cont_reffreq,
            }

        stokes.set_spectrum(chan_freqs, specfunc, full_pol=full_stokes, **kwargs)
        if src.is_transient:
            lightcurve_func = exoplanet_transient_logistic
            t0 = unique_times.min()
            unique_times_rel = unique_times - t0
            kwargs = {
                "start_time": unique_times_rel.min(),
                "end_time": unique_times_rel.max(),
                "ntimes": unique_times_rel.shape[0],
                "transient_start": src.transient_start,
                "transient_period": src.transient_period,
                "transient_ingress": src.transient_ingress,
                "transient_absorb": src.transient_absorb,
            }
            stokes.set_lightcurve(lightcurve_func, **kwargs)
        setattr(src, "stokes", stokes)
        mod_sources.append(src)

    return mod_sources


def skymodel_from_catalogue(
    catfile: File, map_path, delimiter, chan_freqs: np.ndarray, unique_times, full_stokes: bool = True
):
    """AI is creating summary for skymodel_from_catalogue

    Args:
        catfile (File): [description]
        map_path ([type]): [description]
        delimiter ([type]): [description]
        chan_freqs (np.ndarray): [description]
        unique_times ([type]): [description]
        full_stokes (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    sources = load_sources(catfile, map_path, delimiter)
    return skymodel_from_sources(sources, chan_freqs=chan_freqs, unique_times=unique_times, full_stokes=full_stokes)


def skymodel_from_fits(
    input_fitsimages: Union[File, List[File]],
    ra0: float,
    dec0: float,
    chan_freqs: np.ndarray,
    ms_delta_nu: float,
    ncorr: int,
    basis: str,
    tol: float = 1e-7,
    use_dft: Optional[bool] = None,
    stack_axis="STOKES",
) -> tuple:
    """
    Processes FITS skymodel into DFT input
    Args:
        input_fitsimages: FITS image or sorted list of FITS images if polarisation is present
        ra0 (float): RA of phase-tracking centre in radians
        dec0 (float): Dec of phase-tracking centre in radians
        chan_freqs (np.ndarray): MS frequencies
        ms_delta_nu (float): MS channel width
        ncorr (int): number of correlations
        basis (str): polarisation basis ("linear" or "circular")
        tol (float): tolerance for pixel brightness
        stokes (Union[int,str]): Stokes parameter to use (0 = I, 1 = Q, 2 = U, 3 = V). If 'all',
        all Stokes parameters are used.
        stack_axis (str|Dict): Stack FITS images along this axis if multiple input images given.
        If Dict, then these should be options to 'fitstoolz.reader.FitsData.add_axis()'
    Returns:
        predict_image (np.ndarray): pixel-by-pixel brightness matrix for each channel and correlation
        lm (np.ndarray): (l, m) coordinate grid for DFT
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
    elif is_range_in_range(ms_range, fits_range):
        raise SkymodelError(
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
                interp_data = data.interp(ra=ra_grid, dec=dec_grid, freq=chan_freqs)
                interp_stokes.append(interp_data)

            # combine into new array with shape (n_stokes, n_pix_l, n_pix_m, len(chan_freqs))
            skymodel = da.stack(interp_stokes, axis=0)
    else:
        expand_freq_dim = nchan > 1

    skymodel = StokesDataFits(fds.coords["STOKES"], dim_idx=0, data=skymodel)
    # The stokes parameters in this class will be transposed to the correct basis.

    predict_image = skymodel.get_brightness_matrix(ncorr, linear_pol_basis=basis == "linear")
    predict_nchan = 1 if expand_freq_dim else nchan
    # first transpose stokes axis to the end,
    predict_image = np.transpose(predict_image, (1, 2, 3, 0))
    # then reshape predict_image to im_to_vis expectations
    reshaped_predict_image = predict_image.reshape(n_pix_l * n_pix_m, predict_nchan, ncorr)

    # get only pixels with brightness > tol
    tol_mask = np.any(np.abs(reshaped_predict_image) > tol, axis=(1, 2))
    non_zero_predict_image = reshaped_predict_image[tol_mask]

    # decide whether image is sparse enough for DFT
    sparsity = 1 - (non_zero_predict_image.size / predict_image.size)

    print(f"ra_grid:{ra_grid}, dec_grid:{dec_grid}")
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
                "use_dft": use_dft,
                "ra_pixel_size": None,
                "dec_pixel_size": None,
            }
        )
