from typing import Optional, Union, List
from scabha.basetypes import File
from simms import BIN, get_logger
from simms.utilities import (
    FITSSkymodelError as SkymodelError, 
    )
import numpy as np
from numba import njit, prange
from simms.skymodel.fitstools import FitsData
from simms.skymodel.catalogue_reader import load_sources
from simms.skymodel.source_factory import (
        StokesData,
        StokesDataFits,
        gauss_1d,
        contspec,
)
from scipy.interpolate import RegularGridInterpolator
from simms.skymodel.converters import radec2lm
from astropy import units

log = get_logger(BIN.skysim)

    
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
            l, m = radec2lm(ra0, dec0, ra_coords[i], dec_coords[j])
            lm[i, j, 0] = l
            lm[i, j, 1] = m
    
    return lm


# TODO: consider assuming degrees for RA and Dec if no units are given
def compute_lm_coords(phase_centre: np.ndarray, n_ra: float, n_dec: float, ra_coords: Optional[np.ndarray]=None,
                    dec_coords: Optional[np.ndarray]=None, tol_mask: Optional[np.ndarray]=None):
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

def skymodel_from_catalogue(catfile:File, map_path, delimiter, 
                chan_freqs: np.ndarray, full_stokes:bool=True,
                ):
    
    sources = load_sources(catfile, map_path, delimiter)
    mod_sources = []
    for src in sources:
        stokes = StokesData([src.stokes_i, src.stokes_q, src.stokes_u, src.stokes_v])
        if src.line_peak:
            specfunc = gauss_1d
            kwargs = {
                "x0" : src.line_peak,
                "width": src.line_width,
            }
        else:
            specfunc = contspec
            kwargs = {
                "coeff": src.cont_coeff_1,
                "nu_ref": src.cont_reffreq,
            }
        stokes.set_spectrum(chan_freqs, specfunc, full_pol=full_stokes, **kwargs)
        setattr(src, "stokes", stokes)
        mod_sources.append(src)
    
    return mod_sources
    


def skymodel_from_fits(input_fitsimages: Union[File, List[File]], ra0: float, dec0: float, chan_freqs: np.ndarray,
                        ms_delta_nu: float, ncorr: int, basis: str, tol: float=1e-7, full_stokes:bool=True,
                        use_dft: Optional[bool]=None) -> tuple:
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
        stokes Union[int,str]: Stokes parameter to use (0 = I, 1 = Q, 2 = U, 3 = V). 
                            If 'all', all Stokes parameters are used.
    Returns:
        predict_image (np.ndarray): pixel-by-pixel brightness matrix for each channel and correlation
        lm (np.ndarray): (l, m) coordinate grid for DFT
    """
    phase_centre = np.array([ra0, dec0])
    nchan = chan_freqs.size
    
    if isinstance(input_fitsimages, List):
        fds = FitsData(input_fitsimages[0])
        fds.register_dimensions(set_dims=['spectral', 'celestial'])
        # No need to read the rest of the files in not using full_stokes
        if full_stokes:
            for fits_image in input_fitsimages[1:]:
                fds.extend_stokes(fits_image)
        
    else:
        fds = FitsData(input_fitsimages)
        fds.register_dimensions(set_dims=['spectral', 'celestial'])
    
    has_stokes = "STOKES" in fds.coord_names
    # ra_coords, delta_ra, dec_coords, delta_dec = None, None, None, None
                
    # computes edges of FITS and MS frequency axes
    ms_start_freq = chan_freqs[0] - 0.5*(ms_delta_nu)
    ms_end_freq = chan_freqs[-1] + 0.5*(ms_delta_nu)
    
    if fds.spectral_coord == "VRAD":
        fits_freqs = fds.get_freq_from_vrad()
        
        dspec = fds.coords["VRAD"].pixel_size * getattr(units, 
                                                    fds.coords["RAD"].units)
        fits_d_nu = dspec.to(units.Hz,
                        doppler_rest=fds.spectral_restfreq*units.Hz,
                        doppler_convention="radio").value

    elif fds.spectral_coord == "VOPT":
        fits_freqs = fds.get_freq_from_vopt()
        dspec = fds.coords["VOPT"].pixel_size * getattr(units, 
                                                    fds.coords["VOPT"].units)
        fits_d_nu = dspec.to(units.Hz,
                        doppler_rest=fds.spectral_restfreq*units.Hz,
                        doppler_convention="optical").value
    else:
        fits_freqs = fds.coords["FREQ"].data
        fits_d_nu = fds.coords["FREQ"].pixel_size
    
    
    nchan_fits = len(fits_freqs)
    fits_start_freq = fits_freqs[0] - 0.5 * fits_d_nu
    fits_end_freq = fits_freqs[-1] + 0.5 * fits_d_nu
    
    ra_coords = fds.coords["RA"]
    dec_coords = fds.coords["DEC"]
    pixel_area = abs(ra_coords.pixel_size * dec_coords.pixel_size)
    
    if ms_start_freq < fits_start_freq or ms_end_freq > fits_end_freq:
        raise SkymodelError(f"Some MS frequencies [{ms_start_freq/1e9:.6f} GHz, {ms_end_freq/1e9:.6f} GHz] "
                            f"are out of bounds of FITS image frequencies[{fits_start_freq/1e9:.6f} GHz, {fits_end_freq/1e9:.6f} GHz]. "
                            "Cannot interpolate FITS image onto MS frequency grid.")
    
    # reshape FITS data to (n_pix_l, n_pix_m, nchan)
    
    trgt_shape = ["STOKES", "RA", "DEC", "FREQ"] if has_stokes else ["RA", "DEC", "FREQ"]
    skymodel = fds.get_xds(transpose=trgt_shape).data
    
    # get image shape
    n_pix_l = ra_coords.size 
    n_pix_m = dec_coords.size
    dra_pix = ra_coords.pixel_size
    ddec_pix = dec_coords.pixel_size

    # convert from intensity to Jy
    if fds.data_units == 'jy/beam':
        fds.register_beam_info()
        beam_area = (np.pi * fds.beam_info["bmaj"] * fds.beam_info["bmin"]) / (4 * np.log(2)) #this should also be an array
        beam_area_pixels = beam_area / pixel_area
        if has_stokes:
            skymodel = skymodel / beam_area_pixels[np.newaxis, np.newaxis, np.newaxis, :]
        else:
            skymodel = skymodel / beam_area_pixels[np.newaxis, np.newaxis, :]
        
    elif fds.data_units == '':
        log.warning(f"FITS sky model has no BUNIT specified. Assuming data is in Jy")
        
    else:
        log.warning(f"FITS image sky model has unknown BUNIT='{fds.data_units}'. Assuming data is in Jy")
        
        
    if nchan_fits > 1 and (len(chan_freqs) != len(fits_freqs) or np.any(fits_freqs != chan_freqs)):
        # interpolate FITS cube
        log.warning(f"Interpolating FITS sky model onto MS channel frequency grid. This uses a lot of memory.")
        ## The RA and Dec coordinates need to re-computed to ensure that they remain monotonically decreasing when passing zero
        ra_grid = [ra_coords.data[0] + dra_pix*i for i in range(n_pix_l)]
        dec_grid = [dec_coords.data[0] + ddec_pix*i for i in range(n_pix_m)]
        fits_interp = RegularGridInterpolator((ra_grid, dec_grid, fits_freqs), skymodel)
        
        del ra_grid, dec_grid
        ra, dec, vv = np.meshgrid(ra_coords.data, dec_coords.data, chan_freqs, indexing="ij")
        radecv = np.vstack((ra.ravel(), dec.ravel(), vv.ravel())).T
        skymodel = fits_interp(radecv).reshape(n_pix_l, n_pix_m, nchan)

    #TODO(mika,senkhosi): Is there a reason to have the above interpolation and conversions here, and not at the end of the function?
    
    skymodel = StokesDataFits(fds.coords["STOKES"], skymodel)
    stokes_i = skymodel.I
    stokes_q = skymodel.Q
    stokes_u = skymodel.U
    stokes_v = skymodel.V
        
    # compute per-pixel brghtness matrix
    if not has_stokes or full_stokes is False: # if no has_stokes is present
        predict_image = np.zeros((n_pix_l, n_pix_m, nchan, ncorr)) # create pixel grid for sky model
        
        if ncorr == 2: # if ncorr is 2, we only need compute XX and duplicate to YY
            predict_image[:, :, :, 0] = stokes_i
            predict_image[:, :, :, 1] = stokes_i
        elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
            predict_image[:, :, :, 0] = stokes_i
            predict_image[:, :, :, 3] = stokes_i
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    else: # if has_stokes is present
        predict_image = np.zeros((n_pix_l, n_pix_m, nchan, ncorr), dtype=np.complex128) # create pixel grid for sky model
        if basis == "linear":
            if ncorr == 2: # if ncorr is 2, we only need compute XX and YY correlations
                log.warning("Only two correlations requested, but four are present in the FITS image directory. Using only Stokes I and Q.")
                predict_image[:, :, :, 0] = stokes_i + stokes_q
                predict_image[:, :, :, 1] = stokes_i - stokes_q
            elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
                predict_image[:, :, :, 0] = stokes_i + stokes_q
                predict_image[:, :, :, 1] = stokes_u + 1j * stokes_v
                predict_image[:, :, :, 2] = stokes_u - 1j * stokes_v
                predict_image[:, :, :, 3] = stokes_i - stokes_q
            else:
                raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
        elif basis == "circular":
            if ncorr == 2: # if ncorr is 2, we only need compute XX and YY correlations
                log.warning("Only two correlations requested, but four are present in the FITS image directory. Using only Stokes I and V.")
                predict_image[:, :, :, 0] = stokes_i + stokes_v
                predict_image[:, :, :, 1] = stokes_i - stokes_v
            elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
                predict_image[:, :, :, 0] = stokes_i + stokes_v
                predict_image[:, :, :, 1] = stokes_q + 1j * stokes_u
                predict_image[:, :, :, 2] = stokes_q - 1j * stokes_u
                predict_image[:, :, :, 3] = stokes_i - stokes_v
            else:
                raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
        else:
            raise ValueError(f"Unrecognised has_stokes basis '{basis}'. Use 'linear' or 'circular'.")


    #TODO(mika,senkhosi) there are two many reshaping operations. If you know the actual target (required by im_to_vis), then reshape in the original data to that shape from the start.
    
    # reshape predict_image to im_to_vis expectations
    reshaped_predict_image = predict_image.reshape(n_pix_l * n_pix_m, nchan, ncorr)
    
    # get only pixels with brightness > tol
    tol_mask = np.any(np.abs(reshaped_predict_image) > tol, axis=(1, 2))
    non_zero_predict_image = reshaped_predict_image[tol_mask]
    
    # decide whether image is sparse enough for DFT
    sparsity = 1 - (non_zero_predict_image.size/predict_image.size)
    
    
    if use_dft is None:
        if sparsity >= 0.8:
            log.info(f"More than 80% of pixels have intensity < {(tol*1e6):.2f} μJy. DFT will be used for visibility prediction.")
            use_dft = True
            non_zero_lm = compute_lm_coords(
                phase_centre,
                n_pix_l,
                n_pix_m, 
                ra_coords.data, 
                dec_coords.data, 
                tol_mask
            )
            
            return non_zero_predict_image, non_zero_lm, has_stokes, use_dft, None, None
        else:
            log.info(f"More than 20% of pixels have intensity > {(tol*1e6):.2f} μJy. FFT will be used for visibility prediction.")
            use_dft = False
            
            return predict_image, None, has_stokes, use_dft, ra_coords.pixel_size, dec_coords.pixel_size,
    else:
        log.info(f"Filtered out {sparsity*100:.2f}% of pixels using {(tol*1e6):.2f}-μJy tolerance.")
        non_zero_lm = compute_lm_coords(
            phase_centre,
            n_pix_l,
            n_pix_m, 
            ra_coords, 
            dec_coords, 
            tol_mask
        )
        
        return non_zero_predict_image, non_zero_lm, has_stokes, use_dft, None, None
    