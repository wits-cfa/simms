from typing import Optional, Union, List
from scabha.basetypes import File, MS
from simms import BIN, get_logger
from simms.utilities import FITSSkymodelError as SkymodelError
from simms.skymodel.source_factory import singlegauss_1d, contspec
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from daskms import xds_from_ms, xds_from_table
from africanus.dft import im_to_vis
from simms.constants import gauss_scale_fact, C, FWHM_scale_fact
from simms.skymodel.converters import (
    convert2float, 
    convert2Hz, 
    convertdec2rad, 
    convertra2rad,
    convert2Jy,
    convert2rad,
    radec2lm
)

log = get_logger(BIN.skysim)


class Source:
    
    def __init__(self, name, ra, dec, emaj, emin, pa):
        self.name = name
        self.ra = convertra2rad(ra)
        self.dec = convertdec2rad(dec)
        self.spectrum = None
        self.shape = None
        self.emaj = convert2rad(emaj)
        self.emin = convert2rad(emin)
        self.pa = convert2rad(pa)

    def set_lm(self, ra0, dec0):
        self.l, self.m = radec2lm(np.array([ra0, dec0]), self.ra, self.dec)
    
    @property
    def is_point(self):
        
        return self.emaj in  ('null',None) and self.emin in (None, 'null') 

            
class Spectrum:
    
        def __init__(self, stokes_i, stokes_q, stokes_u, stokes_v, line_peak, line_width, line_restfreq, cont_reffreq, cont_coeff_1, cont_coeff_2):
            self.stokes_i = convert2Jy(stokes_i)
            self.stokes_q = convert2Jy(stokes_q)
            self.stokes_u = convert2Jy(stokes_u)
            self.stokes_v = convert2Jy(stokes_v)
            self.line_peak = convert2Hz(line_peak)
            self.line_width = convert2Hz(line_width)
            self.line_restfreq = convert2Hz(line_restfreq)
            self.cont_reffreq = convert2Hz(cont_reffreq)
            self.cont_coeff_1 = convert2float(cont_coeff_1, null_value=0)
            self.cont_coeff_2 = convert2float(cont_coeff_2, null_value=0)
     
        def make_spectrum(self, freqs):
            # only Stokes I case
            if all(param in [None, "null"] for param in [self.stokes_q, self.stokes_u, self.stokes_v]):
                if self.line_peak not in [None, "null"]:
                    return singlegauss_1d(freqs, self.stokes_i, self.line_width, self.line_peak)
                elif self.cont_reffreq not in [None, "null"]:
                    return contspec(freqs, self.stokes_i, self.cont_coeff_1, self.cont_reffreq)
                else:
                    return self.stokes_i
            
            # case with Stokes I plus at least one of the other Stokes parameters
            spectrum = []
            for stokes_param in [self.stokes_i, self.stokes_q, self.stokes_u, self.stokes_v]:
                if stokes_param in [None, "null"]:
                    spectrum.append(np.zeros_like(freqs))
                elif self.line_peak not in [None, "null"]:
                    spectrum.append(singlegauss_1d(freqs, stokes_param, self.line_width, self.line_peak))
                elif self.cont_reffreq not in [None, "null"]:
                    spectrum.append(contspec(freqs, stokes_param, self.cont_coeff_1, self.cont_reffreq))
                else:
                    spectrum.append(stokes_param * np.ones_like(freqs))
            
            return np.stack(spectrum, axis=0)

def makesources(data, freqs, ra0, dec0):
    num_sources = len(data['name'][1])
    
    sources = []
    for i in range(num_sources):
        source = Source(
            name = data['name'][1][i],
            ra = data['ra'][1][i],
            dec = data['dec'][1][i],
            emaj = data['emaj'][1][i] if i < len(data['emaj'][1]) else None,
            emin = data['emin'][1][i] if i < len(data['emin'][1]) else None,
            pa = data['pa'][1][i] if i < len(data['pa'][1]) else '0deg'
            
        )
        
        spectrum = Spectrum(
            stokes_i = data['stokes_i'][1][i],
            stokes_q = data['stokes_q'][1][i] if i < len(data['stokes_q'][1]) else None,
            stokes_u = data['stokes_u'][1][i] if i < len(data['stokes_u'][1]) else None,
            stokes_v = data['stokes_v'][1][i] if i < len(data['stokes_v'][1]) else None,
            cont_reffreq = data['cont_reffreq'][1][i] if i < len(data['cont_reffreq'][1]) else None,
            line_peak = data['line_peak'][1][i] if i < len(data['line_peak'][1]) else None,
            line_width = data['line_width'][1][i] if i < len(data['line_width'][1]) else None,
            line_restfreq = data['line_restfreq'][1][i] if i < len(data['line_restfreq'][1]) else None,
            cont_coeff_1 = (data['cont_coeff_1'][1][i]) if i < len(data['cont_coeff_1'][1]) else None,
            cont_coeff_2 = data['cont_coeff_2'][1][i] if i < len(data['cont_coeff_2'][1]) else None
            
        )
        
        source.set_lm(ra0, dec0)
        source.spectrum = spectrum.make_spectrum(freqs)
        sources.append(source)
        
    return sources


def check_var_axis(header, var: str):
    """
    Finds the axis number of the variable in the FITS header
    Args:
        - header: FITS header
        - var: variable to find
    Returns:
        - axis_num: axis number of the variable
    """
    axis_num = 1
    while f'CTYPE{axis_num}' in header:
        ctype = header[f'CTYPE{axis_num}']
        if ctype.lower().startswith(var.lower()):
            return str(axis_num)
        axis_num += 1
        
    raise SkymodelError(f"Could not find axis with CTYPE {var.upper()} or starting with CTYPE {var.upper()}")


def check_data_axis_ordering(header, required_axes: List[str]):
    """
    Finds the indices of the required axes in the data array
    Args:
        - header: FITS header
        - required_axes: list of required axes
    Returns:
        - data_axis_indices: array of axis indices in the data array in order of required_axes
    """
    naxis = header['NAXIS']
    data_axis_indices = {}
    
    for axis in required_axes:
        axis_num = check_var_axis(header, axis)
        data_axis_indices[axis] = naxis - int(axis_num)
    
    return data_axis_indices


def check_header_axis_ordering(header, required_axes: List[str]):
    """
    Finds the indices of the required axes in the header
    Args:
        - header: FITS header
        - required_axes: list of required axes
    """
    header_axis_indices = np.array([], dtype=int)
    
    for axis in required_axes:
        axis_num = check_var_axis(header, axis)
        header_axis_indices = np.append(header_axis_indices, int(axis_num))
        
    return header_axis_indices


def compute_lm_coords(wcs: WCS, phase_centre: np.ndarray, img_dims: np.ndarray, spectral_axis: Optional[bool]=False):
    """
    Calculates pixel (l, m) coordinates
    Args:
        wcs (WCS): WCS object (axes ordering: longitude, latitude, spectral)
        phase_centre (np.ndarray): phase centre coordinates
        ra (np.ndarray): RA coordinates of pixels
        dec (np.ndarray): Dec coordinates of pixels
        img_dims (np.ndarray): image l and m dimensions
        spectral_axis (bool): True if spectral axis is present, False otherwise
    Returns:
        l (np.ndarray): l coordinates
        m (np.ndarray): m coordinates
    """
    ra0, dec0 = phase_centre
    n_pix_l, n_pix_m = img_dims
    x_index, y_index = wcs.wcs.lng, wcs.wcs.lat
    
    if wcs.wcs.radesys:
        frame = wcs.wcs.radesys.lower() # get frame from header
    else:
        log.warning("No RA/Dec system found in header. Assuming ICRS.")
        frame = 'icrs'
        
    pc = SkyCoord(ra0, dec0, frame=frame, unit='rad') # create SkyCoord object for phase centre
    
    # get image dimensions
    if spectral_axis:
        x_pix_0, y_pix_0, _ = wcs.world_to_pixel_values(pc.ra, pc.dec, 0) # get pixel coordinates of phase centre
    else:
        x_pix_0, y_pix_0 = wcs.world_to_pixel_values(pc.ra, pc.dec) # get pixel coordinates of phase centre

    # get pixel scale
    delta_l, delta_m = wcs.wcs.cdelt[x_index], wcs.wcs.cdelt[wcs.wcs.lng]
    
    if wcs.wcs.cunit[x_index] == wcs.wcs.cunit[y_index]:
        if wcs.wcs.cunit[x_index] in ["RAD", "rad"] and wcs.wcs.cunit[y_index] in ["RAD", "rad"]:
            pass
        elif wcs.wcs.cunit[x_index] in ["DEG", "deg"] and wcs.wcs.cunit[y_index] in ["DEG", "deg"]:
            delta_l = np.deg2rad(delta_l)
            delta_m = np.deg2rad(delta_m)
        else:
            raise SkymodelError("RA and Dec units must be in radians or degrees")
    else:
        raise SkymodelError("RA and Dec units must be the same")
    
    # calculate l, m coordinates
    l_coords = np.sort(np.arange(1 - x_pix_0, 1 - x_pix_0 + n_pix_l) * delta_l)
    m_coords = np.arange(1 - y_pix_0, 1 - y_pix_0 + n_pix_m) * delta_m    
    
    return l_coords, m_coords
    

def process_fits_skymodel(input_fitsimages: Union[File, List[File]], ra0: float, dec0: float, chan_freqs: np.ndarray, ncorr: int, tol: float=1e-6, stokes: int = 0):
    """
    Processes FITS skymodel into DFT input
    Args:
        input_fitsimages: FITS image or sorted list of FITS images if polarisation is present
        ra0:                     RA of phase-tracking centre in radians
        dec0:                   Dec of phase-tracking centre in radians
        chan_freqs:         MS frequencies
        ncorr:             number of correlations
        tol:                tolerance for pixel brightness
        stokes:             Stokes parameter to use (0 = I, 1 = Q, 2 = U, 3 = V)
    Returns:
        intensities:    pixel-by-pixel brightness matrix for each channel and correlation
        lm:                 (l, m) coordinate grid for DFT
    """
    
    # if single fits image, turn into list so all processing is the same
    if not isinstance(input_fitsimages, list):
        input_fitsimages = [input_fitsimages]
    
    nchan = chan_freqs.size
    
    model_cubes = []
    for fits_image in input_fitsimages:
        # get header and data
        with fits.open(fits_image) as hdulist:
            header = hdulist[0].header
            naxis = header['NAXIS']
            if naxis < 2:
                raise SkymodelError("FITS image must have at least 2 dimensions")
            
            # get all axis types
            orig_dims = [header[f"CTYPE{naxis - n}"].strip() for n in range(naxis)]
            
            data_slice = [slice(None)] * naxis
            dims = []
            
            for i, dim in enumerate(orig_dims):
                dim_name = dim.split("-")[0].lower()
                # axes of interest
                if dim_name in ["ra", "dec", "freq"]:
                    dims.append(dim_name)
                # axes to be ignored
                else:
                    # Stokes axis
                    if dim_name == "STOKES":
                        data_slice[i] = stokes
                    # other axes
                    else:
                        data_slice[i] = 0
                        if header[f"NAXIS{naxis - i}"] > 1:
                            if dim_name == "time":
                                log.warning(f"Removing 'TIME' axis with size > 1. Using only first time stamp. Use separate files for time-varying models.")
                            else:
                                log.warning(f"Removing '{dim_name.upper()}' axis with size > 1. Using only first element.")
        
        # if not wcs.has_celestial:
        #     raise SkymodelError("FITS image does not have or has unrecognised celestial coordinates")
        # # TODO (Mika, Senkhosi): assume image in l-m coords already if no celestial coords
        # # Would this mean the lengths of CTYPE and CRVAL are not the same? Is that even possible?
    
        # if spectral axis exists and is not singleton
        if "freq" in dims and skymodel.shape[dims.index("freq")] > 1:
            # find frequency axis
            freq_axis = check_var_axis(header, "FREQ")
            
            nband = header[f"NAXIS{freq_axis}"]
            # print(nband)
            refpix_nu = header[f"CRPIX{freq_axis}"]
            if header[f"CUNIT{freq_axis}"] == "Hz":
                delta_nu = header[f"CDELT{freq_axis}"]  # assumes units are Hz
            else:
                raise SkymodelError("Frequency units must be in Hz")
            ref_freq = header[f"CRVAL{freq_axis}"]

            freqs = ref_freq + np.arange(1 - refpix_nu, 1 - refpix_nu + nband) * delta_nu # calculate frequencies
            
            if freqs[0] < chan_freqs[0] or freqs[-1] > chan_freqs[-1]:
                raise SkymodelError("At least one frequency in FITS image is out of bounds of MS channel frequencies.")
            
            # reshape FITS data to (n_pix_l, n_pix_m, nchan)
            skymodel = np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec"), dims.index("freq")))
            
            # get image shape
            n_pix_l, n_pix_m, _ = skymodel.shape
            
            # this knows the coordinate system of the image (e.g. FK5, Galactic or ICRS)
            wcs = WCS(header, naxis=['longitude', 'latitude', 'spectral'])
            
            # calculate pixel (l, m) coordinates
            l_coords, m_coords = compute_lm_coords(wcs, np.array([ra0, dec0]), np.array([n_pix_l, n_pix_m]), spectral_axis=True)
            
            # reproject FITS cube to MS frequencies
            if len(chan_freqs) != len(freqs) or np.any(freqs != chan_freqs):
                log.warning("Reprojecting FITS cube to MS freqs. This uses a lot of memory.")
                
                # interpolate FITS cube
                fits_interp = RegularGridInterpolator((l_coords, m_coords, freqs), skymodel, bounds_error=True)
                
                # reevaluate at MS frequencies
                ll, mm, vv = np.meshgrid(l_coords, m_coords, chan_freqs, indexing="ij")
                lmv = np.vstack((ll.ravel(), mm.ravel(), vv.ravel())).T
                model_cube = fits_interp(lmv).reshape(n_pix_l, n_pix_m, nchan)
            else:
                model_cube = skymodel
            
        else: # singleton or no spectral axis
            # this knows the coordinate system of the image (e.g. FK5, Galactic or ICRS)
            wcs = WCS(header, naxis=['longitude', 'latitude'])
            if len(dims) == 2:
                # reshape FITS data to (n_pix_l, n_pix_m)
                skymodel = np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec")))
                n_pix_l, n_pix_m = skymodel.shape
                
                # calculate pixel (l, m) coordinates
                l_coords, m_coords = compute_lm_coords(wcs, np.array([ra0, dec0]), np.array([n_pix_l, n_pix_m]))
            else:
                # find frequency axis
                freq_axis = check_var_axis(header, "FREQ")
                freq = header[f"CRVAL{freq_axis}"]
                
                # raise error if frequency is not in bounds of MS channel frequencies
                if freq < chan_freqs[0] or freq > chan_freqs[-1]:
                    raise SkymodelError("FITS image frequency is out of bounds of MS channel frequencies.")
                
                # reshape FITS data to (n_pix_l, n_pix_m, nchan)
                skymodel= np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec"), dims.index("freq")))
                skymodel = np.squeeze(skymodel)
                
                # get image shape
                n_pix_l, n_pix_m = skymodel.shape
                
                # calculate pixel (l, m) coordinates
                l_coords, m_coords = compute_lm_coords(wcs, np.array([ra0, dec0]), np.array([n_pix_l, n_pix_m]))
            
            freqs = chan_freqs
            model_cube = np.repeat(skymodel[:, :, np.newaxis], nchan, axis=2) # repeat the image along the frequency axis
        
        model_cubes.append(model_cube)
    
    # compute sky model
    polarisation = False if len(model_cubes) == 1 else True
    
    if not polarisation: # if no polarisation is present
        intensities = np.zeros((n_pix_l, n_pix_m, nchan, ncorr)) # create pixel grid for sky model
        I = model_cubes[0]
        
        if ncorr == 2: # if ncorr is 2, we only need compute XX and duplicate to YY
            intensities[:, :, :, 0] = I
            intensities[:, :, :, 1] = I
        elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
            intensities[:, :, :, 0] = I
            intensities[:, :, :, 1] = 0.0
            intensities[:, :, :, 2] = 0.0
            intensities[:, :, :, 3] = I
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    else: # if polarisation is present
        intensities = np.zeros((n_pix_l, n_pix_m, nchan, ncorr), dtype=np.complex128) # create pixel grid for sky model
        if ncorr == 2: # if ncorr is 2, we only need compute XX and YY correlations
            log.warning("Only two correlations requested, but four are present in the FITS image directory. Using only Stokes I and Q.")
            I, Q, _, _ = model_cubes
            intensities[:, :, :, 0] = I + Q
            intensities[:, :, :, 1] = I - Q
        elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
            I, Q, U, V = model_cubes
            intensities[:, :, :, 0] = I + Q
            intensities[:, :, :, 1] = U + 1j * V
            intensities[:, :, :, 2] = U - 1j * V
            intensities[:, :, :, 3] = I - Q
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
        
    intensities = intensities.reshape(n_pix_l * n_pix_m, nchan, ncorr) # reshape image for compatibility with im_to_vis
    
    # set up coordinates for DFT
    ll, mm = np.meshgrid(l_coords, m_coords)
    lm = np.vstack((ll.ravel(), mm.ravel())).T

    # get only pixels with brightness > tol
    tol_mask = np.any(np.abs(intensities) > tol, axis=(1, 2))
    non_zero_intensities = intensities[tol_mask]
    non_zero_lm = lm[tol_mask]
    
    # TODO: decide whether image is sparse enough for DFT, else use FFT
    
    return non_zero_intensities, non_zero_lm


def augmented_im_to_vis(image: np.ndarray, uvw: np.ndarray, lm: np.ndarray, frequency: np.ndarray, subtract: Optional[bool]=False, mod_data: Optional[np.ndarray]=None, noise: Optional[float]=None):
    """
    Augmented version of im_to_vis
    Args:
        image: image array
        uvw: UVW coordinates
        lm: (l, m) coordinates
        frequency: frequency array
        noise: RMS noise
        subtract: True if visibilities should be subtracted from the model data, False otherwise
    Returns:
        vis: visibility array
    """
    # predict visibilities
    vis = im_to_vis(image, uvw, lm, frequency)
    
    # add noise
    if noise:
        vis += noise * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape))
    
    # subtract visibilities
    if mod_data:
        vis = vis - mod_data if subtract else vis + mod_data
        
    return vis


def computevis(srcs, uvw, freqs, ncorr, polarisation, mod_data=None, noise=None, subtract=False):
    """
    Computes visibilities

    Args:
        srcs (list):                List of Source objects
        uvw (numpy.ndarray):        Array of shape (3, nrows) containing the UVW coordinates
        freqs (numpy.ndarray):      Array of shape (nchan,) containing the frequencies
        ncorr (int):                Number of correlations
        polarisation (bool):        True if polarisation information is present, False otherwise
        mod_data (numpy.ndarray):   Array of shape (nrows, nchan, ncorr) containing the model data 
            to/from which computed visibilities should be added/subtracted
        noise (float):              RMS noise
        subtract (bool):            True if visibilities should be subtracted from the model data, False otherwise

    Returns:
        vis (numpy.ndarray):        Visibility array of shape (nrows, nchan, ncorr)
    """

    wavs = 2.99e8 / freqs
    uvw_scaled = uvw.T[...,np.newaxis] / wavs 
    
    # if polarisation is detected, we need to compute different correlations separately
    if polarisation:
        xx, yy = 0j, 0j
        if ncorr==2: # if ncorr is 2, we only need compute XX and YY correlations
            for source in srcs:
                el, em = source.l, source.m
                n_term = np.sqrt(1 - el*el - em*em) - 1
                arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
                if source.emaj in [None, "null"] and source.emin in [None, "null"]:
                    phase_factor = np.exp(-2 * np.pi * 1j * arg)
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * phase_factor # I + Q
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * phase_factor # I - Q
                else:
                    ell = source.emaj * np.sin(source.pa)
                    emm = source.emaj * np.cos(source.pa)
                    ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
                
                    fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
                    fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)

                    shape_phase = fu1 * fu1 + fv1 * fv1
                    phase_factor = np.exp(-2j*np.pi * arg - shape_phase)
                
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * phase_factor # I + Q
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * phase_factor # I - Q
                
            vis = np.stack([xx, yy], axis=2)
            
        elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
            xy, yx = 0j, 0j
            for source in srcs:
                el, em = source.l, source.m
                n_term = np.sqrt(1 - el*el - em*em) - 1
                arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
                if source.emaj in [None, "null"] and source.emin in [None, "null"]:
                    phase_factor = np.exp(-2 * np.pi * 1j * arg)
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * phase_factor       # I + Q
                    xy += (source.spectrum[2, :] + 1j * source.spectrum[3, :]) * phase_factor  # U + iV
                    yx += (source.spectrum[2, :] - 1j * source.spectrum[3, :]) * phase_factor  # U - iV
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * phase_factor       # I - Q
                else:
                    ell = source.emaj * np.sin(source.pa)
                    emm = source.emaj * np.cos(source.pa)
                    ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
                
                    fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
                    fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)

                    shape_phase = fu1 * fu1 + fv1 * fv1
                    phase_factor = np.exp(-2j*np.pi * arg - shape_phase)
                
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * phase_factor       # I + Q
                    xy += (source.spectrum[2, :] + 1j * source.spectrum[3, :]) * phase_factor  # U + iV
                    yx += (source.spectrum[2, :] - 1j * source.spectrum[3, :]) * phase_factor  # U - iV
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * phase_factor       # I - Q
            
            vis = np.stack([xx, xy, yx, yy], axis=2)
        
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    # if no polarisation is detected, we only need compute XX and duplicate to YY     
    else:
        vis = 0j    
        for source in srcs:
            el, em = source.l, source.m
            n_term = np.sqrt(1 - el*el - em*em) - 1
            arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
            if source.emaj in [None, "null"] and source.emin in [None, "null"]:
                vis += source.spectrum * np.exp(-2 * np.pi * 1j * arg)
            else:
                ell = source.emaj * np.sin(source.pa)
                emm = source.emaj * np.cos(source.pa)
                ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
            
                fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
                fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)

                shape_phase = fu1 * fu1 + fv1 * fv1
            
                vis += source.spectrum * np.exp(-2j*np.pi * arg - shape_phase)
            
        if ncorr == 2:
            vis = np.stack([vis, vis], axis=2)
        elif ncorr == 4:
            vis = np.stack([vis, np.empty_like(vis), np.empty_like(vis), vis], axis=2)
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    if noise:
        vis += noise * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape))
    
    if isinstance(mod_data, np.ndarray):
        vis = vis - mod_data if subtract else vis + mod_data
        
    return vis


def read_ms(ms: MS, spw_id: int, field_id: int, chunks: dict, sefd: float, input_column: str):
    """
    Reads MS info
    Args:
        ms: MS file
        spw_id: spectral window ID
        field_id: field ID
        chunks: dask chunking strategy
        sefd: system equivalent flux density; used if return_noise is True
        input_column: whether to read a column for manipulation
    Returns:
        ms_dsl: xarray dataset list
        ra0: RA of phase-tracking centre in radians
        dec0: Dec of phase-tracking centre in radians
        chan_freqs: MS channel frequencies
        nrows: number of rows
        nchan: number of channels
        ncorr: number of correlations
        noise: RMS noise
        input_column_data: data from input column
    """
    ms_dsl = xds_from_ms(ms, index_cols=["TIME", "ANTENNA1", "ANTENNA2"], chunks=chunks)
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    
    radec0 = field_ds.PHASE_DIR.data[field_id].compute()
    ra0, dec0 = radec0[0][0], radec0[0][1]
    chan_freqs = spw_ds.CHAN_FREQ.data[spw_id].compute()
    nrow, nchan, ncorr = ms_dsl[0].DATA.data.shape
    
    if sefd:
        df = spw_ds.CHAN_WIDTH.data[spw_id][0].compute()
        dt = ms_dsl[0].EXPOSURE.data[0].compute()
        noise = sefd / np.sqrt(2*dt*df)
    else:
        noise = None
        
    if input_column:
        input_column_data =  getattr(ms_dsl[0], input_column).data
        input_column_dims = ("row", "chan", "corr")
    else:
        input_column_data = None
        input_column_dims = None
    
    return ms_dsl, ra0, dec0, chan_freqs, nrow, nchan, ncorr, noise, input_column_data, input_column_dims
