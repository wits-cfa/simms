from typing import Optional, Union, List
from scabha.basetypes import File, MS
from simms import BIN, get_logger
from simms.utilities import FITSSkymodelError as SkymodelError
from simms.skymodel.source_factory import singlegauss_1d, contspec
from numba import njit, prange
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from daskms import xds_from_ms, xds_from_table
from africanus.dft import im_to_vis as dft_im_to_vis
from ducc0.wgridder import dirty2ms
from simms.constants import gauss_scale_fact, C, FWHM_scale_fact
from simms.skymodel.converters import (
    convert2float,
    convert2Hz,
    convert2Jy,
    convert2rad,
    convertdec2rad,
    convertra2rad,
    radec2lm
)
from simms.skymodel.source_factory import contspec, singlegauss_1d


log = get_logger(BIN.skysim)


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
    
    df = spw_ds.CHAN_WIDTH.data[spw_id][0].compute()
    
    if sefd:
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
    
    return ms_dsl, ra0, dec0, chan_freqs, nrow, nchan, df, ncorr, noise, input_column_data, input_column_dims


def add_noise(vis, noise):
    if noise:
        vis += noise * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape))
    else:
        pass
    

def add_to_column(vis, mod_data, mode):
    if mode in ['subtract', 'add']:
        vis = mod_data - vis if mode == 'subtract' else vis + mod_data
    else:
        pass


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
        self.l, self.m = radec2lm(ra0, dec0, self.ra, self.dec)
    
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


def make_sources(data, freqs, ra0, dec0):
    num_sources = len(data['name'][1])
    sources = []
    for i in range(num_sources):
        source = Source(
            name=data["name"][1][i],
            ra=data["ra"][1][i],
            dec=data["dec"][1][i],
            emaj=data["emaj"][1][i] if i < len(data["emaj"][1]) else None,
            emin=data["emin"][1][i] if i < len(data["emin"][1]) else None,
            pa=data["pa"][1][i] if i < len(data["pa"][1]) else "0deg",
        )

        spectrum = Spectrum(
            stokes_i=data["stokes_i"][1][i],
            stokes_q=data["stokes_q"][1][i] if i < len(data["stokes_q"][1]) else None,
            stokes_u=data["stokes_u"][1][i] if i < len(data["stokes_u"][1]) else None,
            stokes_v=data["stokes_v"][1][i] if i < len(data["stokes_v"][1]) else None,
            cont_reffreq=(
                data["cont_reffreq"][1][i] if i < len(data["cont_reffreq"][1]) else None
            ),
            line_peak=(
                data["line_peak"][1][i] if i < len(data["line_peak"][1]) else None
            ),
            line_width=(
                data["line_width"][1][i] if i < len(data["line_width"][1]) else None
            ),
            line_restfreq=(
                data["line_restfreq"][1][i]
                if i < len(data["line_restfreq"][1])
                else None
            ),
            cont_coeff_1=(
                (data["cont_coeff_1"][1][i])
                if i < len(data["cont_coeff_1"][1])
                else None
            ),
            cont_coeff_2=(
                data["cont_coeff_2"][1][i] if i < len(data["cont_coeff_2"][1]) else None
            ),
        )
        
        source.set_lm(ra0, dec0)
        source.spectrum = spectrum.make_spectrum(freqs)
        sources.append(source)

    return sources


def compute_brightness_matrix(spectrum: np.ndarray, elements: str, basis: str):
    """
    Computes the brightness matrix for a given spectrum and basis
    Args:
        spectrum: spectrum array
        elements: elements to compute
        basis: polarisation basis ("linear" or "circular")
    Returns:
        brightness_matrix: brightness matrix elements
    """
    if basis == "linear":
        order = [0, 1, 2, 3] # I, Q, U, V
    elif basis == "circular":
        order = [0, 3, 1, 2] # I, V, Q, U
    else:
        raise ValueError(f"Unrecognised polarisation basis '{basis}'. Use 'linear' or 'circular'.")
    
    if elements == 'diagonal':
        return (spectrum[order[0], :] + spectrum[order[1], :], 
                spectrum[order[0], :] - spectrum[order[1], :])
    elif elements == 'all':
        return (spectrum[order[0], :] + spectrum[order[1], :], 
                spectrum[order[2], :] + 1j * spectrum[order[3], :],
                spectrum[order[2], :] - 1j * spectrum[order[3], :],
                spectrum[order[0], :] - spectrum[order[1], :])
    else:
        raise ValueError(f"Unrecognised elements '{elements}'. Use 'diagonal' or 'all'.")
    

def stack_unpolarised_vis(vis: np.ndarray, ncorr: int) -> np.ndarray:
    """
    Takes XX or RR visibilities and creates visibility array with shape (nrows, nchan, ncorr).
    Used to avoid double computation of identical correlations.
    
    Args:
        vis: numpy array of shape (nrows, nchan) containing the visibility data
        ncorr: number of correlations (2 or 4)
    Returns:
        vis: numpy array of shape (nrows, nchan, ncorr) containing the visibility data
    """
    if ncorr == 2:
        vis = np.stack([vis, vis], axis=2)
    elif ncorr == 4:
        vis = np.stack([vis, np.zeros_like(vis), np.zeros_like(vis), vis], axis=2)
    else:
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    return vis


def compute_vis(srcs: List[Source], uvw: np.ndarray, freqs: np.ndarray, ncorr: int, polarisation: bool, basis: str,
                mode: Union[None, str], mod_data: Union[None, np.ndarray], noise: Optional[float] = None):
    """
    Computes visibilities

    Args:
        srcs (list):                List of Source objects
        uvw (numpy.ndarray):        Array of shape (3, nrows) containing the UVW coordinates
        freqs (numpy.ndarray):      Array of shape (nchan,) containing the frequencies
        ncorr (int):                Number of correlations
        polarisation (bool):        True if polarisation information is present, False otherwise
        basis (str):                Polarisation basis ("linear" or "circular")
        mod_data (numpy.ndarray):   Array of shape (nrows, nchan, ncorr) containing the model data 
            to/from which computed visibilities should be added/subtracted
        noise (float):              RMS noise
        subtract (bool):            True if visibilities should be subtracted from the model data, False otherwise

    Returns:
        vis (numpy.ndarray):        Visibility array of shape (nrows, nchan, ncorr)
    """

    wavs = 2.99e8 / freqs
    uvw_scaled = uvw.T[...,np.newaxis] / wavs
    
    # helper function to calculate phase factor
    def calculate_phase_factor(source, uvw_scaled):
        el, em = source.l, source.m
        n_term = np.sqrt(1 - el*el - em*em) - 1
        arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
        
        if source.emaj in [None, "null"] and source.emin in [None, "null"]:
            # point source
            return np.exp(2 * np.pi * 1j * arg)
        else:
            # extended source
            ell = source.emaj * np.sin(source.pa)
            emm = source.emaj * np.cos(source.pa)
            ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
            
            fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
            fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)
            
            shape_phase = fu1 * fu1 + fv1 * fv1
            return np.exp(2 *np.pi * 1j * arg - shape_phase)
    
    # if polarisation is detected, we need to compute different correlations separately
    if polarisation:
        xx, yy = 0j, 0j
        if ncorr == 2:  # if ncorr is 2, we only need compute XX and YY correlations
            for source in srcs:
                phase_factor = calculate_phase_factor(source, uvw_scaled)
                source_xx, source_yy = compute_brightness_matrix(source.spectrum, 'diagonal', basis)
                xx += source_xx * phase_factor
                yy += source_yy * phase_factor
                
            vis = np.stack([xx, yy], axis=2)

        elif ncorr == 4:  # if ncorr is 4, we need to compute all correlations
            xy, yx = 0j, 0j
            for source in srcs:
                phase_factor = calculate_phase_factor(source, uvw_scaled)
                source_xx, source_xy, source_yx, source_yy = compute_brightness_matrix(source.spectrum, 'all', basis)
                xx += source_xx * phase_factor
                xy += source_xy * phase_factor
                yx += source_yx * phase_factor
                yy += source_yy * phase_factor
            
            vis = np.stack([xx, xy, yx, yy], axis=2)

        else:
            raise ValueError(
                f"Only two or four correlations allowed, but {ncorr} were requested."
            )

    # if no polarisation is detected, we only need compute XX and duplicate to YY
    else:
        vis = 0j
        for source in srcs:
            phase_factor = calculate_phase_factor(source, uvw_scaled)
            vis += source.spectrum * phase_factor
            
        vis = stack_unpolarised_vis(vis, ncorr)
    
    # add noise
    add_noise(vis, noise)
    # do addition/subtraction of model data
    add_to_column(vis, mod_data, mode)
    
    return vis


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
    

def compute_radec_coords(header, n_ra: float, n_dec: float):
    """
    Calculates pixel (RA, Dec) coordinates
    Args:
        header (FITS header): FITS header
        n_ra (float): number of RA pixels
        n_dec (float): number of Dec pixels
    """
    ra_axis = check_var_axis(header, "RA")
    dec_axis = check_var_axis(header, "DEC")
    delta_ra = header[f"CDELT{ra_axis}"]
    delta_dec = header[f"CDELT{dec_axis}"]
    
    # get reference pixel info
    refpix_ra = header[f"CRPIX{ra_axis}"]
    refpix_dec = header[f"CRPIX{dec_axis}"]
    ref_ra = header[f"CRVAL{ra_axis}"]
    ref_dec = header[f"CRVAL{dec_axis}"]
    
    if header[f"CUNIT{ra_axis}"] == header[f"CUNIT{dec_axis}"]:
        if header[f"CUNIT{ra_axis}"] in ["DEG", "deg"]:
            ref_ra = np.deg2rad(ref_ra)
            ref_dec = np.deg2rad(ref_dec)
            delta_ra = np.deg2rad(delta_ra)
            delta_dec = np.deg2rad(delta_dec)
        elif header[f"CUNIT{ra_axis}"] in ["RAD", "rad"]:
            pass
        else:
            raise SkymodelError("RA and Dec units must be in degrees or radians")
    else:
        raise SkymodelError("RA and Dec units must be the same")
    
    # calculate pixel (RA, Dec) coordinates
    ra_coords = ref_ra + (np.arange(1, n_ra + 1) - refpix_ra)  * delta_ra
    dec_coords = ref_dec + (np.arange(1, n_dec + 1) - refpix_dec) * delta_dec
    
    return ra_coords, delta_ra, dec_coords, delta_dec


# TODO: consider assuming degrees for RA and Dec if no units are given
def compute_lm_coords(header, phase_centre: np.ndarray, n_ra: float, n_dec: float, ra_coords: Optional[np.ndarray]=None,
                    delta_ra: Optional[float]=None, dec_coords: Optional[np.ndarray]=None, delta_dec: Optional[float]=None,
                    tol_mask: Optional[np.ndarray]=None):
    """
    Calculates pixel (l, m) coordinates
    """
    if not isinstance(ra_coords, np.ndarray) or not isinstance(dec_coords, np.ndarray):
        ra_coords, delta_ra, dec_coords, delta_dec = compute_radec_coords(header, n_ra, n_dec)
    
    # calculate pixel (l, m) coordinates
    ra0, dec0 = phase_centre
    lm = pix_radec2lm(ra0, dec0, ra_coords, dec_coords)
    
    if isinstance(tol_mask, np.ndarray):
        # reshape lm for DFT
        reshaped_lm = lm.reshape(n_ra * n_dec, 2)
        non_zero_lm = reshaped_lm[tol_mask]
        return non_zero_lm
    
    return lm
    

def process_fits_skymodel(input_fitsimages: Union[File, List[File]], ra0: float, dec0: float, chan_freqs: np.ndarray,
                          ms_delta_nu: float, ncorr: int, basis: str, tol: float=1e-7, stokes: int = 0,
                          use_dft: Optional[bool]=None) -> tuple:
    """
    Processes FITS skymodel into DFT input
    Args:
        input_fitsimages: FITS image or sorted list of FITS images if polarisation is present
        ra0:                     RA of phase-tracking centre in radians
        dec0:                   Dec of phase-tracking centre in radians
        chan_freqs:         MS frequencies
        ms_delta_nu:     MS channel width
        ncorr:             number of correlations
        basis:             polarisation basis ("linear" or "circular")
        tol:                tolerance for pixel brightness
        stokes:             Stokes parameter to use (0 = I, 1 = Q, 2 = U, 3 = V)
    Returns:
        intensities:    pixel-by-pixel brightness matrix for each channel and correlation
        lm:                 (l, m) coordinate grid for DFT
    """
    
    # if single fits image, turn into list so all processing is the same
    if not isinstance(input_fitsimages, list):
        input_fitsimages = [input_fitsimages]
    
    phase_centre = np.array([ra0, dec0])
    nchan = chan_freqs.size
    
    model_cubes = []
    ra_coords, delta_ra, dec_coords, delta_dec = None, None, None, None
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
                    if dim_name == "stokes":
                        log.warning(f"Using only Stokes parameter at index {stokes} from FITS image. Use separate files for full Stokes models.")
                        data_slice[i] = stokes
                    # other axes
                    else:
                        data_slice[i] = 0
                        if header[f"NAXIS{i+1}"] > 1:
                            if dim_name == "time":
                                log.warning(f"Removing 'TIME' axis with size > 1. Using only first time stamp. Use separate files for time-varying models.")
                            else:
                                log.warning(f"Removing '{dim_name.upper()}' axis with size > 1. Using only first element.")
            
            skymodel = hdulist[0].data[tuple(data_slice)]
        
        # # TODO (Mika, Senkhosi): assume image in l-m coords already if no celestial coords
        # # Would this mean the lengths of CTYPE and CRVAL are not the same? Is that even possible?
        if 'ra' not in dims or 'dec' not in dims:
            raise SkymodelError("FITS image does not have or has unrecognised RA/Dec coordinates")
        
        # spectral axis exists
        if "freq" in dims:
            freq_axis = check_var_axis(header, "FREQ")
            n_freqs = header[f"NAXIS{freq_axis}"]

            # get frequency info
            refpix_nu = header[f"CRPIX{freq_axis}"]
            if header[f"CUNIT{freq_axis}"] == "Hz":
                fits_delta_nu = header[f"CDELT{freq_axis}"]  # assumes units are Hz
            else:
                raise SkymodelError("Frequency units must be in Hz")
            ref_freq = header[f"CRVAL{freq_axis}"]
            
            # computes edges of FITS and MS frequency axes
            ms_start_freq = chan_freqs[0] - 0.5*(ms_delta_nu)
            ms_end_freq = chan_freqs[-1] + 0.5*(ms_delta_nu)
            
            fits_start_freq = ref_freq - (refpix_nu - 1 + 0.5) * fits_delta_nu
            fits_end_freq = fits_start_freq  + (n_freqs * fits_delta_nu)
            
            # if spectral axis is not singleton
            if n_freqs > 1:
                # construct frequency axis
                freqs = ref_freq + (np.arange(1, n_freqs + 1) - refpix_nu) * fits_delta_nu
                
                if ms_start_freq < fits_start_freq or ms_end_freq > fits_end_freq:
                    raise SkymodelError(f"Some MS frequencies [{ms_start_freq/1e9:.6f} GHz, {ms_end_freq/1e9:.6f} GHz] "
                                        f"are out of bounds of FITS image frequencies[{fits_start_freq/1e9:.6f} GHz, {fits_end_freq/1e9:.6f} GHz]. "
                                        "Cannot interpolate FITS image onto MS frequency grid.")
                
                # reshape FITS data to (n_pix_l, n_pix_m, nchan)
                skymodel = np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec"), dims.index("freq")))
                
                # get image shape
                n_pix_l, n_pix_m, _ = skymodel.shape
            
                if len(chan_freqs) != len(freqs) or np.any(freqs != chan_freqs):
                    # interpolate FITS cube
                    log.warning(f"Interpolating {fits_image} onto MS channel frequency grid. This uses a lot of memory.")
                    ra_coords, delta_ra, dec_coords, delta_dec = compute_radec_coords(header, n_pix_l, n_pix_m)
                    fits_interp = RegularGridInterpolator((ra_coords, dec_coords, freqs), skymodel)
                    ra, dec, vv = np.meshgrid(ra_coords, dec_coords, chan_freqs, indexing="ij")
                    radecv = np.vstack((ra.ravel(), dec.ravel(), vv.ravel())).T
                    model_cube = fits_interp(radecv).reshape(n_pix_l, n_pix_m, nchan)
                    
                else:
                    model_cube = skymodel
            
            else: # singleton spectral axis
                # raise error if frequency is not in bounds of MS channel frequencies
                if fits_start_freq < ms_start_freq or fits_end_freq > ms_end_freq:    
                    raise SkymodelError(f"{fits_image} frequency range does not fall in MS channel frequency range.")
                
                # reshape FITS data to (n_pix_l, n_pix_m, nchan)
                skymodel= np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec"), dims.index("freq")))
                skymodel = np.squeeze(skymodel)
                
                # get image shape
                n_pix_l, n_pix_m = skymodel.shape

                freqs = chan_freqs
                # repeat the image along the frequency axis
                model_cube = np.repeat(skymodel[:, :, np.newaxis], nchan, axis=2)

        # no spectral axis
        else:
            # reshape FITS data to (n_pix_l, n_pix_m)
            skymodel = np.transpose(skymodel, axes=(dims.index("ra"), dims.index("dec")))
            
            # get image shape
            n_pix_l, n_pix_m = skymodel.shape
        
            freqs = chan_freqs
            # repeat the image along the frequency axis
            model_cube = np.repeat(skymodel[:, :, np.newaxis], nchan, axis=2)
        
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
            intensities[:, :, :, 3] = I
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
    
    else: # if polarisation is present
        intensities = np.zeros((n_pix_l, n_pix_m, nchan, ncorr), dtype=np.complex128) # create pixel grid for sky model
        if basis == "linear":
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
        elif basis == "circular":
            if ncorr == 2: # if ncorr is 2, we only need compute XX and YY correlations
                log.warning("Only two correlations requested, but four are present in the FITS image directory. Using only Stokes I and V.")
                I, _, _, V = model_cubes
                intensities[:, :, :, 0] = I + V
                intensities[:, :, :, 1] = I - V
            elif ncorr == 4: # if ncorr is 4, we need to compute all correlations
                I, Q, U, V = model_cubes
                intensities[:, :, :, 0] = I + V
                intensities[:, :, :, 1] = Q + 1j * U
                intensities[:, :, :, 2] = Q - 1j * U
                intensities[:, :, :, 3] = I - V
            else:
                raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
        else:
            raise ValueError(f"Unrecognised polarisation basis '{basis}'. Use 'linear' or 'circular'.")
        
    # get spatial coordinate info
    if ra_coords is None:
        ra_coords, delta_ra, dec_coords, delta_dec = compute_radec_coords(header, n_pix_l, n_pix_m)
    
    # reshape intensities to im_to_vis expectations
    reshaped_intensities = intensities.reshape(n_pix_l * n_pix_m, nchan, ncorr)
    
    # get only pixels with brightness > tol
    tol_mask = np.any(np.abs(reshaped_intensities) > tol, axis=(1, 2))
    non_zero_intensities = reshaped_intensities[tol_mask]
    
    # decide whether image is sparse enough for DFT
    sparsity = 1 - (non_zero_intensities.size/intensities.size)
    
    if use_dft is None:
        if sparsity >= 0.8:
            log.info(f"More than 80% of pixels have intensity < {(tol*1e6):.2f} μJy. DFT will be used for visibility prediction.")
            use_dft = True
            non_zero_lm = compute_lm_coords(header, phase_centre, n_pix_l, n_pix_m, ra_coords, delta_ra, dec_coords, delta_dec, tol_mask)
            
            return non_zero_intensities, non_zero_lm, polarisation, use_dft, None, None
        else:
            log.info(f"More than 20% of pixels have intensity > {(tol*1e6):.2f} μJy. FFT will be used for visibility prediction.")
            use_dft = False
            
            return intensities, None, polarisation, use_dft, delta_ra, delta_dec
    else:
        log.info(f"Filtered {sparsity*100:.2f}% of pixels using {(tol*1e6):.2f}-μJy tolerance.")
        non_zero_lm = compute_lm_coords(header, phase_centre, n_pix_l, n_pix_m, ra_coords, delta_ra, dec_coords, delta_dec, tol_mask)
        
        return non_zero_intensities, non_zero_lm, polarisation, use_dft, None, None
    
    
def fft_im_to_vis(uvw: np.ndarray, chan_freq: np.ndarray, image: np.ndarray, pixsize_x: float, pixsize_y: float,
                  epsilon: Optional[float]=1e-7, nthreads: Optional[int]=8, do_wstacking: Optional[bool]=True) -> np.ndarray:
    """
    ducc0.wgridder.dirty2ms wrapper to add squeezing and conjugation.
    NB: Image should be 2D.
    """
    result = dirty2ms(
        uvw,
        chan_freq,
        image,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wstacking=True,
        nthreads=nthreads
    )
    
    return np.conj(np.squeeze(result))
    
    
def augmented_im_to_vis(image: np.ndarray, uvw: np.ndarray, lm: Union[None, np.ndarray], chan_freqs: np.ndarray,
                        polarisation: bool, use_dft: bool, mode: Union[None, str], mod_data: Union[None, np.ndarray],
                        ncorr: int, delta_ra: Optional[int]=None, delta_dec: Optional[int]=None, do_wstacking: Optional[bool]=True,
                        epsilon: Optional[float]=1e-7, noise: Optional[float]=None, nthreads: Optional[int]=8,):
    """
    Augmented version of im_to_vis
    Args:
        image: image array
        uvw: UVW coordinates
        lm: (l, m) coordinates (used for DFT)
        chan_freqs: frequency array
        polarisation: True if polarisation information is present, False otherwise (used for FFT)
        use_dft: True if DFT should be used, False if FFT should be used
        mode: 'add' or 'subtract' to specify whether to add or subtract model data
        mod_data: model data to/from which computed visibilities should be added/subtracted
        ncorr: number of correlations (must be 2 or 4; used for FFT)
        delta_ra: pixel size in RA direction (used for FFT)
        delta_dec: pixel size in Dec direction (used for FFT)
        epsilon: numerical precision for FFT
        noise: RMS noise
        nthreads: number of threads to use for FFT
    Returns:
        vis: visibility array
    """
    # if sparse, use DFT
    if use_dft:
        if polarisation:
            vis = dft_im_to_vis(image, uvw, lm, chan_freqs, convention='casa')
        else:
            image = image[..., 0]
            vis = dft_im_to_vis(image[..., np.newaxis], uvw, lm, chan_freqs, convention='casa')
            vis = stack_unpolarised_vis(vis[...,0], ncorr)

    # else, use FFT
    else:
        image = np.transpose(image, axes=(3, 2, 0, 1))
        if polarisation:
            vis = np.zeros((uvw.shape[0], chan_freqs.size, ncorr), dtype=np.complex128)
            for corr in range(ncorr):
                for chan in range(chan_freqs.size):
                    vis[:, chan, corr] = fft_im_to_vis(
                        uvw,
                        np.array([chan_freqs[chan]]),
                        image[corr, chan],
                        pixsize_x=np.abs(delta_ra),
                        pixsize_y=delta_dec,
                        epsilon=epsilon,
                        do_wstacking=do_wstacking,
                        nthreads=nthreads
                    )
        else:
            vis = np.zeros((uvw.shape[0], chan_freqs.size), dtype=np.complex128)
            for chan in range(chan_freqs.size):
                vis[:, chan] = fft_im_to_vis(
                    uvw,
                    np.array([chan_freqs[chan]]),
                    image[0, chan],
                    pixsize_x=np.abs(delta_ra),
                    pixsize_y=delta_dec,
                    epsilon=epsilon,
                    do_wstacking=do_wstacking,
                    nthreads=nthreads
                )
            
            vis = stack_unpolarised_vis(vis, ncorr)

    # add noise
    add_noise(vis, noise)
    # do addition/subtraction of model data
    add_to_column(vis, mod_data, mode)
    
    return vis
