from typing import Optional, Union, List
from scabha.basetypes import File, MS
from simms import BIN, get_logger
from simms.skymodel.source_factory import singlegauss_1d, contspec
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from daskms import xds_from_ms, xds_from_table, xds_to_table
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

#class Shape:
#        def __init__(self, emaj, emin, pa):
        #     self.emaj = convert2rad(emaj)
        #     self.emin = convert2rad(emin)
        #     self.pa = convert2rad(pa)
            

        # def set_shape(self):
        #     if self.emaj != 'null':
        #         pass
        #     else:
        #         self.shape = 1

            
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
     
        def set_spectrum(self, freqs):
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

def makesources(data,freqs, ra0, dec0):
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
        source.spectrum = spectrum.set_spectrum(freqs)
        sources.append(source)
        
    return sources


def check_var_axis(var: str, starts_with: Optional[bool]=False):
    axis_num = 1
    while f'CTYPE{axis_num}' in header:  # Check if CTYPE{n} exists
        ctype = header[f'CTYPE{axis_num}']
        if starts_with:
            if ctype.startswith(var):
                return str(axis_num)
        else:
            if ctype == var:
                return str(axis_num)
        axis_num += 1
        
    raise SkymodelError(f"Could not find axis with CTYPE starting with {var}" if starts_with else f"Could not find axis with CTYPE {var}")


# TODO - update docs to state that we require FITS image of shape (nchan, npix_l, npix_m) i.e. the convention,
# unless the sky model is the same across frequency band
def process_fits_skymodel(input_fitsimages: Union[File, List[File]], ra0: float, dec0: float, chan_freqs: np.ndarray, ncorr: int):
    """
    Processes FITS skymodel into DFT input. The frequency interpolation part is adapted from:
    https://github.com/ratt-ru/codex-africanus/blob/master/africanus/dft/examples/predict_from_fits.py
    Args:
        input_fitsimages:    FITS image or sorted list of FITS images if polarisation is present
        ra0:                     RA of phase-tracking centre in radians
        dec0:                   Dec of phase-tracking centre in radians
        chan_freqs:         MS frequencies
        ncorr:             number of correlations
    Returns:
        intensities:    pixel-by-pixel brightness matrix
        lm:                 (l, m) coordinate grid for DFT
    """
    
    # if single fits image, turn into list so all processing is the same
    if not isinstance(input_fitsimages, list):
        input_fitsimages = [input_fitsimages]
    
    model_cubes = []
    for fits_image in input_fitsimages:
        # get header and data
        hdulist = fits.open(fits_image)
        header = hdulist[0].header
        skymodel = np.squeeze(hdulist[0].data)
        hdulist.close()
        
        wcs = WCS(header) # this knows the coordinate system of the image (e.g. FK5, Galactic or ICRS)
        
        if not wcs.has_celestial:
            raise SkymodelError("FITS image does not have celestial coordinates")
        
        ny, nx = wcs.array_shape # get image dimensions
        y_pix, x_pix = np.indices((ny, nx)) # create pixel coordinate grid
        
        world_coords = wcs.pixel_to_world(x_pix, y_pix) # convert pixel coordinates to world coordinates
        ra, dec = world_coords.ra.rad, world_coords.dec.rad
        
        # set up coordinates for DFT
        lm = radec2lm(np.array([ra0, dec0]), ra, dec)
        
        nchan = chan_freqs.size
        
        # read in spectral info
        if wcs.has_spectral:
            # find frequency axis
            freq_axis = check_var_axis("FREQ")
            
            nband = header[f"NAXIS{freq_axis}"]
            refpix_nu = header[f"CRPIX{freq_axis}"]
            delta_nu = header[f"CDELT{freq_axis}"]  # assumes units are Hz
            ref_freq = header[f"CRVAL{freq_axis}"]

            freqs = ref_freq + np.arange(1 - refpix_nu, 1 - refpix_nu + nband) * delta_nu # calculate frequencies
            
            # if frequencies do not match we need to reprojects fits cube
            if np.any(freqs != chan_freqs):
                log.warning(
                    "Reprojecting fits cube to MS freqs. " "This uses a lot of memory. "
                )
                from scipy.interpolate import RegularGridInterpolator

                # interpolate fits cube
                fits_interp = RegularGridInterpolator(
                    (freqs, l_coords, m_coords), skymodel, bounds_error=False, fill_value=None
                )
                # reevaluate at ms freqs
                l_coords, m_coords = lm[:, 0], lm[:, 1]
                vv, ll, mm = np.meshgrid(chan_freqs, l_coords, m_coords, indexing="ij")
                vlm = np.vstack((vv.ravel(), ll.ravel(), mm.ravel())).T
                model_cube = fits_interp(vlm).reshape(nchan, nx, ny)
            else:
                model_cube = skymodel
            
            # reshape model cube to (nx, ny, nchan)
            model_cube = model_cube.reshape(nx, ny, nchan)
        else:
            freqs = chan_freqs
            model_cube = np.repeat(skymodel[:, :, np.newaxis], nchan, axis=2)
        
        model_cubes.append(model_cube)
        
    intensities = np.zeros((nx, ny, nchan, ncorr)) # create pixel grid for sky model
    
    # compute sky model
    if len(model_cubes) == 1:   # if no polarisation
        I = model_cubes[0]
        intensities[:, :, :, 0] = I
        intensities[:, :, :, 1] = I
    else:
        I, Q, U, V = model_cubes
        intensities[:, :, :, 0] = I + Q
        intensities[:, :, :, 1] = U + 1j * V
        intensities[:, :, :, 2] = U - 1j * V
        intensities[:, :, :, 3] = I - Q
        
    intensities = intensities.reshape(nx * ny, nchan, ncorr) # reshape image for compatibility with im_to_vis
    
    return intensities, lm


def computevis(srcs, uvw, freqs, ncorr, polarisation, mod_data=None, noise=None, subtract=False):
    """
    Compute visibilities.

    Args:
        srcs (list): List of Source objects.
        uvw (numpy.ndarray): Array of shape (3, nrows) containing the UVW coordinates.
        freqs (numpy.ndarray): Array of shape (nchan,) containing the frequencies.
        ncorr (int): Number of correlations.
        polarisation (bool): True if polarisation information is present, False otherwise.
        mod_data (numpy.ndarray): Array of shape (nrows, nchan, ncorr) containing the model data 
            to/from which computed visibilities should be added/subtracted.
        noise (float): RMS noise.
        subtract (bool): True if visibilities should be subtracted from the model data, False otherwise.

    Returns:
        numpy.ndarray: Array of shape (nrows, nchan, ncorr) containing the visibilities.
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


def read_ms(ms: MS, spw_id: int, field_id: int, chunks: dict, df: bool=False, dt: bool=False):
    """
    Reads MS info
    Args:
        ms: MS file
        spw_id: spectral window ID
        field_id: field ID
        chunks: dask chunking strategy
        df: read channel width
        dt: read integration time
    Returns:
        ms_dsl: xarray dataset list
        ra0: RA of phase-tracking centre in radians
        dec0: Dec of phase-tracking centre in radians
        chan_freqs: MS channel frequencies
        nrows: number of rows
        nchan: number of channels
        ncorr: number of correlations
        df: channel width
        dt: integration time
    """
    ms_dsl = xds_from_ms(ms, index_cols=["TIME", "ANTENNA1", "ANTENNA2"], chunks=chunks)
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    
    radec0 = field_ds.PHASE_DIR.data[field_id].compute()
    ra0, dec0 = radec0[0][0], radec0[0][1]
    chan_freqs = spw_ds.CHAN_FREQ.data[spw_id].compute()
    nrow, nchan, ncorr = ms_dsl[0].DATA.data.shape
    
    if df:
        df = spw_ds.CHAN_WIDTH.data[opts.spwid][0].compute()
    if dt:
        dt = ms_dsl[0].EXPOSURE.data[0].compute() 
    
    return ms_dsl, ra0, dec0, chan_freqs, nrow, nchan, ncorr, df, dt