from simms.skymodel.source_factory import singlegauss_1d, contspec
import numpy as np
from simms.constants import gauss_scale_fact, C, FWHM_scale_fact
from simms.skymodel.converters import (
    convert2float, 
    convert2Hz, 
    convertdec2rad, 
    convertra2rad,
    convert2Jy,
    convert2rad,
)


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


        
    
    def radec2lm(self, ra0, dec0):
        dra = self.ra - ra0
        self.l = np.cos(self.dec) * np.sin(dra) 
        self.m = np.sin(self.dec) * np.cos(dec0) - np.cos(self.dec) * np.sin(dec0) * np.cos(dra)
    
        return self.l, self.m
    
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
            cont_coeff_2 = data['cont_coeff_2'][1][i] if i < len(data['cont_coeff_2'][1]) else None,
            
        )
        
        source.l, source.m = source.radec2lm(ra0,dec0)
        source.spectrum = spectrum.set_spectrum(freqs)
        sources.append(source)
        
    return sources


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
    
    if polarisation:
        xx, yy = 0j, 0j
        if ncorr==2:
            print("Warning: Q, U and/or V detected but only two correlations requested. U and V will be absent from the output MS.")
            for source in srcs:
                el, em = source.l, source.m
                n_term = np.sqrt(1 - el*el - em*em) - 1
                arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
                if source.emaj in [None, "null"] and source.emin in [None, "null"]:
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * np.exp(-2 * np.pi * 1j * arg) # I + Q
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * np.exp(-2 * np.pi * 1j * arg) # I - Q
                else:
                    ell = source.emaj * np.sin(source.pa)
                    emm = source.emaj * np.cos(source.pa)
                    ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
                
                    fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
                    fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)

                    shape_phase = fu1 * fu1 + fv1 * fv1
                
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * np.exp(-2j*np.pi * arg - shape_phase) # I + Q
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * np.exp(-2j*np.pi * arg - shape_phase) # I - Q
                
            vis = np.stack([xx, yy], axis=2)
            
        elif ncorr == 4:
            xy, yx = 0j, 0j
            for source in srcs:
                el, em = source.l, source.m
                n_term = np.sqrt(1 - el*el - em*em) - 1
                arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
                if source.emaj in [None, "null"] and source.emin in [None, "null"]:
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * np.exp(-2 * np.pi * 1j * arg)       # I + Q
                    xy += (source.spectrum[2, :] + 1j * source.spectrum[3, :]) * np.exp(-2 * np.pi * 1j * arg)  # U + iV
                    yx += (source.spectrum[2, :] - 1j * source.spectrum[3, :]) * np.exp(-2 * np.pi * 1j * arg)  # U - iV
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * np.exp(-2 * np.pi * 1j * arg)       # I - Q
                else:
                    ell = source.emaj * np.sin(source.pa)
                    emm = source.emaj * np.cos(source.pa)
                    ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
                
                    fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
                    fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)

                    shape_phase = fu1 * fu1 + fv1 * fv1
                
                    xx += (source.spectrum[0, :] + source.spectrum[1, :]) * np.exp(-2j*np.pi * arg - shape_phase)       # I + Q
                    xy += (source.spectrum[2, :] + 1j * source.spectrum[3, :]) * np.exp(-2j*np.pi * arg - shape_phase)  # U + iV
                    yx += (source.spectrum[2, :] - 1j * source.spectrum[3, :]) * np.exp(-2j*np.pi * arg - shape_phase)  # U - iV
                    yy += (source.spectrum[0, :] - source.spectrum[1, :]) * np.exp(-2j*np.pi * arg - shape_phase)       # I - Q
            
            vis = np.stack([xx, xy, yx, yy], axis=2)
        
        else:
            raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")
            
    
    else:    
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
        vis = vis - mod_data if subtract else vis + mod_data # maybe we should first check if mod_data is the right shape
        
    return vis

