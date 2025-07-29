import numpy as np
from simms.constants import FWHM_scale_fact
import xarray as xr
from simms.skymodel.converters import (
    convert2float,
    convert2Hz,
    convert2Jy,
    convert2rad,
    convertdec2rad,
    convertra2rad,
    radec2lm
)

def gauss_1d(xaxis:np.ndarray, peak:float, width:float, x0:float):
    """
    Function for a single gaussian line spectrum
    
    Args:
        xaxis (np.ndarray): x-axis grid
        peak: Gaussian peak
        x0: x-coordinate of peak
        width: FWHM width in same units as xaxis
    """
    sigma = width / FWHM_scale_fact
    return peak*np.exp(-(xaxis-x0)**2/(2*sigma**2))


class StokesData:
    def __init__(self, coord:xr.DataArray, data:np.ndarray):
        self.data = data
        self.nstokes = coord.size
        self.idx = coord.dim_idx
        ndim = len(data.shape)
        self.__dslice__ = [slice(None)]*ndim
    
    def __stokes_x__(self, x:str):
        """ Get intensity data for stokes parameter x = I|Q|U|V

        Args:
            x (str): Stokes parameter

        Returns:
            np.ndarray | int: Data along given stokes parameter. Retruns zero if no data found
        """
        stokes_types = dict(I=0, Q=1, U=2, V=3)
        if x not in stokes_types.keys():
            raise RuntimeError(f"Uknown Stokes paramter '{x}'")
        
        dslice = list(self.__dslice__)
        dslice[self.idx] = stokes_types[x]
        
        try:
            xdata =  self.data[tuple(dslice)]
        except IndexError:
            xdata = 0
        
        return xdata
    
    @property
    def I(self):
        return self.__stokes_x__("I")
    
    
    @property
    def Q(self):
        return self.__stokes_x__("Q")
    
    
    @property
    def U(self):
        return self.__stokes_x__("U")

    @property
    def V(self):
        return self.__stokes_x__("V")

class Source: 
    def __init__(self, name:str, ra:float, dec:float,
            emaj:float, emin:float, pa:float):
        """AI is creating summary for __init__

        Args:
            name (str): [description]
            ra (float): [description]
            dec (float): [description]
            emaj (float): [description]
            emin (float): [description]
            pa (float): [description]
        """
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
    def __init__(self, stokes_i, stokes_q, stokes_u,
                 stokes_v, line_peak, line_width,
                 line_restfreq, cont_reffreq,
                 cont_coeff_1, cont_coeff_2):
        
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
                return gauss_1d(freqs, self.stokes_i, self.line_width, self.line_peak)
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
                spectrum.append(gauss_1d(freqs, stokes_param, self.line_width, self.line_peak))
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


def poly(x, coeffs):
    return np.polyval(coeffs, x)

def contspec(freqs,flux, coeff,nu_ref):
    return flux*(freqs/nu_ref)**(coeff)
