import numpy as np
from simms.constants import FWHM_scale_fact
import xarray as xr
from scabha.basetypes import List
from simms.skymodel.converters import (
    convert2float,
    convert2Hz,
    convert2Jy,
    convert2rad,
    convertdec2rad,
    convertra2rad,
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
    def __init__(self, data:List, linear_basis=True):
        self.data = np.array(data)
        self.linear_basis = linear_basis
        if linear_basis:
            self.param_string = "IQUV"
        else:
            self.param_string = "IVQU"
            
    def set_spectrum(self, freqs, specfunc,
                full_pol=True, **kwargs):
        
        nchan = len(freqs)
        if full_pol:
            spectrum = np.zeros([4,nchan], dtype=freqs.dtype)
            for idx, stokes_param in enumerate(self.param_string):
                flux = getattr(self, stokes_param)
                spectrum[idx,...] = specfunc(freqs=freqs, flux=flux, **kwargs)
        else:
            spectrum = specfunc(freqs=freqs, flux=self.I, **kwargs)[np.newaxis,:]
        
        self.data = spectrum

        
    def __stokes_x__(self, x:str):
        """ Get intensity data for stokes parameter x = I|Q|U|V

        Args:
            x (str): Stokes parameter

        Returns:
            int: Data along given stokes parameter. Retruns zero if no data found
        """
        
        
        if x not in self.param_string:
            raise RuntimeError(f"Uknown Stokes paramter '{x}'")
        
        try:
            xdata =  self.data[self.param_string.index(x)]
        except IndexError:
            xdata = 0
        
        return xdata
    
    def get_brightness_matrix(self, ncorr:int, linear_pol_basis=True) -> np.ndarray:
        """AI is creating summary for get_brightness_matrix

        Args:
            ncorr (int): [description]
            linear_pol_basis (bool): [description]

        Returns:
            np.ndarray: [description]
        """
        
        dshape = list(self.data.shape)
        dshape[self.idx] = ncorr
        
        bmatrix = np.zeros(dshape, dtype=np.complex128)
        
        def tslice(i):
            dslice = [slice(None)] * self.data.ndim
            dslice[self.idx] = i
            return tuple(dslice)
        
        if ncorr == 2:
            if linear_pol_basis:
                bmatrix[tslice(0)] = self.I + self.Q
                bmatrix[tslice(1)] = self.I - self.Q
            else:
                bmatrix[tslice(0)] = self.I + self.V
                bmatrix[tslice(1)] = self.I - self.V
        else:
            if linear_pol_basis:
                bmatrix[tslice(0)] = self.I + self.Q
                bmatrix[tslice(1)] = self.U + 1j*self.V
                bmatrix[tslice(2)] = self.U - 1j*self.V
                bmatrix[tslice(3)] = self.I - self.Q
            else:
                bmatrix[tslice(0)] = self.I + self.V
                bmatrix[tslice(1)] = self.Q + 1j*self.U
                bmatrix[tslice(2)] = self.Q - 1j*self.U
                bmatrix[tslice(3)] = self.I - self.V
                
        return bmatrix
            
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
    
    @property
    def is_polarised(self):
        return any([self.Q, self.U, self.V])
    
class CatSource: 
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
        self.shape = None
        self.emaj = convert2rad(emaj,0)
        self.emin = convert2rad(emin,0)
        self.pa = convert2rad(pa,0)

    def add_stokes(self, stokes_i:int, stokes_q:int, stokes_u:int, stokes_v:int): 
        # Intensity
        self.stokes_i = convert2Jy(stokes_i,0),
        self.stokes_q = convert2Jy(stokes_q,0),
        self.stokes_u = convert2Jy(stokes_u,0),
        self.stokes_v = convert2Jy(stokes_v,0),
        
    def add_spectral(self, line_peak, line_width, line_restfreq,
                cont_reffreq, cont_coeff_1, cont_coeff_2):
    # Frequency info
        self.line_peak = convert2Hz(line_peak,False)
        self.line_width = convert2Hz(line_width,0)
        self.line_restfreq = convert2Hz(line_restfreq, None)
        self.cont_reffreq = convert2Hz(cont_reffreq, None)
        self.cont_coeff_1 = convert2float(cont_coeff_1, null_value=0)
        self.cont_coeff_2 = convert2float(cont_coeff_2, null_value=0)

    @property
    def is_point(self):
        return self.emaj in  ('null',None) and self.emin in (None, 'null') 

class Source(CatSource):
    
        pass
        
    

class StokesDataFits(StokesData):
    def __init__(self, coord:xr.DataArray, dim_idx:int,
            data:np.ndarray):
        """_summary_

        Args:
            coord (xr.DataArray): _description_
            dim_idx (int): _description_
            data (np.ndarray): _description_
            pol_basis (str, optional): _description_. Defaults to "linear".
        """
        self.data = data
        self.nstokes = coord.size
        self.idx = dim_idx
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
    def is_polarised(self):
        return self.nstokes > 1


def poly(x, coeffs):
    return np.polyval(coeffs, x)

def contspec(freqs,flux, coeff,nu_ref):
    if coeff == 0 or nu_ref == 0:
        return flux * np.ones_like(freqs)
    else:
        return flux*(freqs/nu_ref)**(coeff)
