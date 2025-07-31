import numpy as np
from simms.constants import FWHM_scale_fact
import xarray as xr
from scabha.basetypes import List
from simms.skymodel.catalogue_reader import CatSource


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
    def __init__(self, data:List):
        self.data = np.array(data)

    def set_spectrum(self, freqs, specfunc,
                full_pol=True, **kwargs):
        
        nchan = len(freqs)
        if full_pol:
            spectrum = np.zeros([4,nchan], dtype=freqs.dtype)
            for idx,stokes_param in enumerate("IQUV"):
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
        stokes_types = dict(I=0, Q=1, U=2, V=3)
        
        if x not in stokes_types.keys():
            raise RuntimeError(f"Uknown Stokes paramter '{x}'")
        
        try:
            xdata =  self.data[stokes_types[x]]
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
       
        dshape = [ncorr] + list(self.I.shape)
        bmatrix = np.ones(dshape, dtype=self.data.dtype) * 1j 
        if linear_pol_basis:
            stokes_params = "IQUV"
        else:
            stokes_params = "IVQU"
        
        if ncorr == 2:
            bmatrix[0,...] = self.I + getattr(self, stokes_params[1])
            bmatrix[1,...] = self.I - getattr(self, stokes_params[1])
        else:
            bmatrix[0,...] = self.I + getattr(self, stokes_params[1])
            bmatrix[1,...] = getattr(self, stokes_params[2]) + 1j*getattr(self, stokes_params[3])
            bmatrix[2,...] = getattr(self, stokes_params[2]) - 1j*getattr(self, stokes_params[3])
            bmatrix[3,...] = self.I - getattr(self, stokes_params[1])
        
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
    
    
class Source(CatSource):
    
        pass
        
    

class StokesDataFits(StokesData):
    def __init__(self, coord:xr.DataArray, data:np.ndarray, pol_basis="linear"):
        self.data = data
        self.nstokes = coord.size
        self.idx = coord.dim_idx
        ndim = len(data.shape)
        self.__dslice__ = [slice(None)]*ndim
        self.pol_basis = pol_basis
    
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
