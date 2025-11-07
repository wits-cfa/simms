from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import xarray as xr

from simms.constants import FWHM_scale_fact
from simms.skymodel.converters import convert
from simms.utilities import ParameterError as SkymodelError


def gauss_1d(xaxis: np.ndarray, peak: float, width: float, x0: float):
    """
    Function for a single gaussian line spectrum

    Args:
        xaxis (np.ndarray): x-axis grid
        peak: Gaussian peak
        x0: x-coordinate of peak
        width: FWHM width in same units as xaxis
    """
    sigma = width / FWHM_scale_fact
    return peak * np.exp(-((xaxis - x0) ** 2) / (2 * sigma**2))


def exoplanet_transient_logistic(
    start_time: int,
    end_time: int,
    ntimes: int,
    transient_start: int,
    transient_absorb: float,
    transient_ingress: int,
    transient_period: int,
):
    """
    Function for a transient profile

    Args:
        start_time (float):
            Start time of the observation (in seconds).
        end_time (float):
            End time of the observation (in seconds).
        ntimes (int):
            Total number of integrations between start and end.
        transient_start (float):
            Time at which the transit begins (seconds).
        transient_absorb (float):
            Maximum fractional flux decrease during the transit (e.g. 0.01 = 1% dip).
        transient_ingress (float):
            Duration of ingress (time it takes for the full dip to occur).
        transient_period (float):
            Total duration of the transit event, including ingress and egress.

    Returns:

    """

    # helper function to calculate ingress/egress using logistic function
    def logistic_step(z, L=10.0):
        "Logistic function mapped to [0, 1] using internal steepness scaling L."
        z = np.clip(z, 0, 1)
        k = L  # steepness across [0, 1]
        raw = 1 / (1 + np.exp(-k * (z - 0.5)))
        f0 = 1 / (1 + np.exp(k / 2))
        f1 = 1 / (1 + np.exp(-k / 2))
        normalized = (raw - f0) / (f1 - f0)
        return normalized

    times = np.linspace(start_time, end_time, ntimes)
    baseline = 1.0

    intensity = np.full_like(times, baseline, dtype=np.float64)

    ingress_start = transient_start
    ingress_end = ingress_start + transient_ingress

    egress_end = transient_start + transient_period
    egress_start = egress_end - transient_ingress

    plateau_start = ingress_end
    plateau_end = egress_start

    # Ingress
    mask_ingress = (times >= ingress_start) & (times < ingress_end)
    z_ingress = (times[mask_ingress] - ingress_start) / transient_ingress
    intensity[mask_ingress] = baseline - transient_absorb * logistic_step(z_ingress, L=10)

    # Flat bottom
    mask_plateau = (times >= plateau_start) & (times < plateau_end)
    intensity[mask_plateau] = baseline - transient_absorb

    # Egress
    mask_egress = (times >= egress_start) & (times < egress_end)
    z_egress = (times[mask_egress] - egress_start) / transient_ingress
    intensity[mask_egress] = baseline - transient_absorb * (1 - logistic_step(z_egress, L=10))

    return intensity


class StokesData:
    def __init__(self, data: List, linear_basis=True):
        """
        Object that holds a source/image intensity (stokes data)

        Args:
            data (List): List of stokes parameter data.
            linear_basis (bool, optional): Is the stokes data in a linear basis? Defaults to True.
        """
        self.data = np.array(data)
        self.linear_basis = linear_basis
        if linear_basis:
            self.param_string = "IQUV"
        else:
            self.param_string = "IVQU"

    def set_spectrum(self, freqs: np.ndarray, specfunc: Callable, full_pol: bool = True, **kwargs):
        """
        Add a spectral axis

        Args:
            freqs (np.ndarray): Array of frequencies
            specfunc (Callable): Function that
            full_pol (bool, optional): Set all 4 stokes parameters? Defaults to True.
        """
        nchan = freqs.size
        self.idx = 0
        if full_pol:
            spectrum = np.zeros([4, nchan], dtype=freqs.dtype)
            for idx, stokes_param in enumerate(self.param_string):
                flux = getattr(self, stokes_param)
                spectrum[idx, ...] = specfunc(freqs, flux, **kwargs)
        else:
            spectrum = specfunc(freqs, self.I, **kwargs)[np.newaxis, :]

        self.data = spectrum

    def set_lightcurve(self, lightcurve_func: Callable, **kwargs):
        """
        Add a time axis

        Args:
            lightcurve_func (Callable): _description_
        """
        light_curve = lightcurve_func(**kwargs)

        self.idx = 0

        ndim = self.data.ndim + 1
        slc = [np.newaxis] * ndim
        slc[1] = slice(None)
        dslice = [slice(None)] * ndim
        dslice[1] = np.newaxis

        self.data = self.data[tuple(dslice)] * light_curve[tuple(slc)]

    def __stokes_x__(self, x: str):
        """Get intensity data for stokes parameter x = I|Q|U|V

        Args:
            x (str): Stokes parameter

        Returns:
            int: Data along given stokes parameter. Retruns zero if no data found
        """

        if x not in self.param_string:
            raise RuntimeError(f"Uknown Stokes paramter '{x}'")

        try:
            xdata = self.data[self.param_string.index(x)]
        except IndexError:
            xdata = 0

        return xdata

    def get_brightness_matrix(self, ncorr: int, linear_pol_basis=True) -> np.ndarray:
        """Returns the brightness matrix of this source instance

        Args:
            ncorr (int): Number of correlations for target MS
            linear_pol_basis (bool): Is the polarisation basis linear?

        Returns:
            np.ndarray: The brightness matrix
        """

        dshape = list(self.data.shape)
        dshape[self.idx] = ncorr

        if ncorr == 2:
            # ensure dtype is a numpy type (doing this to avoid ducc0.wgridder.dirty2ms)
            # when dtype is '>f8' (from xarray) dirty2ms fails with:
            # "type matching failed: 'dirty' has neither type 'f4' nor 'f8'"
            # this also means we can't do a FFT predict in full-stokes mode
            dtype = np.finfo(self.data.dtype).dtype
            bmatrix = np.zeros(dshape, dtype=dtype)
        else:
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
                bmatrix[tslice(1)] = self.U + 1j * self.V
                bmatrix[tslice(2)] = self.U - 1j * self.V
                bmatrix[tslice(3)] = self.I - self.Q
            else:
                bmatrix[tslice(0)] = self.I + self.V
                bmatrix[tslice(1)] = self.Q + 1j * self.U
                bmatrix[tslice(2)] = self.Q - 1j * self.U
                bmatrix[tslice(3)] = self.I - self.V

        return bmatrix

    @property
    def I(self):  # noqa: E743
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


@dataclass
class CatSource:
    """
    Catalogue source dataclass

    Args:
        name (str): Source name
        ra (float | str): Right ascension
        dec (float | str): Declination
        stokes_i (float | str): Stokes I flux
        emaj (float | str, optional): Major axis size. Defaults to 0.
        emin (float | str, optional): Minor axis size. Defaults to 0.
        pa (float | str, optional): Position angle. Defaults to 0.
        stokes_q (float | str, optional): Stokes Q flux. Defaults to 0.
        stokes_u (float | str, optional): Stokes U flux. Defaults to 0.
        stokes_v (float | str, optional): Stokes V flux. Defaults to 0.
        line_peak (float | str, optional): Line peak flux. Defaults to None.
        line_width (float | str, optional): Line width. Defaults to 0.
        line_restfreq (float | str, optional): Line rest frequency. Defaults to None.
        cont_coeff_1 (float | str, optional): Continuum coefficient 1. Defaults to 0.
        cont_coeff_2 (float | str, optional): Continuum coefficient 2. Defaults to 0.
        cont_reffreq (float | str, optional): Continuum reference frequency. Defaults to None.
        transient_start (float | str, optional): Transient start time. Defaults to None.
        transient_absorb (float | str, optional): Transient absorption depth. Defaults to None.
        transient_ingress (float | str, optional): Transient ingress time. Defaults to None.
        transient_period (float | str, optional): Transient period. Defaults to None.
    """

    name: str
    ra: float | str
    dec: float | str
    stokes_i: float | str
    emaj: float | str = 0
    emin: float | str = 0
    pa: float | str = 0
    stokes_q: float | str = 0
    stokes_u: float | str = 0
    stokes_v: float | str = 0
    line_peak: float | str = None
    line_width: float | str = 0
    line_restfreq: float | str = None
    cont_coeff_1: float | str = 0
    cont_coeff_2: float | str = 0
    cont_reffreq: float | str = None
    transient_start: float | str = None
    transient_absorb: float | str = None
    transient_ingress: float | str = None
    transient_period: float | str = None

    def __post_init__(self):
        self.__update_attr__("ra", "angle_ra")
        self.__update_attr__("dec", "angle_dec")
        self.__update_attr__("emaj", "angle")
        self.__update_attr__("emin", "angle")
        self.__update_attr__("pa", "angle")
        # stokes info
        self.__update_attr__("stokes_i", "flux")
        self.__update_attr__("stokes_q", "flux")
        self.__update_attr__("stokes_u", "flux")
        self.__update_attr__("stokes_v", "flux")
        # frequency info
        self.__update_attr__("line_peak", "frequency")
        self.__update_attr__("line_width", "frequency")
        self.__update_attr__("line_restfreq", "frequency")
        self.__update_attr__("line_reffreq", "frequency")
        self.__update_attr__("cont_coeff_1", "float")
        self.__update_attr__("cont_coeff_2", "float")
        self.__update_attr__("cont_reffreq", "frequency")
        # transient info
        self.__update_attr__("transient_start", None)
        self.__update_attr__("transient_absorb", None)
        self.__update_attr__("transient_period", None)
        self.__update_attr__("transient_ingress", None)

    def __update_attr__(self, attr: str, qtype: str):
        if hasattr(self, attr):
            value = getattr(self, attr)
            setattr(self, attr, convert(value, qtype))

    @property
    def is_point(self):
        return self.emaj in ("null", None) and self.emin in (None, "null")

    @property
    def is_transient(self):
        if any(
            [
                self.transient_start not in [None, "null"],
                self.transient_period not in [None, "null"],
                self.transient_ingress not in [None, "null"],
                self.transient_absorb not in [None, "null"],
            ]
        ):
            missing = []
            for option in ["transient_start", "transient_period", "transient_ingress", "transient_absorb"]:
                val = getattr(self, option)
                if val in [None, "null"]:
                    missing.append(option)
            if missing:
                raise SkymodelError(
                    f"Transient source specification is missing required parameter(s): {', '.join(missing)}"
                )
            return True
        return False


class Source(CatSource):
    pass


class StokesDataFits(StokesData):
    def __init__(self, coord: xr.DataArray, dim_idx: int, data: np.ndarray):
        """
        Object that holds Stokes data from FITS file
        
        Args:
            coord (xr.DataArray): Coordinate information
            dim_idx (int): Dimension index for the stokes parameter
            data (np.ndarray): Stokes data array
            pol_basis (str, optional): Polarization basis. Defaults to "linear".
        """
        self.data = data
        self.nstokes = coord.size
        self.idx = dim_idx
        ndim = len(data.shape)
        self.__dslice__ = [slice(None)] * ndim

    def __stokes_x__(self, x: str):
        """Get intensity data for stokes parameter x = I|Q|U|V

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
            xdata = self.data[tuple(dslice)]
        except IndexError:
            xdata = 0

        return xdata

    @property
    def is_polarised(self):
        return self.nstokes > 1


def poly(x, coeffs):
    return np.polyval(coeffs, x)


def contspec(freqs: np.ndarray, flux: float | np.ndarray | List, coeff: float, nu_ref: float):
    """
    Returns a contiuum (power law) spectral profile

    Args:
        freqs (float): Frequency array
        flux (float): Intensity
        coeff (float|np.ndarray|List): Power law coeficient (spectral index, curvature, ...)
        nu_ref (float): Reference frequency

    Returns:
        np.ndarray: Spectral profile
    """
    if nu_ref and coeff:
        if isinstance(coeff, (list, np.ndarray)):
            if len(coeff) == 1:
                poly_pow = coeff[0]
            else:
                poly_pow = np.polynomial.Polynomial(coeff)
        else:
            poly_pow = coeff
        return flux * (freqs / nu_ref) ** (poly_pow)
    else:
        return flux * np.ones_like(freqs)
