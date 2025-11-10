from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import xarray as xr
from scabha.basetypes import EmptyListDefault

from simms.constants import FWHM_scale_fact
from simms.exceptions import ASCIISourceError


@dataclass
class SourceType:
    name: str
    required: List[str] = EmptyListDefault()
    inherit: Any = None
    surplus: List[str] = EmptyListDefault()

    def __post_init__(self):
        parent = getattr(self, "inherit", None)
        from_parent = []
        if parent:
            from_parent = parent.required

        self.required_no_parent = self.required
        updated_required = from_parent + self.required
        self.required = updated_required

        # first absolute reqs
        self.abs_reqs = list(filter(lambda item: "|" not in item, self.required))
        # then get conditional reqs
        self.cond_reqs = list(filter(lambda item: "|" in item, self.required))
        # group them in a nested list
        self.cond_req_groups = []
        for conds in self.cond_reqs:
            self.cond_req_groups.append(conds.split("|"))

    def is_valid(self, fields: List[str], raise_exception: bool = False, none_or_all: bool = False):
        # this removes fields that are in abs_reqs
        missing_abs_req = list(set(self.abs_reqs) - set(fields))
        fields_no_abs_reqs = set(fields) - set(self.abs_reqs)

        missing_cond_reqs = []
        missing_cond_reqs_flat = []
        # now remove fields in the conditional groups
        for group in self.cond_req_groups:
            # if one exists, remove the full group
            if fields_no_abs_reqs.isdisjoint(group):
                missing_cond_reqs.append("|".join(group))
                missing_cond_reqs_flat + group

        missing = missing_abs_req + missing_cond_reqs
        error_message = f"Source type '{self.name}' is missing required source fields: {missing}"
        if missing:
            if raise_exception:
                raise ASCIISourceError(error_message)
            elif none_or_all and set(self.required_no_parent).intersection(fields):
                raise ASCIISourceError(error_message)
            else:
                return False
        else:
            return True

    def __str__(self):
        required = f"\nRequired fields: {self.required}"
        surplus = f"\nOptional fields: {self.surplus}" if self.surplus else ""

        return f"Simms sky model source type: {self.name}" + required + surplus


# These are are the supported source types
point_source = SourceType(
    "Point Source", required=["ra|ra_h|ra_m|ra_s", "dec|dec_d|dec_m|dec_s", "stokes_i|stokes_q|stokes_u|stokes_v"]
)
gaussian_source = SourceType("Gaussian Source", inherit=point_source, required=["emaj|emin"], surplus=["pa"])
continuum_source = SourceType(
    "Continuum Source",
    inherit=point_source,
    required=[],
    surplus=["cont_ref_freq", "cont_coeff_1", "cont_coeff_2", "cont_coeff_3"],
)
line_source = SourceType(
    "Spectral Line Source",
    inherit=point_source,
    required=["line_peak", "line_ref_freq|line_redshift"],
    surplus=["line_width"],
)
polarised_source = SourceType("Polarised Source", inherit=point_source, required=["stokes_q|stokes_u|stokes_v"])
exoplanet_transient_source = SourceType(
    "Exoplanet Transient Source",
    inherit=point_source,
    required=[
        "transient_start",
        "transient_period",
        "transient_ingress",
        "transient_absorb",
    ],
)


@dataclass
class StokesData:
    """
    Container for source/image Stokes intensity data.

    Parameters
    ----------
    data : list of int or numpy.ndarray
        Stokes parameter data ordered as I, Q, U, V (or IVQU when not linear).
    linear_basis : bool, optional
        If True, the Stokes parameters are interpreted in a linear basis
        (I, Q, U, V). If False, they are interpreted in a circular basis
        (I, V, Q, U). Default is True.
    """

    data: List[int] | np.ndarray
    linear_basis: Optional[bool] = True

    def __post_init__(self):
        self.data = np.array(self.data) if not isinstance(self.data, np.ndarray) else self.data
        if self.linear_basis:
            self.param_string = "IQUV"
        else:
            self.param_string = "IVQU"

    def set_spectrum(self, freqs: np.ndarray, specfunc: Callable, full_pol: bool = True, **kwargs):
        """
        Add a spectral axis to the Stokes data.

        Parameters
        ----------
        freqs : numpy.ndarray
            Frequency array.
        specfunc : Callable
            Callable with signature f(freqs, flux, **kwargs) returning the
            spectral profile.
        full_pol : bool, optional
            If True, compute spectra for all four Stokes parameters.
            If False, only Stokes I is used. Default is True.
        **kwargs
            Additional keyword arguments forwarded to `specfunc`.

        Returns
        -------
        None
        """
        if isinstance(freqs, list):
            freqs = np.array(freqs)
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
        Add a time axis using a lightcurve.

        Parameters
        ----------
        lightcurve_func : Callable
            Callable returning a 1D time-series array (shape (ntimes,)).
        **kwargs
            Additional keyword arguments forwarded to `lightcurve_func`.

        Returns
        -------
        None
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
        """
        Get intensity data for a Stokes parameter.

        Parameters
        ----------
        x : {'I', 'Q', 'U', 'V'}
            Stokes parameter to extract.

        Returns
        -------
        numpy.ndarray or int
            Data array for the requested Stokes parameter, or 0 if not present.

        Raises
        ------
        RuntimeError
            If an unknown Stokes parameter is requested.
        """

        if x not in self.param_string:
            raise RuntimeError(f"Uknown Stokes paramter '{x}'")

        try:
            xdata = self.data[self.param_string.index(x)]
        except IndexError:
            xdata = 0

        return xdata

    def get_brightness_matrix(self, ncorr: int) -> np.ndarray:
        """
        Build the brightness (coherency) matrix.

        Parameters
        ----------
        ncorr : int
            Number of output correlations (2 or 4).

        Returns
        -------
        numpy.ndarray
            Brightness matrix with the same shape as `self.data` except that
            the Stokes axis is replaced by a correlation axis of length `ncorr`.
            For ncorr == 2 the dtype is float; for ncorr == 4 the dtype is complex.
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
            if self.linear_basis:
                bmatrix[tslice(0)] = self.I + self.Q
                bmatrix[tslice(1)] = self.I - self.Q
            else:
                bmatrix[tslice(0)] = self.I + self.V
                bmatrix[tslice(1)] = self.I - self.V
        else:
            if self.linear_basis:
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


class StokesDataFits(StokesData):
    def __init__(self, coord: xr.DataArray, dim_idx: int, data: np.ndarray, linear_basis: bool = True):
        """
        Wrap FITS Stokes data with axis mapping.

        Parameters
        ----------
        coord : xarray.DataArray
            Coordinate array describing the Stokes axis.
        dim_idx : int
            Index of the Stokes dimension in `data`.
        data : numpy.ndarray
            FITS image/cube data containing Stokes planes.
        linear_basis : bool, optional
            If True, interpret data in the linear basis. Default is True.
        """
        self.data = data
        self.nstokes = coord.size
        self.idx = dim_idx
        ndim = len(data.shape)
        self.__dslice__ = [slice(None)] * ndim
        self.linear_basis = linear_basis

    def __stokes_x__(self, x: str):
        """
        Get intensity data for a Stokes parameter.

        Parameters
        ----------
        x : {'I', 'Q', 'U', 'V'}
            Stokes parameter to extract.

        Returns
        -------
        numpy.ndarray or int
            Data array for the requested Stokes parameter, or 0 if not present.

        Raises
        ------
        RuntimeError
            If an unknown Stokes parameter is requested.
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


def contspec(freqs: np.ndarray, flux: float, coeff: float | np.ndarray | List, nu_ref: float):
    """
    Continuum (power-law) spectral profile.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frequency array.
    flux : float
        Reference flux density at `nu_ref`.
    coeff : float or array-like
        Power-law coefficient(s). If array-like with length > 1, treated as
        polynomial coefficients of log-log curvature (via numpy.polynomial).
        If length == 1 or float, treated as spectral index.
    nu_ref : float
        Reference frequency.

    Returns
    -------
    numpy.ndarray
        Spectral profile evaluated at `freqs`.
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


def gauss_1d(xaxis: np.ndarray, peak: float, width: float, x0: float):
    """
    Single Gaussian spectral line profile.

    Parameters
    ----------
    xaxis : numpy.ndarray
        Grid along the spectral axis.
    peak : float
        Gaussian peak amplitude.
    width : float
        Full width at half maximum (FWHM) in the same units as `xaxis`.
    x0 : float
        Centre position of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Profile evaluated at `xaxis`.
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
    Logistic transit lightcurve for an exoplanet-like transient.

    Parameters
    ----------
    start_time : int
        Start time of the observation (seconds).
    end_time : int
        End time of the observation (seconds).
    ntimes : int
        Number of time samples between start and end.
    transient_start : int
        Time at which the transit begins (seconds).
    transient_absorb : float
        Maximum fractional flux decrease during transit (e.g., 0.01 = 1% dip).
    transient_ingress : int
        Duration of ingress/egress (seconds).
    transient_period : int
        Total duration of the transit event, including ingress and egress (seconds).

    Returns
    -------
    numpy.ndarray
        Normalized intensity time series of shape (ntimes,).
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
