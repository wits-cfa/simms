import os.path
from dataclasses import dataclass
from typing import Dict, List, Optional

import astropy.units as aunits
import numpy as np
from astropy.coordinates import (
    Angle,
    Latitude,
    Longitude,
    SpectralCoord,
)
from omegaconf import OmegaConf
from scabha.basetypes import File
from scabha.cargo import Parameter

from simms import SCHEMADIR
from simms.exceptions import (
    ASCIISkymodelError,
    ASCIISourceError,
)
from simms.skymodel.source_factory import (
    StokesData,
    continuum_source,
    contspec,
    exoplanet_transient_logistic,
    exoplanet_transient_source,
    gauss_1d,
    gaussian_source,
    line_source,
    point_source,
    polarised_source,
)
from simms.utilities import ObjDict, quantity_to_value

DEFAULT_SOURCE_SCHEMA = os.path.join(SCHEMADIR, "source_schema.yaml")

PTYPE_MAPPER = {
    # parameter type: (converter, default_units, null_value)
    "longitude": (Longitude, "rad", None),
    "latitude": (Latitude, "rad", None),
    "angle": (Angle, "rad", 0),
    "frequency": (SpectralCoord, "Hz", None),
    "number": (aunits.Quantity, None, None),
    "flux": (aunits.Quantity, "Jy", 0),
    "time": (aunits.Quantity, "s", None),
    "string": (str, None, ""),
}


@dataclass
class SkymodelParameter(Parameter):
    info: Optional[str] = None
    units: Optional[str] = None
    alias: Optional[str] = None
    ptype: Optional[str] = "number"
    frame: Optional[str] = None
    required: Optional[bool] = False
    join: Optional[List[str]] = None

    def set_value(self, value: str | float | int):
        """
        Set and convert the parameter value to the appropriate type and units.

        Parameters
        ----------
        value : str or float or int
            Value to set. Can be a plain number or a string with units.

        Returns
        -------
        None
            The converted value is stored in ``self.value``.

        Raises
        ------
        KeyError
            If the parameter type is unknown.
        ValueError
            If the value cannot be parsed or converted to the target units.
        """
        ptype_coord, target_units, null_value = PTYPE_MAPPER[self.ptype]
        self.value = quantity_to_value(ptype_coord, value, self.units, target_units=target_units, null_value=null_value)


@dataclass
class ASCIISourceSchema:
    info: str
    parameters: Dict[str, SkymodelParameter]


@dataclass
class ASCIISource:
    schema: ASCIISourceSchema

    def __post_init__(self):
        parameters = {}

        # use this struct to set dataclass defaults defaults
        param_struct = OmegaConf.structured(SkymodelParameter)
        for key, val in self.schema.parameters.items():
            _val = OmegaConf.merge(param_struct, val)
            parameters[key] = SkymodelParameter(**_val)
        self.parameters = ObjDict(parameters)

        self.existing_fields = []
        self.is_finalised = False

    def set_source_param(self, field: str, value: str | float | int):
        """
        Set a parameter value for the source.

        Parameters
        ----------
        field : str
            Name of the source parameter (e.g., 'ra', 'dec', 'flux', 'name').
        value : str or float or int
            Value to assign. Converted to the appropriate type and units.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ``field`` is not a valid parameter in the schema.
        ValueError
            If the value cannot be converted to the required type/units.

        Notes
        -----
        This method updates the instance in place.
        """
        param = getattr(self.parameters, field)
        param.set_value(value)
        setattr(self, field, param.value)
        self.existing_fields.append(field)

    def alias_to_field_mapper(self):
        mapper = OmegaConf.create({})
        for key, val in self.schema.parameters.items():
            mapper[getattr(val, "alias", None) or key] = key
        return mapper

    def field_to_alias_mapper(self):
        mapper = OmegaConf.create({})
        for key, val in self.schema.parameters.items():
            mapper[key] = getattr(val, "alias", None) or key
        return mapper

    def finalise(self):
        """
        Validate set fields and derive source type flags.

        Notes
        -----
        - Ensures the source satisfies the minimal point-source requirements.
        - Sets boolean flags: ``is_point``, ``is_polarised``, ``is_line``,
          ``is_continuum``, ``is_transient``, and ``is_exoplanet_transient``.
        - Computes and sets fields that are defined as a join of other fields.

        Raises
        ------
        ASCIISourceError
            If the source does not satisfy minimal point-source requirements.
        """
        fields = self.existing_fields
        # If the source is not a valid point source, it's not a valid source
        point_source.is_valid(fields, raise_exception=True)

        self.is_point = not gaussian_source.is_valid(fields)

        self.is_polarised = polarised_source.is_valid(fields)

        self.is_line = line_source.is_valid(fields)

        self.is_continuum = continuum_source.is_valid(fields)

        self.is_transient = self.is_exoplanet_transient = exoplanet_transient_source.is_valid(fields, none_or_all=True)

        for field, param in self.parameters.items():
            if getattr(param, "join", []) and field not in fields:
                join_us = param.join
                joined = np.sum([getattr(self, key, 0) for key in join_us])
                setattr(self, field, joined)

    def value_or_default(self, field: str) -> int | float | str:
        """
        Return the value of a field or its default.

        Parameters
        ----------
        field : str
            Source field name (e.g., 'ra', 'dec', 'stokes_i').

        Returns
        -------
        int or float or str
            The value set from the ASCII file, or the schema default if unset.
        """
        val = getattr(self, field, None)
        if val is None:
            param = getattr(self.parameters, field)
            param.set_value(None)
            val = param.value
        return val

    def get_brightness_matrix(
        self,
        chan_freqs: np.ndarray,
        ncorr: int,
        unique_times: np.ndarray = None,
        time_index_mapper: np.ndarray = None,
        full_stokes: bool = True,
        linear_basis: bool = True,
    ) -> np.ndarray:
        """
        Generate the brightness matrix across frequency (and time if transient).

        Parameters
        ----------
        chan_freqs : numpy.ndarray
            1D array of channel frequencies in Hz.
        ncorr : int
            Number of correlations (e.g., 4 for full Stokes).
        unique_times : numpy.ndarray, optional
            1D array of unique time samples (seconds) for transient sources.
        time_index_mapper : numpy.ndarray, optional
            Mapping indices from unique time samples to the output time axis.
        full_stokes : bool, default True
            If True, compute all Stokes parameters; otherwise Stokes I only.
        linear_basis : bool, default True
            If True, use linear polarization basis (XX, XY, YX, YY). If False, use circular basis (RR, RL, LR, LL).

        Returns
        -------
        numpy.ndarray
            Brightness matrix with shape ``(nfreq, ntime, ncorr)`` for transient
            sources or ``(nfreq, ncorr)`` for non-transient sources.

        Raises
        ------
        ValueError
            If required spectral or transient fields are missing or invalid.

        Notes
        -----
        - Spectrum is modeled as a Gaussian line or a polynomial continuum,
          depending on which fields are present.
        - For transient sources, a logistic lightcurve is applied.
        """
        self.stokes = StokesData([getattr(self, f"stokes_{x}", 0) for x in "iquv"], linear_basis=linear_basis)
        if self.is_line:
            specfunc = gauss_1d
            kwargs = {
                "x0": self.value_or_default("line_peak"),
                "width": self.value_or_default("line_width"),
            }
        elif self.is_continuum:
            specfunc = contspec
            kwargs = {
                "coeff": [
                    self.value_or_default("cont_coeff_1"),
                    self.value_or_default("cont_coeff_2"),
                    self.value_or_default("cont_coeff_3"),
                ],
                "nu_ref": self.value_or_default("cont_reffreq"),
            }

        self.stokes.set_spectrum(chan_freqs, specfunc, full_pol=full_stokes, **kwargs)
        if self.is_transient:
            lightcurve_func = exoplanet_transient_logistic
            t0 = unique_times.min()
            unique_times_rel = unique_times - t0
            kwargs = {
                "start_time": unique_times_rel.min(),
                "end_time": unique_times_rel.max(),
                "ntimes": unique_times_rel.shape[0],
                "transient_start": self.value_or_default("transient_start"),
                "transient_period": self.value_or_default("transient_period"),
                "transient_ingress": self.value_or_default("self.transient_ingress"),
                "transient_absorb": self.value_or_default("transient_absorb"),
            }
            self.stokes.set_lightcurve(lightcurve_func, **kwargs)
            return self.stokes.get_brightness_matrix(ncorr)[:, time_index_mapper, ...]
        else:
            return self.stokes.get_brightness_matrix(ncorr)


@dataclass
class ASCIISkymodel:
    """
    A sky model built from an ASCII column file.

    Parameters
    ----------
    skymodel_file : str or scabha.basetypes.File
        Path to the sky model file.
    delimiter : str, optional
        Column delimiter used in the file. If ``None``, split on whitespace.
    source_schema_file : str or scabha.basetypes.File, optional
        Path to the YAML source schema. Defaults to ``DEFAULT_SOURCE_SCHEMA``.
    sources : list of ASCIISource, optional
        Populated after parsing.

    Raises
    ------
    ASCIISkymodelError
        If the file cannot be read or validated.
    """

    skymodel_file: str | File
    delimiter: str = None
    source_schema_file: str | File = None
    sources: List[ASCIISource] = None

    def __post_init__(self):
        self.source_schema_file = self.source_schema_file or File(DEFAULT_SOURCE_SCHEMA)
        schema = OmegaConf.load(self.source_schema_file)
        self.schema = ASCIISourceSchema(**schema)
        sources = []

        # make a dummy source for some book keeping
        dummy_source = ASCIISource(self.schema)
        alias_to_field = dummy_source.alias_to_field_mapper()

        with open(self.skymodel_file) as stdr:
            line = stdr.readline().strip()

            if not line.startswith("#format:"):
                raise ASCIISkymodelError("ASCII sky model needs to have a header starting with the string #format:")

            header = line.strip().replace("#format:", "").strip().split(self.delimiter)

            try:
                point_source.is_valid(fields=[alias_to_field[key] for key in header], raise_exception=True)
            except ASCIISourceError as exc:
                raise ASCIISkymodelError(f"ASCII Sky model file header is missig required fields: {exc}")

            for counter, line in enumerate(stdr.readlines()):
                # skip lines that are commented
                if line.startswith("#"):
                    continue
                rowdata = line.strip().split(self.delimiter)
                # skip empty lines
                if len(rowdata) == 0:
                    continue
                elif len(rowdata) != len(header):
                    # plus 2 because header line is not counted (+1), and python starts counting at 0 (+1)
                    raise ASCIISkymodelError(
                        f"The number of columns in row {counter + 2} rows does not match the number of"
                        " the number of columns in the header"
                    )

                source = ASCIISource(self.schema)
                for param, value in zip(header, rowdata):
                    source.set_source_param(alias_to_field[param], value)
                # source has been fully set, finalise and add it to rest of the sources
                source.finalise()
                sources.append(source)

        self.sources = sources

    def _has_source_type(self, source_type: str):
        has_it = False
        for source in self.sources:
            if getattr(source, f"is_{source_type}", False):
                has_it = True
                break
        return has_it

    @property
    def has_transient(self):
        return self._has_source_type("transient")

    @property
    def has_exoplanet_transient(self):
        return self._has_source_type("exoplanet_transient")

    @property
    def has_line_source(self):
        return self._has_source_type("line")

    @property
    def has_continuum_source(self):
        return self._has_source_type("continuum")
