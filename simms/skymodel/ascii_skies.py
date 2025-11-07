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
    "flux": (aunits.Quantity, "Jy", None),
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

    def set_value(self, value: str | float | int):
        """Set and convert the parameter value to the appropriate type and units.

        This method takes a value in various formats (string, float, or int) and converts it
        to the appropriate coordinate type and target units based on the parameter type (ptype).
        The conversion is performed using the PTYPE_MAPPER configuration.

            value (str | float | int): The value to set. Can be a numeric value or a string
                representation of a quantity with units.

            None: The converted value is stored in self.value attribute.

        Note:
            The conversion process uses the ptype_coord, target_units, and null_value
            determined by the PTYPE_MAPPER for this parameter's ptype.
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
        """Set a parameter value for the current source.

        This method updates a specific field of the current source with the provided value.
        The field must be a valid source parameter name, and the value will be converted
        to the appropriate type if necessary.

        Args:
            field (str): The name of the source parameter field to set (e.g., 'ra', 'dec',
                         'flux', 'name'). Must correspond to a valid source attribute.
            value (str | float | int): The value to assign to the specified field.
                                       Type will be validated against the field requirements.

        Raises:
            CatalogueError: If the field name is invalid, the value type is incompatible
                            with the field, or if no source is currently active.

        Returns:
            None: This method modifies the source in place and does not return a value.
        """
        param = getattr(self.parameters, field)
        param.set_value(value)
        setattr(self, field, param.value)
        self.existing_fields.append(field)

    def required_fields(self):
        return list(
            filter(
                lambda item: getattr(self.schema.parameters[item], "required", False),
                self.schema.parameters,
            )
        )

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
        fields = self.existing_fields
        # If the source is not a valid point source, it's not a valid source
        point_source.is_valid(fields, raise_exception=True)

        self.is_point = not gaussian_source.is_valid(fields)

        self.is_polarised = polarised_source.is_valid(fields)

        self.is_line = line_source.is_valid(fields)

        self.is_continuum = continuum_source.is_valid(fields)

        self.is_transient = self.is_exoplanet_transient = exoplanet_transient_source.is_valid(fields, none_or_all=True)

    def value_or_default(self, field):
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
    ):
        """Populate self.stokes and polarisation flag from stokes_i/q/u/v.

        Builds a StokesData vector from stokes_i, stokes_q, stokes_u, and stokes_v
        (missing components default to 0). Sets self.is_polarised to True if more
        than one component is provided.

        Args:
            linear_basis: Whether to use the linear polarisation basis; forwarded to
                StokesData.

        """
        self.stokes = StokesData([getattr(self, f"stokes_{x}", 0) for x in "iquv"], linear_basis=linear_basis)
        if self.is_line:
            specfunc = gauss_1d
            kwargs = {
                "x0": self.line_peak,
                "width": self.line_width,
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
        field_to_alias = dummy_source.field_to_alias_mapper()
        alias_to_field = dummy_source.alias_to_field_mapper()

        required_fields = [field_to_alias[field] for field in dummy_source.required_fields()]

        with open(self.skymodel_file) as stdr:
            line = stdr.readline().strip()

            if not line.startswith("#format:"):
                raise ASCIISkymodelError("ASCII sky model needs to have a header starting with the string #format:")

            header = line.strip().replace("#format:", "").strip().split(self.delimiter)

            missing_required = set(required_fields) - set(header)
            if missing_required:
                raise ASCIISkymodelError(
                    f"ASCII Sky model file header is missig required field(s): {', '.join(missing_required)}"
                )

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
