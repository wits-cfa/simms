import os.path
from dataclasses import dataclass
from typing import Dict, List, Optional

import astropy.units as aunits
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
from simms.utilities import ObjDict, quantity_to_value

DEFAULT_SOURCE_SCHEMA = os.path.join(SCHEMADIR, "skymodel_schema.yaml")

PTYPE_MAPPER = {
    # converter, default units, null_value
    "longitude": (Longitude, "rad", None),
    "latitude": (Latitude, "rad", None),
    "angle" : (Angle, "rad", 0),
    "frequency":  (SpectralCoord, "Hz", None),
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

    def set_value(self, value:str|float|int):
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
        self.value = quantity_to_value(ptype_coord, value, self.units, target_units=target_units,
                                    null_value=null_value)


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

    def set_source_param(self, field:str, value:str|float|int):
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
        return list(filter(
            lambda item: getattr(self.schema.parameters[item], "required", False),
            self.schema.parameters,
        ))
    
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
        self.is_point = {"emaj", "emin"}.isdisjoint(self.existing_fields)

        self.is_polarised =  len(set([f"stokes_{i}" for i in "iquv"]).intersection(self.existing_fields)) > 1

        req_trans_flds = {"transient_start", "transient_period", "transient_ingress", "transient_absorb"}
        # this retturns the required fields that are not in existing_field
        missing_trans_flds = req_trans_flds - set(self.existing_fields)
        
        # if missing fields are the same as the required, then there's no intersection
        # and no required transient fields are set. This is not a transient source
        if missing_trans_flds == req_trans_flds:
            self.is_transient = False
        # if missing fields are not empty at this point, some required fields are set but not all
        # transient source are all-or-nothing here, so raise an error
        elif missing_trans_flds:
            raise ASCIISourceError(
                "Transient source specification is missing required parameter(s):"f" {', '.join(missing_trans_flds)}"
                )
        # if we get to this point, then the required fields are a subset of existing_fields
        # this is a transient
        else:
            self.is_transient = True


@dataclass
class ASCIISkyModel:
    skymodel_file: str|File
    delimiter:str = None
    source_schema_file: str|File = File(DEFAULT_SOURCE_SCHEMA)
    sources: List[ASCIISource] = None

    def __post_init__(self):
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

            missing_required  = set(required_fields) - set(header)
            if missing_required:
                raise ASCIISkymodelError("ASCII Sky model file header is missig required field(s): "
                                         f"{', '.join(missing_required)}")

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
                        f"The number of columns in row {counter+2} rows does not match the number of"
                        " the number of columns in the header"
                        )
                
                source = ASCIISource(self.schema)
                for param, value in zip(header, rowdata):
                    source.set_source_param(alias_to_field[param], value)
                # source has been fully set, finalise and add it to rest of the sources
                source.finalise()
                sources.append(source)
        
        self.sources = sources
