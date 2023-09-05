from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import yaml


SCHEMA = "./schema_freq.yaml"

TYPES = [int, float, bool, str, List, Union]

class ValidationError(Exception):
    pass


class Parameter(object):
    def __init__(self, dtype, info, default=(), required=False):
        self.dtype = dtype
        self.info = info
        self.default = default
        self.required: required

        if self.dtype not in TYPES:
            raise ValidationError(f"Type {dtype} {dtype.__class__.__name__} is not supported.")
    
        if default and not isinstance(default, dtype):
            raise ValidationError(f"Default value does not match dtype ({dtype.__class__.__name__})")

        if not isinstance(required, bool):
            raise ValidationError("The required option has to be a boolean")
    
    def setme(self, value):
        if isinstance(value, self.dtype):
            self.value = value
        else:
            raise ValidationError("Parameter has wrong type")

    
def validate(valme, schemafile=SCHEMA):
    with open(schemafile) as stdr:
        schema = yaml.load(stdr, Loader=yaml.FullLoader)

    section = schema[valme.__class__.__name__]
    for key in section:
        if isinstance(section[key], str):
            continue 
        param = Parameter(**section[key])
        value = getattr(valme, key, None)
        if value is None and param.required:
            raise ValidationError(f"Required parameter '{key}' has not been set")
        param.setme(value)

@dataclass
class Line(object):
    """
    blah blah
    """
    peak_freq: float
    width: int
    stokes: float

@dataclass
class Cont(object):
    """
    blah blah
    """
    ref_freq: float
    stokes: float
    coeffs: List[float]


line = Line(1.4, 4, 10)
validate(line)