from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import yaml


SCHEMA = "./schema_freq.yaml"

TYPES = [int, float, bool, str, list, Union]

TYPE_MAP = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    'List': list,
    'Union': Union}


class ValidationError(Exception):
    pass


class Parameter(object):
    def __init__(self, dtype, info, default=(), required=False):
        self.dtype = TYPE_MAP.get(dtype, dtype)
        self.info = TYPE_MAP.get(info, info)
        self.default = default
        self.required = TYPE_MAP.get(required, required)

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
            raise ValidationError( f"Parameter has wrong type. Expected type: {self.dtype.__name__}, "f"Actual type: {type(value).__name__}, Actual value: {value}")
    
def validate(valme, schemafile=SCHEMA):
    with open(schemafile) as stdr:
        schema = yaml.load(stdr, Loader=yaml.FullLoader)

    section = schema[valme.__class__.__name__]
    print(section)
    for key in section:
        if isinstance(section[key], str):
            continue 
        param = Parameter(**section[key])
        value = getattr(valme, key, None)
        print(f"Processing key: {key}, value: {value}")
        print (param, value)
        if value is None and param.required:
            raise ValidationError(f"Required parameter '{key}' has not been set")
        param.setme(value)

@dataclass
class Line(object):
    """
    blah blah
    """
    freq_peak: float
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


line = Line(1.3, 4, 10.2)
validate(line)
cont=Cont(5.2, 10.6, [5.1,2.2,3.4,5])
validate(cont)