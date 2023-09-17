
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import yaml
#import schema
#from validator import validate


SCHEMA = "./schema_freq.yaml"

TYPES = [int, float, bool, str, list, Union]

TYPE_MAP = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    'List[float]': list,
    'Union': Union
    }


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
        if self.dtype == List[float]:
            self.dtype = List[float]
        #if isinstance(value, list) and all(isinstance(item, float) for item in value):
        if isinstance(value, self.dtype):
            self.value = value
        else:
            raise ValidationError( f"Parameter has wrong type. Expected type: {self.dtype.__name__}, "f"Actual type: {type(value).__name__}, Actual value: {value}")

schema = {
    'Line': {
        'freq_peak': {'dtype': 'float', 'info': 'float', 'default': None, 'required': True},
        'width': {'dtype': 'int', 'info': 'int', 'default': None, 'required': True},
        'stokes': {'dtype': 'float', 'info': 'float', 'default': None, 'required': True}
    },
    'Cont': {
        'ref_freq': {'dtype': 'float', 'info': 'float', 'default': None, 'required': True},
        'stokes': {'dtype': 'float', 'info': 'float', 'default': None, 'required': True},
        'coeffs': {'dtype': 'List[float]', 'info': 'List[float]', 'default': None, 'required': True}
    }
}

@dataclass
class Line(object):
    freq_peak: float
    width: int
    stokes: float

@dataclass
class Cont(object):
    ref_freq: float
    stokes: float
    coeffs: List[float]

def validate(valme, schema):
    class_name = valme.__class__.__name__
    section = schema.get(class_name)

    if section is None:
        raise ValidationError(f"No schema found for class '{class_name}'")

    for key, value in section.items():
        if isinstance(value, str):
            continue
        param = Parameter(**value)
        attr_value = getattr(valme, key, None)
        print(f"Processing key: {key}, value: {attr_value}")
        param.setme(attr_value)

line = Line(1.3, 4, 10.2)
validate(line, schema)

cont = Cont(5.2, 10.6, ['hey'])
validate(cont, schema)


