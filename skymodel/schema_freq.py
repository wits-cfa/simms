from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import yaml
from  utilities import ValidationError, MyList

SCHEMA = "./schema_freq.yaml"

TYPE_MAP = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    "List": MyList,
    #'Union': Union,
    }

def type_map(dtype):
    """
    Ensures that the data type is a data type and not a string.
    """
    if isinstance (dtype, str):
        if dtype.startswith('List['):
            return MyList
        else:
            return TYPE_MAP.get(dtype, None)
    else:
        return dtype
    


class Parameter(object):
    def __init__(self, dtype, info, default=(), required=False):
        self.dtype = type_map(dtype)
        self.info = info
        self.default = default
        self.required = required


        
        if self.dtype not in TYPE_MAP.values():
            raise ValidationError(f"Type {self.dtype} is not supported.")


        if self.default and not self.__isinstance(self.default, self.dtype):
            raise ValidationError(f"Default value does not match dtype ({self.dtype.__class__.__name__})")

        if not isinstance(self.required, bool):
            raise ValidationError("The required option has to be a boolean")
    
    def __isinstance(self, value, dtype):
        """
        Extends the isinstance() function to handle our MyList types

        Parameters
        ---------
        value: Any
            The value being checked
        dtype:
            The type that value has to be

        
        Returns
        --------------
        Boolean value indicating whether value is of type dtype.
        """
        if dtype is MyList:
            """
            Need to figure out how to validate list of stuff
            """
        else:
            return isinstance(value, dtype)
    
    def setme(self, value):
        if isinstance(value, self.dtype):
            self.value = value
        else:
            raise ValidationError( f"Parameter has wrong type. "
                                  f"Expected type: {self.dtype.__name__}, "
                                  f"Actual type: {type(value).__name__}, "
                                  f"Actual value: {value}")
    
def validate(valme, schemafile=SCHEMA):
    with open(schemafile) as stdr:
        schema = yaml.load(stdr, Loader=yaml.FullLoader)

    section = schema[valme.__class__.__name__]
    for key in section:
        if isinstance(section[key], str):
            continue 
        param = Parameter(**section[key])
        value = getattr(valme, key, None)
        print(f"Processing key: {key}, value: {value}")
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