
from dataclasses import dataclass
from typing import Any, List, Optional, Union
from enum import Enum
import yaml
from  utilities import ValidationError, ListSpec, BASE_TYPES
from copy import deepcopy

class Parameter(object):
    def __init__(self, dtype, info, default=None, required=False):
        self.dtype = dtype
        self.info = info
        # ensure that the default is not overwritten if value is updated
        self.default = deepcopy(default)
        self.value = default
        self.required = required
        self.islist = False

        if isinstance(self.dtype, str):
            if self.dtype.startswith("List["):
                self.dtype = ListSpec(self.dtype)
                self.islist = True
            elif self.dtype in BASE_TYPES.keys():
                self.dtype = getattr(BASE_TYPES, self.dtype)
            else:
                raise ValidationError(f"Type {self.dtype} is not supported.")

        if self.value is not None:
            self.validate_value()

        if not isinstance(required, bool):
            raise ValidationError("The required option has to be a boolean")
    
    def validate_value(self, value=None):
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
        value = value or self.value
        if self.islist:
            if not isinstance(value, list):
                #TODO(sphe) add a warning here
                value = [value]
            self.dtype.set_dtype()
            return all( isinstance(item, self.dtype.dtype) for item in value )
        else:
            return isinstance(value, self.dtype)
    
    def update_value(self, value):
        if self.validate_value(value):
            self.value = value
        else:
            raise ValidationError( f"Parameter has wrong type. "
                                  f"Expected type: {self.dtype}, "
                                  f"Actual type: {type(value)}, "
                                  f"Actual value: {value}")
    
def validate(valme, schemafile=None):

    
    valme.set_schema(schemafile)
    
    section = valme.schema[valme.schema_section]
    for key in section:
        value = getattr(valme, key, None)
        if isinstance(section[key], str):
            continue 
        print(f"Processing key: {key}, value: {value}")
        param = Parameter(**section[key])
        if value is None and param.required:
            raise ValidationError(f"Required parameter '{key}' has not been set")
        param.update_value(value)
