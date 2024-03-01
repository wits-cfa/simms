<<<<<<< HEAD
from  utilities import ValidationError, ListSpec
from utilities import BASE_TYPES, readyaml, get_class_attributes
=======
import os
from dataclasses import dataclass
from typing import List, Union
from  simms.utilities import ValidationError, ListSpec
from simms.utilities import BASE_TYPES, CLASS_TYPES
from simms.utilities import readyaml, get_class_attributes
>>>>>>> c4aac864c86e7fddf8728c89c3bd08f1bec94b28
from copy import deepcopy
from simms import LOG


class DType():
    def __init__(self, stringspec: List|str):
        """
        Convetrs string type specifications to Python objects

        Parameters
        ------------
    
        stringspec: List|str
            string type speccification, or a list of string specifications
        """

        self.stringspec = stringspec
        if not self.islist:
            self.stringspec = [self.stringspec]
        
        self.ntypes = len(self.stringspec)

        self.dtype = None 
        self.classparam = [False]*self.ntypes
        self.listparam = [False]*self.ntypes

        
    @property
    def islist(self):
        """
        Check if dtype is a list
        """
        return isinstance(self.stringspec, list)
    
    def __call__(self):
        """
        Return the types
        """
        types = [None]*self.ntypes
        for i, item in enumerate(self.stringspec):
            if item in BASE_TYPES:
                types[i] = BASE_TYPES[item]
            elif item in CLASS_TYPES:
                self.classparam[i] = True
                if item.startswith('List['):
                    types[i] = CLASS_TYPES['List']
                    self.listparam[i] = True
                else:
                    types[i] = CLASS_TYPES[item]
            else:
                raise ValidationError(f"Type {item} is not supported.")
        self.dtype = Union[tuple(types)] 

class Parameter(object):
    def __init__(self, key, dtype, info, default=None, required=False):
        self.key = key 
        self.dtype = DType(dtype)
        self.info = info
        # ensure that the default is not overwritten if value is updated
        self.default = deepcopy(default)
        self.value = self.default
        self.required = required


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

        def _validate(dtype, val, isclass, islist):
            if isclass and not islist:
                try:
                    dtype(val)
                except:
                    return False
            elif islist:
                return all( isinstance(item, dtype) for item in val )

        
        for i in range(self.dtype.ntypes):
            dtype = self.dtype.dtypes[i]
            classparam = self.dtype.classparam[i]
            listparam = self.dtype.listparam[i]

            if listparam:
                if not isinstance(value, list):
                    LOG.warning(f"List parameter, {self.key}, given as a single value."
                            f" Setting it to a single-valued List")
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
    valme.validate_section(valme.schema_section)

    for key in section:
        value = getattr(valme, key, None)
        # valid parameters can only be dicts, strings are either meta or infomation data
        # which does need to be validated

        if isinstance(section[key], str):
            continue 

        param = Parameter(key, **section[key])
        if value is None:
            if param.required:
                raise ValidationError(f"Required parameter '{key}' has not been set")
        else:
            param.update_value(value)

        LOG.debug(f"Setting key: {param.key}, value: {param.value}")

class SpecBase(object):
    def __init__(self, schemafile=None):
        self.schemafile = schemafile
    
    def set_schema(self, schemafile=None):
        schemafile = schemafile or self.schemafile

        if os.path.exists(schemafile):
            self.schema = readyaml(schemafile)
        else:
            raise FileNotFoundError(f"Schema file '{schemafile}' could not be found")

    def validate_section(self, section=None):
        section = section or self.schema_section
        class_set = set(get_class_attributes(self))
        section_set = set(self.schema[section].keys())

        # ignore these parameters
        novalidate = set(["schemafile", "schema_section", "schema"])
        class_set_valid = class_set.difference(novalidate)
        # check for schema/class mismatches with the rest of the parameters
        mismatch = class_set_valid.difference(section_set)
        if mismatch:
            raise ValidationError(f"Schema file, {self.schemafile}"
                                  f", does not match class definition"
                                  f" for section: {section}."
                                  f"Mismatched parameters are: {mismatch}")

        return True
