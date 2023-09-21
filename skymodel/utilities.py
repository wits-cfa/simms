from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import re


SCHEMA = "./schema_freq.yaml"

class ValidationError(Exception):
    pass

class ObjDict(object):
    def __init__(self, items):
        """
        Converts a dictionary into an object. 

        """
        # First give this objects all the attributes of the input dicttionary
        for item in dir(dict):
            if not item.startswith("__"):
                setattr(self, item, getattr(items, item, None))
        # Now set the dictionary values as attributes
        self.__dict__.update(items)


BASE_TYPES = ObjDict({
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    })


class ListSpec (object):
    def __init__(self, listspec:str, values:Optional[List] = [] )-> None:
        """
        Defines how to interpret list parameters defined the 'List[<type>]' syntax

        Parameters
        -------------
        listspec: str
            List specification. For example, 'List[float]'
        values: values
            The actaul list. Optional.
        
            
        Returns
        --------
        
        """
        if not isinstance(listspec, str):
            raise TypeError("The list specification, listspec, has to be a string.")

        self.listspec = listspec
        self.values = values

    def set_dtype(self, dtype=None):
        if dtype:
            self.dtype = dtype
            return
        
        _thematch = re.findall("^List*\[([a-zA-Z{1,}].*\w{1,})\]", self.listspec)
        if len(_thematch) != 1:
            raise ValidationError(f"List specification '{self.listspec}' is invalid."
                                  f" It should be 'List[type]', e.g, 'List[float]'.")
        
        thematch = _thematch[0]

        if thematch in BASE_TYPES.keys():
            self.dtype = getattr(BASE_TYPES, thematch)

        else:
            raise ValidationError(f"Type {thematch} is not supported. Verify the type of your list elements.")
        
        return self.dtype
    
