import os
import re
from typing import Any, List, Optional

import yaml


class ValidationError(Exception):
    pass


class File(str):
    def __init__(self, path, check=True):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.exists = os.path.exists(self.path)
        self.dirname = os.path.dirname(self.path)

        if check:
            if not self.exists:
                raise FileNotFoundError(f"File {self.path} does not exist.")
            self.isfile = os.path.isfile(self.path)


class Directory(File):
    @property
    def isdir(self):
        if self.check and not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"File {self.path} is not a directory. (does it exist?)")
        else:
            return True


# nicked from https://www.hackertouch.com/how-to-get-a-list-of-class-attributes-in-python.html
def get_class_attributes(cls):
    return [item for item in cls.__dict__ if not callable(getattr(cls, item)) and not item.startswith('__')]


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
    'float': (float, int),
    'bool': bool,
    'str': str,
})


class ListSpec (object):
    def __init__(self, listspec: str) -> None:
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
            raise TypeError(
                "The list specification, listspec, has to be a string.")

        self.listspec = listspec

    def __call__(self):

        _thematch = re.findall(
            "^List*\[([a-zA-Z{1,}].*\w{1,})\]", self.listspec)
        if len(_thematch) != 1:
            raise ValidationError(f"List specification '{self.listspec}' is invalid."
                                  f" It should be 'List[type]', e.g, 'List[float]'.")

        thematch = _thematch[0]

        if thematch in BASE_TYPES.keys():
            self.dtype = getattr(BASE_TYPES, thematch)

        else:
            raise ValidationError(
                f"Type {thematch} is not supported. Verify the type of your list elements.")

        return self.dtype


CLASS_TYPES = ObjDict({
    'List': ListSpec,
    'File': File,
    'Directory': Directory,

})


def readyaml(yamlfile: str) -> dict:
    with open(yamlfile) as stdr:
        return yaml.load(stdr, Loader=yaml.FullLoader)


class CatalogueError(Exception):
    pass

class ParameterError(Exception):
    pass