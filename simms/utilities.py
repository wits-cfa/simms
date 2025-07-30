import numpy as np
from itertools import combinations
from typing import Union, List

class ValidationError(Exception):
    pass


class CatalogueError(Exception):
    pass


class FITSSkymodelError(Exception):
    pass


class ParameterError(Exception):
    pass


def isnummber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
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

def get_noise(sefds: Union[List, float], dtime: int, dfreq: float):
    """
    This function computes the noise given an SEFD/s.
    """

    if isinstance(sefds, (int, float)):
        noise = sefds / np.sqrt(2 * dfreq * dtime)
        return noise

    sefd_pairs = list(combinations(sefds, 2))
    noises = []
    for sefd1, sefd2 in sefd_pairs:
        prod = sefd1 * sefd2
        den = 2 * dfreq * dtime
        noise = np.sqrt(prod / den)
        noises.append(noise)

    return noises
