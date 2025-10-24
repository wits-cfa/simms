from itertools import combinations
from typing import List, Union

import numpy as np


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
    return [item for item in cls.__dict__ if not callable(getattr(cls, item)) and not item.startswith("__")]


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

def is_range_in_range(inner_range, outer_range):
    """
    Checks if a given range (inner_range) is fully contained within 
    another range (outer_range).

    Assumes ranges are inclusive and represented as (start, end) tuples.
    
    Args:
        inner_range (tuple): A tuple (start, end) representing the inner range.
        outer_range (tuple): A tuple (start, end) representing the outer range.

    Returns:
        bool: True if inner_range is fully within outer_range, False otherwise.
    """
    
    # Unpack the tuples
    inner_start, inner_end = inner_range
    outer_start, outer_end = outer_range
    
    # In case the ranges are given "backwards" (e.g., (10, 5)), 
    # we normalize them to be (min, max).
    if inner_start > inner_end:
        inner_start, inner_end = inner_end, inner_start
    if outer_start > outer_end:
        outer_start, outer_end = outer_end, outer_start

    # The check:
    # 1. The inner range's start must be at or after the outer range's start.
    # 2. The inner range's end must be at or before the outer range's end.
    is_start_within = outer_start <= inner_start
    is_end_within = inner_end <= outer_end
    
    return is_start_within and is_end_within
