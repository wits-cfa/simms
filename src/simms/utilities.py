from collections.abc import Callable
from itertools import combinations
from types import NoneType

import numpy as np
import yaml
from astropy import units
from numba import njit

from simms.exceptions import SkymodelSchemaError


class AttrDict(dict):
    """A ``dict`` whose keys are also readable and writable as attributes.

    Replaces the one thing simms actually used ``omegaconf`` for: config loaded from YAML
    that later reads as ``cfg.antnames`` as well as ``cfg["antnames"]``. Nothing here needed
    interpolation, struct mode, or merging, so a dict subclass covers it -- and being a real
    ``dict`` means ``.get()``, ``in``, iteration and ``**`` all behave without special cases.

    One deliberate difference from ``DictConfig``: a missing key raises ``AttributeError``
    rather than returning ``None``, so ``hasattr`` answers the question it appears to ask.
    """

    def __init__(self, mapping=(), **kwargs):
        super().__init__(mapping, **kwargs)
        for key, value in self.items():
            super().__setitem__(key, _wrap(value))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        self[name] = _wrap(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value):
    """Recursively present nested mappings as :class:`AttrDict`, so ``a.b.c`` works."""
    if isinstance(value, AttrDict):
        return value
    if isinstance(value, dict):
        return AttrDict(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_yaml(path) -> AttrDict:
    """Load a YAML file as an :class:`AttrDict`."""
    with open(path) as fh:
        return _wrap(yaml.safe_load(fh) or {})


class ObjDict:
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


def is_numeric(string):
    """
    Checks if a string can be converted to a float.
    """
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


@njit
def radec2lm(ra0: float, dec0: float, ra: float | np.ndarray, dec: float | np.ndarray):
    """
    Convert (RA, Dec) to direction cosine coordinates (l,m)

    Args:
        ra0 (float|np.ndarray): phase centre RA in radians.
        dec0 (float): phase centre Dec in radians.
        ra (float or np.ndarray): RA in radians.
        dec (float or np.ndarray): Dec in radians.
    Returns:
        a tuple of l and m
    """
    dra = ra - ra0
    l_coord = np.cos(dec) * np.sin(dra)
    m_coord = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)

    return l_coord, m_coord


def get_noise(sefds: list | float, dtime: int, dfreq: float):
    """
    This function computes the noise given an SEFD/s.
    """

    if isinstance(sefds, int | float):
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


def quantity_to_value(
    coord: Callable | NoneType,
    value: str | int | float,
    val_units: str | None = None,
    target_units: str | None = None,
    null_value=None,
) -> int | float:
    """
    Converts a value (string or numeric) with units to a float or int in the target units.

    Args:
        coord (Callable|NoneType): Function to convert the given
        value (str|numeric): The value to convert.
        val_units (str): The units of the value.
        target_units (str): The units to convert to.
    Raises:
        SkymodelSchemaError: If the units are unknown or invalid.

    Returns:
        float|int: The converted value in the target units.
    """
    if value in [None, "null"]:
        return null_value

    if isinstance(value, float | int):
        if val_units:
            try:
                quant_value = value * getattr(units, val_units)
            except AttributeError as exc:
                raise SkymodelSchemaError(f"Unknown parameter units '{val_units}'") from exc
        else:
            quant_value = units.Quantity(value)
    elif is_numeric(value):
        quant_value = coord(f"{value} {val_units}") if val_units else coord(value)
    else:
        quant_value = coord(value)

    if isinstance(quant_value, str):
        return quant_value

    if target_units:
        return quant_value.to(target_units).to_value()
    else:
        return quant_value.to_value()
