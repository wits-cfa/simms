from casacore.quanta import quantity as qa
from casacore.measures import measures
from numba import njit
import numpy as np
from typing import Union

def convert2rad(string):
    if string is not None and string != 'null':
        try:
            float(string)
            string += 'deg'
        except ValueError:
            pass
        finally:
            angle = qa(string)
            angle_rad = angle.canonical().get_value()
        return angle_rad
    
def convertra2rad(string):
    dm = measures()
    if string not in [None, "null"]:
        try:
            float(string)
            string += 'deg'
        except ValueError:
            pass
        finally:
            ra_rad = dm.direction(v0=f"{string}")["m0"]["value"]
        return ra_rad


def convertdec2rad(string):
    dm = measures()
    if string not in [None, "null"]:
        try:
            float(string)
            string += 'deg'
        except ValueError:
            pass
        finally:
            dec_rad = dm.direction(v1=f"{string}")["m1"]["value"]
        return dec_rad


def convert2Jy(string, null_value=None):
    if string not in [None, "null"]:
        try:
            float(string)
            string += 'Jy'
        except ValueError:
            pass
        finally:
            flux = qa(string)
            flux_jy = flux.canonical().get_value()*(10**26)
        return flux_jy
    else:
        return null_value

def convert2Hz(string):
    if string is not None and string != 'null':
        freq = qa(string)
        freq_hz = freq.canonical().get_value()
        return freq_hz
    
def convert2float(string, null_value=None):
    if string in [None, "null"]:
        return null_value
    numfloat = float(string) 
    #  else:
    #    print(f"string is null")
    return numfloat

@njit
def radec2lm(ra0: float, dec0: float, ra: Union[float, np.ndarray], dec: Union[float, np.ndarray]):
    """
    Convert RA and Dec to l and m coordinates.
    Args:
        ra0 (float): phase centre RA in radians.
        dec0 (float): phase centre Dec in radians.
        ra (float or np.ndarray): RA in radians.
        dec (float or np.ndarray): Dec in radians.
    """
    dra = ra - ra0
    l = np.cos(dec) * np.sin(dra) 
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)

    return l, m