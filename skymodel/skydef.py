from dataclasses import dataclass
from typing import List

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
