from dataclasses import dataclass
from typing import List, Optional
from simms import SCHEMADIR
import os
from simms.config_spec import SpecBase


@dataclass
class Line(SpecBase):
    freq_peak: float
    width: int
    stokes: List[float]
    restfreq: Optional[float] = None
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Line"

    
@dataclass
class Cont(SpecBase):
    ref_freq: float
    stokes: List[float]
    coeffs: Optional[List[float]] = None
    schema_section: str = "Cont"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
