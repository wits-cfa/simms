from dataclasses import dataclass
from typing import List
from simms import SCHEMADIR
import os
from simms.utilities import readyaml

class Base(object):
    def __init__(self, schemafile=None):
        self.schemafile = schemafile
    
    def set_schema(self, schemafile=None):
        schemafile = schemafile or self.schemafile

        if os.path.exists(schemafile):
            self.schema = readyaml(schemafile)
        else:
            raise FileNotFoundError(f"Schema file '{schemafile}' could not be found")

@dataclass
class Line(Base):
    freq_peak: float
    width: int
    stokes: float
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Line"


@dataclass
class Cont(Base):
    ref_freq: float
    stokes: float
    coeffs: List[float]
    schema_section: str = "Cont"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
