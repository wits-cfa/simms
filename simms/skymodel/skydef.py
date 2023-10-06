from dataclasses import dataclass
from typing import List, Optional
from simms import SCHEMADIR
import os
from simms.utilities import readyaml, get_class_attributes, ValidationError

class Base(object):
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
        

@dataclass
class Line(Base):
    freq_peak: float
    width: int
    stokes: List[float]
    restfreq: Optional[float] = None
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Line"

    
@dataclass
class Cont(Base):
    ref_freq: float
    stokes: List[float]
    coeffs: Optional[List[float]] = None
    schema_section: str = "Cont"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
