from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from simms import SCHEMADIR
from simms.utilities import File
import os
from simms.config_spec import SpecBase

@dataclass
class Antenna(SpecBase):
    name: str
    mount: str = "ALT-AZ"
    size: float = 13.5
    sefd: float = None
    tsys_over_eta: float = 0 # Zero is unset
    sensitivity_file: File = None
    schema_section: str = "Antenna"
    antnames: List[str] = None

@dataclass
class Array(SpecBase):
    name: str
    centre: List[float]
    antlocations: List[Any]
    antnames: List[str] = None
    groups: List[File] = None
    coord_sys: str = "geodetic"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Array"
