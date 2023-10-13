from dataclasses import dataclass
from typing import List, Any, Union
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
    schemafile: str = os.path.join(SCHEMADIR, "schema_observation.yaml")
    schema_section: str = "Antenna"

@dataclass
class ArrayTelescope(SpecBase):
    name: str
    centre: List[float]
    antlocations: List[Any]
    antnames: List[str] = None
    groups: List[File] = None
    coord_sys: str = "geodetic"
    schemafile: str = os.path.join(SCHEMADIR, "schema_observation.yaml")
    schema_section: str = "Array"

@dataclass
class Observation(SpecBase):
    name: str
    desc: str
    telescope: Any
    direction: List[str]
    start_time: str
    dtime: float = 10
    ntimes: int = 10 
    start_freq: Union[str,float] = "900MHz"
    dfreq: Union[str,float] = "2MHz"
    nchan: int = 10
    correlations: List[str] = ["XX", "YY"]
    schemafile: str = os.path.join(SCHEMADIR, "schema_observation.yaml")
    schema_section: str = "Observation"
    
