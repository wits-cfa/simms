import os
from dataclasses import dataclass
from typing import List, Any, Union
from simms import SCHEMADIR
from simms.utilities import File
from simms.config_spec import SpecBase
from .array_utilities import Array

@dataclass
class Antenna(SpecBase):
    name: str
    mount: str = "ALT-AZ"
    size: float = None
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

    def set_antennas(self):
        pass


@dataclass
class Observation(SpecBase):
    ms: str
    antennas: Union[File, str]
    direction: List[str]
    dtime: float
    ntimes: int
    start_freq: Union[str, float]
    desc: str = None
    start_hour_angle: float = None
    start_time: str = None
    dfreq: Union[str,float] = None
    nchan: int = None
    correlations: List[str] = None
    longitude: float = None
    latitude: float = None
    schemafile: str = os.path.join(SCHEMADIR, "schema_observation.yaml")
    schema_section: str = "Observation"

    def set_array(self):
        """
        Creates an ArrayTelescope instance
        """

        antennas = Array(self.antennas, degrees=True)
        antennas.set_arrayinfo()

        


