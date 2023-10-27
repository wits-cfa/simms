from dataclasses import dataclass
from typing import List, Any, Union
from simms import SCHEMADIR
from simms.utilities import File
import os
from simms.config_spec import SpecBase
from simms.telescope import array_utilities
from casacore.measures import measures
import numpy as np
import ephem


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

@dataclass
class Observation(SpecBase):
    name: str
    desc: str
    telescope: Any
    direction: List[str]
    start_time: str
    dtime: float = None
    ntimes: int = None 
    start_freq: Union[str,float] = None
    dfreq: Union[str,float] = None
    nchan: int = None
    correlations: List[str] = None
    schemafile: str = os.path.join(SCHEMADIR, "schema_observation.yaml")
    schema_section: str = "Observation"
    
    
<<<<<<< HEAD
    def get_uvw(self,h0,longitude,latitude,data_file = None,observatory=None):
        """
        Get the uvw baseline positions using the global2uvw function in Array class
        """
        pointing_direction = self.direction
=======
    def get_uvw(self,h0,longitude,latitude,pointing_direction,data_file = None,observatory=None):
        """
        Get the uvw baseline positions using the global2uvw function in Array class
        """
        pointing_direction = pointing_direction or self.direction
>>>>>>> 1b54b03711f2f3e97f2dd266c172bbe809f7b28d
        observatory = self.telescope or observatory

        initialization = array_utilities.Array(pointing_direction=pointing_direction,
                                              data_file=data_file,
                                              observatory=observatory)
        
        uvw,time = initialization.global2uvw(h0=h0,longitude = longitude,
                                                  latitude = latitude,
                                                  date = self.start_time,
                                                  dtime = self.dtime,
                                                  ntimes = self.ntimes)
        
        return uvw,time

  



<<<<<<< HEAD
obj = Observation(name = 'test',
                  desc = 'testing',
                  telescope='meerkat',
                  direction = ['J200','0deg','-30deg'],
                  start_time=['UTC','2023/10/25 12:0:0'],
                  dtime = 10,
                  ntimes = 10
                  )
obj_uvw = obj.get_uvw(h0 = [-1,1],
            longitude = 21.440810448238672,
                   latitude = -30.714037075541743,
                   )
print(obj_uvw[0].shape)
=======
>>>>>>> 1b54b03711f2f3e97f2dd266c172bbe809f7b28d
