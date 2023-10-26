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
    
    #stolen from https://github.com/SpheMakh/uvgen/blob/master/uvgen.py
    def source_info(self,longitude,latitude):

        
        dm = measures()
        longitude = np.rad2deg(longitude)
        latitude = np.deg2rad(latitude)
   
        # Set up observer        
        obs = ephem.Observer()
        obs.lon, obs.lat = longitude,latitude

        if len(self.direction) ==3:
                
            ra = dm.direction(self.direction[0],self.direction[1],self.direction[2])['m0']['value']
            dec = dm.direction(self.direction[0],self.direction[1],self.direction[2])['m1']['value']
        else:
            ra = dm.direction(self.direction[0],self.direction[1])['m0']['value']
            dec = dm.direction(self.direction[0],self.direction[1])['m1']['value']
        
      
        def sunrise_equation(latitude,dec):
            arg = -np.tan(latitude) * np.tan(dec)
            if arg > 1 or arg< -1:
                if latitude*dec < 0:
                    print("Pointing center is always below the horizon!")
                    return 0
                else:
                    print("Pointing center is always above horizon")
                    return 0
            th_ha = np.arccos( arg )
            return th_ha
        

        obs.date = self.start_time or "2023/10/25 12:0:0"#%(time.localtime()[:3])
        lst = obs.sidereal_time() 

        def change (angle):
            if angle > 2*np.pi:
                angle -= 2*np.pi
            elif angle < 0:
                angle += 2*np.pi
            return angle
 
        altitude_transit = lambda latitude, dec: np.sign(latitude)*(np.cos(latitude)*np.sin(dec) + np.sin(latitude)*np.cos(dec) )
                
        # First lets find the altitude at transit (hour angle = 0 or LST=RA)
        # If this is negative, then the pointing direction is below the horizon at its peak.
        #alt_trans = altitude_transit(lat,dec)
        #if alt_trans < 0 :
        #    warn(" Altitude at transit is %f deg, i.e."
        #         " pointing center is always below the horizon!"%(numpy.rad2deg(alt_trans)))
        #    return 0

        altitude = altitude_transit(latitude,dec)
        H0 = sunrise_equation(latitude,dec)

        # Lets find transit (hour angle = 0, or LST=RA)
        lst,ra = map(change,(lst,ra))
        diff =  (lst - ra )/(2*np.pi)

        date = obs.date
        obs.date = date + diff
        # LST should now be transit
        transit = change(obs.sidereal_time())
        if ra==0:
            obs.date = date - lst/(2*np.pi)
        elif transit-ra > .1*np.pi/12:
            obs.date = date - diff

        # This is the time at transit
        ih0 = change((obs.date)/(2*np.pi)%(2*np.pi))
        # Account for the lower hemisphere
        if latitude<0:
            ih0 -= np.pi
            obs.date -= 0.5
        
        date = obs.date.datetime().ctime()
        return ih0, date, H0, altitude
  
    def get_uvw(self,pointing_direction,data_file = None,observatory=None):
        """
        Get the uvw baseline positions using the global2uvw function in Array class
        """
        pointing_direction = pointing_direction or self.direction
        observatory = self.telescope or observatory

        initialization = array_utilities.Array(pointing_direction=pointing_direction,
                                              data_file=data_file,
                                              observatory=observatory)
        uvw_positions = initialization.global2uvw()
        
        return uvw_positions

    def generate_observation(self):
        """
        Given the observation information, generate observation data.
        """

        dm = measures()
        uvw_positions = self.get_uvw(pointing_direction=self.direction)
        
        #starting time of observation. should be given as a list of str with epoch
        start_time = self.start_time or ['UTC','2023/10/25 12:0:0']
       
        start_time_rad = dm.epoch(start_time[0],start_time[1])['m0']['value']
        start_time_rad = start_time_rad * 24 * 3600
        #total time of observation
        total_time = start_time_rad + self.ntimes*self.dtime

        #the time entries
        time_entries = np.arange(start_time_rad,total_time,self.dtime)

        observation_data = []

        for time_entry in time_entries:
            observation_data.append((time_entry,uvw_positions))
        
        return observation_data



obj = Observation(name='test',
                  desc='test 1',
                  telescope='meerkat',
                  direction = ['J2000','0deg','-30deg'],
                  start_time=['UTC','2023/10/25 15:0:0'],
                  dtime=5,
                  ntimes= 10)
observ = obj.generate_observation()
print(f'first dump \n {observ[0]}')
print(f'second dump \n {observ[9]}')