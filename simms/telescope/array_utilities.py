from datetime import datetime
from typing import Union

import astropy.units as u
import ephem
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from casacore import quanta as qa
from casacore.measures import dq, measures
from casacore.tables import table
from omegaconf import OmegaConf
from scabha.basetypes import File, List

from simms import constants, get_logger
from simms.telescope.layouts import known, unknown
from simms.utilities import ObjDict

log = get_logger(name="telsim")


class Array:
    """
    The Array class has functions for converting from one coordinate system to another.
    """

    def __init__(self, layout: Union[str, File], degrees: bool = True,
                 sefd: Union[int, float, List[Union[int, float]]]=None):
        """
        layout: str|File
                    : specify an observatory as a str or a file.
                        If a string is given, then will attenpt to find
                        layout in package database
        degrees: boolean
                    : Specify whether the long-lat is in degrees or not. Default is true.
        """

        self.degrees = degrees
        self.observatories = known()
        if layout not in self.observatories:
            self.layoutname, self.layout = unknown(layout)
        else:
            self.layout = layout
        self.sefd = sefd


    def set_arrayinfo(self):
        """
        Extract the array information from the schema
        """
        fname = None
        vla = None
        if isinstance(self.layout, str):
            if self.layout.startswith("vla-"):
                fname = self.observatories["vla"]
                vla = True
            else:
                fname = self.observatories[self.layout]
                vla = False
        else:
            fname = self.layout.get(self.layoutname)

        info = OmegaConf.load(fname)
        if vla:
            self.antlocations = np.array(info["antlocations"][self.layout])
        else:
            self.antlocations = np.array(info["antlocations"])

        self.centre = np.array(info["centre"])
        self.mount = info["mount"]
        self.names = info["antnames"]
        self.coordsys = info["coord_sys"]
        self.size = info["size"]
        if self.sefd is None:
            sefd = info.get("sefd", None)
            self.sefd = sefd

            if isinstance(sefd, (float, int)):
                self.sefd = [sefd]
            elif (not isinstance(sefd, str)) and isinstance(sefd, (list, List)):
                self.sefd = sefd

    def get_itrf_positions(self):
        """
        Get the ITRF positions of the antennas
        """
        self.set_arrayinfo()

        # convert to radians
        if self.degrees and self.coordsys.lower() == "geodetic":
            if self.antlocations.shape[1] == 3:
                self.altitudes = self.antlocations[:, 2]
            else:
                self.altitudes = np.zeros(self.antlocations.shape[0])
            self.antlocations = np.deg2rad(self.antlocations[:, :2])
            self.centre_altitude = self.centre[2]
            self.centre = np.deg2rad(self.centre[:2])
            itrf_positions, _ = self.geodetic2global()
        
        elif self.coordsys.lower() == "itrf":
            itrf_positions = self.antlocations
            self.centre = self.global2geodetic(self.centre[0],self.centre[1],self.centre[2])
            

        else:
            raise ValueError("Unknown coordinate system. Please use Geodetic (WGS84) or ITRF (XYZ)")     
            
        return itrf_positions
            
    def geodetic2global(self):
        """
        Convert the antenna positions from the geodetic (WGS84) frame to the global (ITRF/ECEF) frame

        Returns
        ----
        An array of the antennas XYZ positions and an array of the array center in XYZ
        """


        longitude = self.antlocations[:, 0]
        latitude = self.antlocations[:, 1]
        altitude = self.altitudes

        ref_longitude, ref_latitude= self.centre
        ref_altitude = self.centre_altitude

        nnp = constants.earth_emaj / np.sqrt(1 - constants.esq * np.sin(latitude)**2)
        nn0 = constants.earth_emaj / np.sqrt(1 - constants.esq * np.sin(ref_latitude**2))

        # calculating the global coordinates of the antennas.
        x = (nnp + altitude) * np.cos(latitude) * np.cos(longitude)
        y = (nnp + altitude) * np.cos(latitude) * np.sin(longitude)
        z = ((1 - constants.esq) * nnp + altitude) * np.sin(latitude)

        # calculating the global coordinates of the array center.
        x0 = (nn0 + ref_altitude) * np.cos(ref_latitude) * np.cos(ref_longitude)
        y0 = (nn0 + ref_altitude) * np.cos(ref_latitude) * np.sin(ref_longitude)
        z0 = ((1 - constants.esq) * nn0 + ref_altitude) * np.sin(ref_latitude)

        xyz = np.column_stack((x, y, z))
        xyz0 = np.column_stack((x0, y0, z0))

        return xyz, xyz0
    
    
    def global2geodetic(self,X,Y,Z):
        """
        Convert the antenna positions from the global (ITRF/ECEF) frame to the geodetic (WGS84) frame.
        
        Parameters
        ----
        (X,Y,Z): float
                : global coordinates of the antennas.
        
        Returns
        ----
        (phi,lam,h): float
                : geodetic coordinates of the antennas.
        """
        
        
        f=1/298.257223563
        b = constants.earth_emaj * (1 - f)
        ep2 = (constants.earth_emaj**2 - b**2) / b**2

        p = (X**2 + Y**2)**0.5
        theta = np.arctan2(Z * constants.earth_emaj, p * b)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        phi = np.arctan2(Z + ep2 * b * sin_theta**3, p - constants.esq * constants.earth_emaj * cos_theta**3)
        lam = np.arctan2(Y, X)
        v = constants.earth_emaj / (1 - constants.esq * np.sin(phi)**2)**0.5
        h = p / np.cos(phi) - v
        
        geodetic = np.array((lam, phi, h))

        return geodetic
        

    def geodetic2local(self):
        """
        Converts the antenna positions from the geodetic (WGS84) frame to the local (ENU) frame

        Returns
        ---
        An array of the antenna positions in the local frame ENU
        """
        if not hasattr(self, "antlocations"):
            self.set_arrayinfo()

        longitude = self.antlocations[:, 0]
        latitude = self.antlocations[:, 1]

        # we need the positions in xyz
        xyz, xyz0 = self.geodetic2global()
        x, y, z = zip(*xyz)
        x0, y0, z0 = xyz0[0, 0], xyz0[0, 1], xyz0[0, 2]

        delta_x = x - x0
        delta_y = y - y0
        delta_z = z - z0

        cos_long = np.cos(longitude)
        sin_long = np.sin(longitude)
        cos_lat = np.cos(latitude)
        sin_lat = np.sin(latitude)

        east = -sin_long * delta_x + cos_long * delta_y + 0 * delta_z
        north = -sin_lat * cos_long * delta_x - sin_lat * sin_long * delta_y + cos_lat * delta_z
        height = cos_lat * cos_long * delta_x + cos_lat * sin_long * delta_y + sin_lat * delta_z

        enu = np.column_stack((east, north, height))

        return enu

    def uvgen(self, pointing_direction, dtime, ntimes, start_freq, dfreq,
              nchan, start_time=None, start_ha=None) -> ObjDict:
        """
        Generate uvw coordimates

        Parameters
        ---

        pointing_direction: List[str]
                    : pointing direction.
        dtime: int
                    : integration time.
        ntimes: int
                    : number of times the sky should be snapped.
        start_freq: Union[str,float]
                    : starting freq of the observation.
        dfreq: Union[str,float]
                    : frequency interval.
        nchan: int
                    : number of channels.

        start_time: Union[str, List[str]]
                    : start time of the observation date and time ("YYYY/MM/DD 12:00:00", ["EPOCH", "YYYY/MM/DD 12:00:00"])
                        default is the current machine time.
        start_ha: float
                    : start hour angle in radians

        Returns
        ---
        An array of the uvw time dependent positions, the time array and the frequency array
        """
        dm = measures()

        # xyz coordinates of the array
        # positions_global, _ = self.geodetic2global()
        
        
        
        positions_global = self.get_itrf_positions()
        # longitude = self.centre[0]
        latitude = self.centre[1]

        if len(pointing_direction) == 3:
            ra_dec = dm.direction(rf=pointing_direction[0], v0 = pointing_direction[1], v1=pointing_direction[2])
        else:
            ra_dec = dm.direction(rf='J2000', v0 = pointing_direction[1], v1=pointing_direction[2])
        ra = ra_dec["m0"]["value"]
        dec = ra_dec["m1"]["value"]
         
        
        if not start_time:
            date = datetime.now()
            start_time = date.strftime("%Y-%m-%dT%H:%M:%S")
            start_day = Time(start_time, format="isot", scale="utc")
        else:
            if isinstance(start_time, str):

                start_day = Time(start_time, format="isot", scale="utc")

        start_time_sec = start_day.to_value("mjd") * 24 * 3600
        total_time = ntimes * dtime

        time_entries = np.arange(start_time_sec, start_time_sec+total_time, dtime)

        if start_ha:
            ih0 = start_ha * constants.PI
        else:
            # obs_location = EarthLocation(lon=longitude * u.rad, lat=latitude * u.rad)
            gmst = start_day.sidereal_time(kind="mean", longitude=0*u.deg)
            gmst_rad = gmst.to_value("rad")
            gha = gmst_rad - ra
            gha = gha % (2 * constants.PI)

            start_day_rads = start_day.to_value("mjd")%1 
            start_day_rads *= 24 * 15 * constants.PI/180
            ih0 = gha

        total_time_rad = np.deg2rad(total_time/3600*15) 
        h0 = ih0 + np.linspace(-total_time_rad/2, total_time_rad/2, ntimes)
        
        source_elevations = self.get_source_elevation(
            latitude, dec, h0)
        
        # antenna_elevations = self.get_antenna_elevation(
        #     self.antlocations[:, 1], dec, h0, ntimes).T
        # antenna_elevations = antenna_elevations.flatten()
        # print(f"antenna_elevations: {antenna_elevations.shape}")


        # Transformation matrix
        dec = np.ones(ntimes) * dec
        transform_matrix = np.array(
            [
                [np.sin(h0), np.cos(h0), np.zeros(ntimes)],
                [-np.sin(dec) * np.cos(h0), np.sin(dec) * np.sin(h0), np.cos(dec)],
                [-np.cos(dec) * np.cos(h0), np.cos(dec) * np.sin(h0), np.sin(dec)]
            ]
        )

        antenna1_list = []
        antenna2_list = []
        baseline_list = []
        baselines_info = self.baseline_info(antlocations=positions_global)
        for base in baselines_info:
            bl = base["baseline"]
            antenna1 = base["antenna1"]
            antenna2 = base["antenna2"]
            baseline_list.append(bl)
            antenna1_list.append(antenna1)
            antenna2_list.append(antenna2)

        bl_array = np.vstack(baseline_list)

        u_coord = (np.outer(transform_matrix[0, 0], bl_array[:, 0]) + np.outer(
            transform_matrix[0, 1], bl_array[:, 1]) + np.outer(transform_matrix[0, 2], bl_array[:, 2]))
        v_coord = (np.outer(transform_matrix[1, 0], bl_array[:, 0]) + np.outer(
            transform_matrix[1, 1], bl_array[:, 1]) + np.outer(transform_matrix[1, 2], bl_array[:, 2]))
        w_coord = (np.outer(transform_matrix[2, 0], bl_array[:, 0]) + np.outer(
            transform_matrix[2, 1], bl_array[:, 1]) + np.outer(transform_matrix[2, 2], bl_array[:, 2]))

        u_coord, v_coord, w_coord = [x.flatten() for x in (u_coord, v_coord, w_coord)]
        uvw = np.column_stack((u_coord, v_coord, w_coord))

        time_table = []
        for time_entry in time_entries:
            baseline_time = [time_entry] * len(baseline_list)
            time_table.append(baseline_time)

        time_table = np.array(time_table).flatten()

        start_freq = dm.frequency(v0=start_freq)["m0"]["value"]
        dfreq = dm.frequency(v0=dfreq)["m0"]["value"]
        total_bandwidth = start_freq + dfreq * nchan
        frequency_entries = np.arange(start_freq, total_bandwidth, dfreq)

        uvcoverage = ObjDict(
            {
                "antenna1": antenna1_list,
                "antenna2": antenna2_list,
                "uvw": uvw,
                "freqs": frequency_entries,
                "times": time_table,
                "source_elevations": source_elevations,
               
            }
        )

        return uvcoverage


    def baseline_info(self, antlocations):
        """
        This function calculates the baselines and store the
        information in a dictionary.
        """
        baseline_info = []
        for i in range(antlocations.shape[0]):
            for j in range(i + 1, antlocations.shape[0]):
                baseline = antlocations[j] - antlocations[i]
                baseline_entry = {"antenna1": i,
                                  "antenna2": j,
                                  "baseline": baseline}
                baseline_info.append(baseline_entry)
        return baseline_info
    
    
    
    def get_source_elevation(self,latitude,declination,hour_angles):
        """
        Track the source during the observation and get its elevation.
        
        Parameters
        ----
        latitude: float
                : latitude of the observer in radians.
        declination: float
                : declination of the source in radians.
        hour_angles: array[float]
                : hour angles of the source in radians.
                
        Returns
        ----
        source_elevations: array[float]
                : the elevations of the source in degrees.
        """
        
        sin_elevation = np.sin(declination) * np.sin(latitude) + \
            np.cos(declination) * np.cos(latitude) * np.cos(hour_angles)
    
        elevation = np.degrees(np.arcsin(sin_elevation))
        
        
        return elevation
        


    def get_antenna_elevation(self,ant_latitudes,dec,h0,ntimes):
        """
        Get the elevations of the antennas at the observing times.
            
        Parameters
        ---
        
        ant_latitudes: Array[float]
                : latitudes of all the antennas in radians.
        declination: float
                : declination of the source in radians.
        h0: Array[float]
                : hour angles of the source in radians

        Returns
        ---
        
        antenna_elevations: Array[float]
                : the elevations of the antennas in degrees.
        """
        
        nants = ant_latitudes.shape[0]
        antenna_elevations = np.zeros((nants,ntimes))

        for i in range(nants):
            antenna_elevations[i] = np.sin(dec) * np.sin(ant_latitudes[i]) + \
            np.cos(dec)* np.cos(ant_latitudes[i]) * np.cos(h0)
        
        altitude = np.degrees(np.arcsin(antenna_elevations))
         
        return altitude
        
             
        
        

def ms_addrow(ms,subtable,nrows):
    
    subtab = table(f"{ms}::{subtable}",
                    readonly=False, lockoptions='user', ack=False)
    try:
        subtab.lock(write=True)
        subtab.addrows(nrows)
        
    finally:
        subtab.unlock()
        subtab.close()
