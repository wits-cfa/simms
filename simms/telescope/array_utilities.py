from datetime import datetime
from typing import Union
import ephem
import numpy as np
from casacore.measures import measures
from omegaconf import OmegaConf
from simms import constants
from .layouts import known
from scabha.basetypes import File
from simms.utilities import ObjDict


class Array:
    """
    The Array class has functions for converting from one coordinate system to another.
    """

    def __init__(self, layout: Union[str, File], degrees: bool = True):
        """
        layout: str|File
                    : specify an observatory as a str or a file.
                        If a string is given, then will attenpt to find
                        layout in package database
        degrees: boolean
                    : Specify whether the long-lat is in degrees or not. Default is true.


        """

        self.layout = layout
        self.degrees = degrees
        self.observatories = known()

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
            fname = self.layout

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

        if self.degrees and self.coordsys.lower() == "geodetic":
            self.antlocations = np.deg2rad(self.antlocations)
            self.centre = np.deg2rad(self.centre)

    def geodetic2global(self):
        """
        Convert the antenna positions from the geodetic frame to the global frame

        Returns
        ----
        An array of the antennas XYZ positions and an array of the array center in XYZ
        """

        self.set_arrayinfo()

        longitude = self.antlocations[:, 0]
        latitude = self.antlocations[:, 1]
        altitude = self.antlocations[:, 2]

        ref_longitude, ref_latitude, ref_altitude = self.centre

        nnp = constants.earth_emaj / \
            np.sqrt(1 - constants.esq * np.sin(latitude) ** 2)
        nn0 = constants.earth_emaj / \
            np.sqrt(1 - constants.esq * np.sin(ref_latitude**2))

        # calculating the global coordinates of the antennas.
        x = (nnp + altitude) * np.cos(latitude) * np.cos(longitude)
        y = (nnp + altitude) * np.cos(latitude) * np.sin(longitude)
        z = ((1 - constants.esq) * nnp + altitude) * np.sin(latitude)

        # calculating the global coordinates of the array center.
        x0 = (nn0 + ref_altitude) * \
            np.cos(ref_latitude) * np.cos(ref_longitude)
        y0 = (nn0 + ref_altitude) * \
            np.cos(ref_latitude) * np.sin(ref_longitude)
        z0 = ((1 - constants.esq) * nn0 + ref_altitude) * np.sin(ref_latitude)

        xyz = np.column_stack((x, y, z))
        xyz0 = np.column_stack((x0, y0, z0))

        return xyz, xyz0

    def geodetic2local(self):
        """
        Converts the antenna positions from the geodetic frame to the local frame

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

        # calculate the vector from the origin to the antenna position
        delta_x = x - x0
        delta_y = y - y0
        delta_z = z - z0

        # local frame components.
        east = -np.sin(longitude) * delta_x + \
            np.cos(longitude) * delta_y + 0 * delta_z
        north = (-np.sin(latitude) * np.cos(longitude) * delta_x - np.sin(latitude)
                 * np.sin(longitude) * delta_y + np.cos(latitude) * delta_z)
        height = (np.cos(latitude) * np.cos(longitude) * delta_x + np.cos(latitude)
                  * np.sin(longitude) * delta_y + np.sin(latitude) * delta_z)

        # arranging the components into an array.
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
        positions_global, _ = self.geodetic2global()

        # get the array centre info
        self.set_arrayinfo()
        longitude = self.centre[0]
        latitude = self.centre[1]

        ra_dec = dm.direction(*pointing_direction)
        ra = ra_dec["m0"]["value"]
        dec = ra_dec["m1"]["value"]

        tot_ha = ((ntimes * dtime) / 3600) * constants.hour_angle
        if start_ha or start_time:
            if start_ha:
                ih0 = start_ha
            else:
                if isinstance(start_time, str):
                    ih0 = self.get_start_ha(
                        longitude, latitude, ra, start_time)
                else:
                    split_start_time = start_time[1]
                    ih0 = self.get_start_ha(
                        longitude, latitude, ra, split_start_time)
        else:
            ih0 = -tot_ha / 2

        h0 = ih0 + np.linspace(0, tot_ha, ntimes)

        # Transformation matrix
        transform_matrix = np.array(
            [
                [np.sin(h0), np.cos(h0), np.array(
                    [0.0 for _ in range(len(h0))])],
                [-np.sin(dec) * np.cos(h0), np.sin(dec) * np.sin(h0),
                 np.array([np.cos(dec) for _ in range(len(h0))])],
                [np.cos(dec) * np.cos(h0), -np.cos(dec) * np.sin(h0),
                 np.array([np.sin(dec) for _ in range(len(h0))])],
            ]
        )

        # calculating the baselines
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

        u_coord, v_coord, w_coord = [x.flatten()
                                     for x in (u_coord, v_coord, w_coord)]
        uvw = np.column_stack((u_coord, v_coord, w_coord))

        if not start_time:
            date = datetime.now()
            start_time = date.strftime("%Y/%m/%d %H:%M:%S")
            start_day = dm.epoch(rf='UTC', v0=start_time)["m0"]["value"]
        else:
            if isinstance(start_time, str):
                start_day = dm.epoch(rf='UTC', v0=start_time)["m0"]["value"]
            else:
                start_day = dm.epoch(*start_time)["m0"]["value"]
        start_time_sec = start_day * 24 * 3600
        total_time = start_time_sec + ntimes * dtime

        time_entries = np.arange(start_time_sec, total_time, dtime)

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
                                  "antenna2": j, "baseline": baseline}
                baseline_info.append(baseline_entry)
        return baseline_info

    def get_start_ha(self, longitude, latitude, ra, date):
        """
        TODO(mukundi and galefang) add docstring
        """

        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

        obs = ephem.Observer()
        obs.lon, obs.lat = longitude, latitude

        obs.date = date
        lst = obs.sidereal_time()

        def change(angle):
            if angle > constants.two_pi:
                angle -= constants.two_pi
            elif angle < 0:
                angle += constants.two_pi
            return angle

        lst, ra = map(change, (lst, ra))
        diff = (lst - ra) / constants.two_pi

        date = obs.date
        obs.date = date + diff

        transit = change(obs.sidereal_time())
        if ra == 0:
            obs.date = (date - lst) / constants.two_pi
        elif transit - ra > 0.1 * constants.hour_angle:
            obs.date = date - diff

        ih0 = change(((obs.date) / constants.two_pi) % constants.two_pi)
        if latitude < 0:
            ih0 -= np.pi
            obs.date -= 0.5

        return ih0
