from typing import Union

import ephem
import numpy as np
from casacore.measures import measures
import ephem
from layouts import known
from typing import Union
from simms import constants, utilities

from .layouts import known


class Array:
    """
    The Array class has functions for converting from one coordinate system to another.
    """

    def __init__(self, layout: Union[str, utilities.File], degrees: bool = True):
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
        # check if the provided array is one of the default arrays.
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

        info = utilities.readyaml(fname)
        if vla:
            self.antlocations = np.array(info["antlocations"][self.layout])
        else:
            self.antlocations = np.array(info["antlocations"])

        self.centre = np.array(info["centre"])
        self.mount = info["mount"]
        self.names = info["antnames"]
        self.coordsys = info["coord_sys"]

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

        nnp = constants.earth_emaj / np.sqrt(1 - constants.esq * np.sin(latitude) ** 2)
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
        east = -np.sin(longitude) * delta_x + np.cos(longitude) * delta_y + 0 * delta_z
        north = (
            -np.sin(latitude) * np.cos(longitude) * delta_x
            - np.sin(latitude) * np.sin(longitude) * delta_y
            + np.cos(latitude) * delta_z
        )
        height = (
            np.cos(latitude) * np.cos(longitude) * delta_x
            + np.cos(latitude) * np.sin(longitude) * delta_y
            + np.sin(latitude) * delta_z
        )

        # arranging the components into an array.
        enu = np.column_stack((east, north, height))

        return enu

    def uvgen(
        self,
        longitude,
        latitude,
        pointing_direction,
        dtime,
        ntimes,
        start_freq,
        dfreq,
        nchan,
        date,
        start_time=None,
        start_ha=None,
    ) -> utilities.ObjDict:
        """
        Generate uvw coordimates

        Parameters
        ---

        longitude: float
                    : longitude of the observer
        latitude: float
                    : latitude of the observer
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
        date: str
                    : starting time of the observation.
        start_time: Union[str, List[str]]
                    : start hour time
        start_ha: float
                    : start hour angle

        Returns
        ---
        An array of the uvw time dependent positions, the time array and the frequency array
        """
        # casacore direction and epoch measures
        dm = measures()

        # xyz coordinates of the array
        positions_global, _ = self.geodetic2global()

        # convert the direction using measures
        ra_dec = dm.direction(*pointing_direction)
        ra = ra_dec["m0"]["value"]
        dec = ra_dec["m1"]["value"]

        tot_ha = ntimes * dtime / 3600 * constants.hour_angle
        # Determine hour angle range start
        if start_time:
            ih0 = self.get_start_ha(longitude, latitude, ra, start_time)
        elif start_ha:
            ih0 = start_ha
        else:
            ih0 = -tot_ha / 2

        h0 = ih0 + np.linspace(0, tot_ha, ntimes)

        first = [np.sin(h0), np.cos(h0), np.array([0.0 for _ in range(len(h0))])]

        second = [
            -np.sin(dec) * np.cos(h0),
            np.sin(dec) * np.sin(h0),
            np.array([np.cos(dec) for _ in range(len(h0))]),
        ]

        third = [
            np.cos(dec) * np.cos(h0),
            -np.cos(dec) * np.sin(h0),
            np.array([np.sin(dec) for _ in range(len(h0))]),
        ]

        transform_matrix = np.vstack([first, second, third])

        # the transformation matrix
        transform_matrix = np.array(
            [
                [np.sin(h0), np.cos(h0), np.array([0.0 for _ in range(len(h0))])],
                [
                    -np.sin(dec) * np.cos(h0),
                    np.sin(dec) * np.sin(h0),
                    np.array([np.cos(dec) for _ in range(len(h0))]),
                ],
                [
                    np.cos(dec) * np.cos(h0),
                    -np.cos(dec) * np.sin(h0),
                    np.array([np.sin(dec) for _ in range(len(h0))]),
                ],
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

        u_coord = (
            np.outer(transform_matrix[0, 0], bl_array[:, 0])
            + np.outer(transform_matrix[0, 1], bl_array[:, 1])
            + np.outer(transform_matrix[0, 2], bl_array[:, 2])
        )
        v_coord = (
            np.outer(transform_matrix[1, 0], bl_array[:, 0])
            + np.outer(transform_matrix[1, 1], bl_array[:, 1])
            + np.outer(transform_matrix[1, 2], bl_array[:, 2])
        )
        w_coord = (
            np.outer(transform_matrix[2, 0], bl_array[:, 0])
            + np.outer(transform_matrix[2, 1], bl_array[:, 1])
            + np.outer(transform_matrix[2, 2], bl_array[:, 2])
        )

        u_coord, v_coord, w_coord = [x.flatten() for x in (u_coord, v_coord, w_coord)]
        uvw = np.column_stack((u_coord, v_coord, w_coord))

        # starting time of the observation in seconds(since 1970)
        # start_time_rad = dm.epoch(start_time)["m0"]["value"]
        start_time_rad = dm.epoch(*date)["m0"]["value"]
        start_time_rad = start_time_rad * 24 * 3600

        # total time of observation
        total_time = start_time_rad + ntimes * dtime

        # the time table
        time_entries = np.arange(start_time_rad, total_time, dtime)

        start_freq = dm.frequency(v0=start_freq)["m0"]["value"]
        dfreq = dm.frequency(v0=dfreq)["m0"]["value"]

        total_bandwidth = start_freq + dfreq * nchan

        frequency_entries = np.arange(start_freq, total_bandwidth, dfreq)

        uvcoverage = utilities.ObjDict(
            {
                "antenna1": antenna1_list,
                "antenna2": antenna2_list,
                "uvw": uvw,
                "freqs": frequency_entries,
                "times": time_entries,
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
                baseline_entry = {"antenna1": i, "antenna2": j, "baseline": baseline}
                baseline_info.append(baseline_entry)
        return baseline_info

    # stolen from https://github.com/SpheMakh/uvgen/blob/master/uvgen.py
    def get_start_ha(self, longitude, latitude, ra, date):
        """
        TODO(mukundi) add docstring
        """

        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

        # Set up observer
        obs = ephem.Observer()
        obs.lon, obs.lat = longitude, latitude

        obs.date = date
        lst = obs.sidereal_time()

        def change(angle):
            if angle > 2 * np.pi:
                angle -= 2 * np.pi
            elif angle < 0:
                angle += 2 * np.pi
            return angle

        # Lets find transit (hour angle = 0, or LST=RA)
        lst, ra = map(change, (lst, ra))
        diff = (lst - ra) / (2 * np.pi)

        date = obs.date
        obs.date = date + diff
        # LST should now be transit
        transit = change(obs.sidereal_time())
        if ra == 0:
            obs.date = date - lst / (2 * np.pi)
        elif transit - ra > 0.1 * np.pi / 12:
            obs.date = date - diff

        # This is the time at transit
        ih0 = change((obs.date) / (2 * np.pi) % (2 * np.pi))
        # Account for the lower hemisphere
        if latitude < 0:
            ih0 -= np.pi
            obs.date -= 0.5

        return ih0
