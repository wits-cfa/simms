import numpy as np
from matplotlib import pyplot as plt
import yaml
import os


class Array(object):
    """
    This class is used to convert antenna positions between different frames.
    Geodetic - Local, Geodetic - Global, Global-uvw.
    """
    #predefine antenna positions so the user can just specify the array to use
    predefined_antennas = {
        'meerkat':'configs/meerkat.geodetic.yaml',
        'kat-7':'configs/kat-7.geodetic.yaml',
        'vla.a':'configs/vla.a.geodetic.yaml',
        'vla.b':'configs/vla.b.geodetic.yaml',
        'vla.c':'configs/vla.c.geodetic.yaml',
        'vla.d':'configs/vla.d.geodetic.yaml',
        'vla.bna':'configs/vla.bna.geodetic.yaml',
        'vla.cnb':'configs/vla.cnb.geodetic.yaml',
        'vla.dnc':'configs/vla.dnc.geodetic.yaml',
        'wsrt':'configs/WSRT.geodetic.yaml',
        'ngvla-core-revC':'configs/ngvla-core-revC.geodetic.yaml',
        'ngvla-core-revB':'configs/ngvla-core-revB.geodetic.yaml',
        'ngvla-lba-revC':'configs/ngvla-lba-revC.geodetic.yaml',
        'ngvla-mid-subarray-revC':'configs/ngvla-mid-subarray-revC.geodetic.yaml',
        'ngvla-sba-revB':'configs/ngvla-sba-revB.geodetic.yaml',
        'ngvla-sba-revC':'configs/ngvla-sba-revC.geodetic.yaml',
        'ngvla-plains-revC':'configs/ngvla-plains-revC.geodetic.yaml',
        'ngvla-plains-revB':'configs/ngvla-plains-revB.geodetic.yaml',
        'ngvla-main-revC':'configs/ngvla-main-revC.geodetic.yaml',
        'ngvla-gb-vlba-revB':'configs/ngvla-gb-vlba-revB.geodetic.yaml',
        'atca':'configs/atca_all.geodetic.yaml',
        'atca_1.5a':'configs/atca_1.5a.geodetic.yaml',
        'atca_1.5b':'configs/atca_1.5b.geodetic.yaml',
        'atca_1.5c':'configs/atca_1.5c.geodetic.yaml',
        'atca_1.5d':'configs/atca_1.5d.geodetic.yaml',
        'atca_6a':'configs/atca_6a.geodetic.yaml',
        'atca_6b':'configs/atca_6b.geodetic.yaml',
        'atca_6c':'configs/atca_6c.geodetic.yaml',
        'atca_6d':'configs/atca_6d.geodetic.yaml'

         }
    predefined_centers = {
        'meerkat':'configs/meerkat.center.yaml',
        'kat-7':'configs/kat-7.center.yaml',
        'vla.a':'configs/vla.center.yaml',
        'vla.b':'configs/vla.center.yaml',
        'vla.c':'configs/vla.center.yaml',
        'vla.d':'configs/vla.center.yaml',
        'vla.bna':'configs/vla.center.yaml',
        'vla.cnb':'configs/vla.center.yaml',
        'vla.dnc':'configs/vla.center.yaml',
        'vla':'configs/vla.center.yaml',
        'ngvla-core-revC':'configs/vla.center.yaml',
        'ngvla-core-revB':'configs/vla.center.yaml',
        'ngvla-lba-revC':'configs/vla.center.yaml',
        'ngvla-mid-subarray-revC':'configs/vla.center.yaml',
        'ngvla-sba-revB':'configs/vla.center.yaml',
        'ngvla-sba-revC':'configs/vla.center.yaml',
        'ngvla-plains-revC':'configs/vla.center.yaml',
        'ngvla-plains-revB':'configs/vla.center.yaml',
        'ngvla-main-revC':'configs/vla.center.yaml',
        'ngvla-gb-vlba-revB':'configs/vla.center.yaml',
        'wsrt':'configs/wsrt.center.yaml',
        'atca':'configs/atca.center.yaml',
        'atca_1.5a':'configs/atca.center.yaml',
        'atca_1.5b':'configs/atca.center.yaml',
        'atca_1.5c':'configs/atca.center.yaml',
        'atca_1.5d':'configs/atca.center.yaml',
        'atca_6a':'configs/atca.center.yaml',
        'atca_6b':'configs/atca.center.yaml',
        'atca_6c':'configs/atca.center.yaml',
        'atca_6d':'configs/atca.center.yaml'
        }

    def __init__(self, 
                 point_dir,
                 array_center=None, 
                 antennas=None,
                 array_name=None,
                 degrees=True):
        """
        antennas: A yaml file.
                    : Positions of the antennas.
        antenna_name: str
                    : The user can specify an array.
        array_center: List[float]
                    : Latitude,longitude and altitude for the array center.
        point_dir: Union[float, str]
                    : Pointing direction. Given in degrees. 
                     Range should be [0-360,-90-90].
        """

        #check if antenna name is given.
        if array_name is not None and array_center is None:
            #if its given, check if its in the predefined arrays
            if array_name in self.predefined_antennas:
                #if it is, get it.
                antennas = self.predefined_antennas[array_name]
                with open(antennas, 'r') as file:
                    antenna_data = yaml.safe_load(file)
                    self.antennas = np.array(antenna_data)

                #array center
                array_c= self.predefined_centers[array_name]
                with open(array_c,'r') as file:
                    ref = yaml.safe_load(file)
                    self.array_center = ref
                
             #if antenna_name is not in predefined arrays, raise an error.       
            else:
                raise ValueError(f"Antenna array '{array_name}' not found in predefined arrays.")
            
        #if antennas positions file is given as input as a yaml file, read the file
        elif antennas is not None and array_center is not None:
            with open(antennas,'r') as file:
                antenna_data = yaml.safe_load(file)
                self.antennas = np.array(antennas)
                self.array_center = array_center
        #if both cases fail, raise an error.
        else:
            raise ValueError('Either name of the array or antenna positions and center should be provided.')


        # Check if the antennas argument is a string (file path)
        #if isinstance(antennas, str):
          #  if antennas.endswith(".yaml"):
                # Load antenna data from a YAML file
            #    with open(antennas, 'r') as file:
            #        antenna_data = yaml.safe_load(file)
            #        self.antennas = np.array(antenna_data)
          #  else:
                # Load antenna data from a text file (assuming space-separated values)
          #      self.antennas = np.genfromtxt(antennas)

        #else:
            # If antennas is not a string, assume it's a NumPy array or list
          #  self.antennas = np.array(antennas)

        self.point_dir = point_dir
        self.degrees  = degrees
        self.lon = self.antennas[:,0]
        self.lat = self.antennas[:,1]
        self.alt = self.antennas[:,2]
    


    def geodetic2global(self,degrees=True):
       
        """
        Converts from the geodetic frame to the global frame.

        Parameters
        ---
        degrees: boolean
                : Specify whether the unit of the input geodetic coords is degrees.
                Default is True

        Output
        ---
        XYZ: ndarray
            : An array containing global coordinates XYZ
        """
        #lons and lats
        lon = self.lon
        lat = self.lat
        alt = self.alt
        ref_alt = self.array_center[2]
        
        #convert values to radians
        if degrees:
            lat = np.deg2rad(lat)
            lon = np.deg2rad(lon)
            ref_lat = np.deg2rad(self.array_center[1])
            ref_lon = np.deg2rad(self.array_center[0])

        # Constants defined by the World Geodetic System 1984 (WGS84)

        #Earth's semi major axis.
        a = 6378137. #[m]

        #Earth's first numerical eccentricity
        esq = 0.00669437999014
        
        #flattening of the ellipsoid
        f = 1 / 298.257223563

        Np = a/np.sqrt(1-esq*np.sin(lat)**2)
        N0 = a/np.sqrt(1-esq*np.sin(ref_lat**2))

        #calculating the global coordinates of the antennas.
        x = (Np+alt)*np.cos(lat)*np.cos(lon)
        y = (Np+alt)*np.cos(lat)*np.sin(lon)
        z = ((1-esq)*Np+alt)*np.sin(lat)

        #calculating the global coordinates of the array center.
        x0 = (N0+ref_alt)*np.cos(ref_lat)*np.cos(ref_lon)
        y0 = (N0+ref_alt)*np.cos(ref_lat)*np.sin(ref_lon)
        z0 = ((1-esq)*N0+ref_alt)*np.sin(ref_lat)

        xyz = np.column_stack((x,y,z))
        xyz0 = np.column_stack((x0,y0,z0))

        return xyz,xyz0
    
    def geodetic2local(self):
        """
        Converts the antenna positions from the geodetic frame
        to the local frame.


        Output
        ---
        ENU: ndarray
                : An array containing the local positions of the antennas.
        """

        #use global coordinates to convert to local
        antennas,ref = self.geodetic2global()
        x = antennas[:,0]
        y = antennas[:,1]
        z = antennas[:,2]
        x0,y0,z0 = ref[0,0],ref[0,1],ref[0,2]

        #calculate the vector from the origin to the antenna position
        delta_x = x-x0
        delta_y = y-y0
        delta_z = z-z0

        #lons and lats
        lon = self.lon
        lat = self.lat

        #local frame components.
        E = -np.sin(lon) * delta_x + np.cos(lon) * delta_y + 0 * delta_z
        N = -np.sin(lat)*np.cos(lon) * delta_x + -np.sin(lat)*np.sin(lon) * delta_y + np.cos(lat) * delta_z
        U =  np.cos(lat)*np.cos(lon) * delta_x + np.cos(lat)*np.sin(lon) * delta_y + np.sin(lat) * delta_z
        
        #arranging the components into an array.
        enu = np.column_stack((E,N,U))

        return enu
    


    def global2uvw(self, point_dir=None):
        """
        This function converts the antenna positions from the global frame to 
        the uvw space.

        Parameters
        ---
        point_dir: Union[float, str]
                    : Pointing direction. should be given as [RA,DEC] in degrees.
        
        Output
        ---
        uvw: ndarray
            : Antenna Positions in the uvw space.

        """
        #you can change the pointing direction
        if point_dir is None:
            point_dir = self.point_dir
        
        #convert geodetic to global to compute uvw
        antennas,ref = self.geodetic2global()
    
        #the transformation matrix
        TM = np.array([
            [np.sin(point_dir[0]), np.cos(point_dir[0]), 0],
            [-np.sin(point_dir[1])*np.cos(point_dir[0]), np.sin(point_dir[1])
             * np.sin(point_dir[0]), np.cos(point_dir[1])],
            [np.cos(point_dir[1])*np.cos(point_dir[0]), -np.cos(point_dir[1])
             * np.sin(point_dir[0]), np.sin(point_dir[1])]
        ])

        #calculating the baselines
        bl = self.baselines(antennas=antennas)

        u = np.outer(TM[0,0],bl[:,0]) + np.outer(TM[0,1],bl[:,1]) + np.outer(TM[0,2],bl[:,2])
        v = np.outer(TM[1,0],bl[:,0]) + np.outer(TM[1,1],bl[:,1]) + np.outer(TM[1,2],bl[:,2])
        w = np.outer(TM[2,0],bl[:,0]) + np.outer(TM[2,1],bl[:,1]) + np.outer(TM[2,2],bl[:,2])
        
        u, v, w = [ x.flatten() for x in (u, v, w) ]
        uvw = np.column_stack((u,v,w))

        return uvw

    def baselines(self, antennas=None):
        """
        This function calculates the baselines.

        Parameter
        ---
        antennas: ndarray
                : Antenna positions.

        Output
        ---
        baselines: ndarray

        """
        if antennas is None:
            antennas = self.antennas

        baselines = []
        for i in range(antennas.shape[0]):
            for j in range(i+1,antennas.shape[0]):
                baseline = antennas[j] - antennas[i]
                baselines.append(baseline)
        return np.array(baselines)