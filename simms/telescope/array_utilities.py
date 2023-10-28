import numpy as np
from casacore.measures import measures
import ephem
import utilities
import re

class Array(object):
    """
    The Array class has functions for converting from one coordinate system to another.
    """

    #observatories that can readily be accessed
    observatories = {
        'vla': '/home/mukundi/simms/simms/simms/telescope/layouts/vla.geodetic.yaml',
        'wsrt': '/home/mukundi/simms/simms/simms/telescope/layouts/WSRT.geodetic.yaml',
        'meerkat': '/home/mukundi/simms/simms/simms/telescope/layouts/meerkat.geodetic.yaml',
        'kat-7': '/home/mukundi/simms/simms/simms/telescope/layouts/kat-7.geodetic.yaml'
    }

  
    def __init__(self, data_file=None, 
                 observatory=None, 
                 degrees=True):

        """
        
        data_file: yaml file.
                    : yaml file containing antenna positions in geodetic frame and
                    array center.
        observatory: str
                    : specify an observatory instead of giving the data_file
        degrees: boolean
                    : Specify whether the long-lat is in degrees or not. Default is true.


        """
        
        self.datafile = data_file
        self.observatory = observatory
        self.degrees = degrees

    def get_arrayinfo(self):
        """
            Extract the array information from the schema
        """
        #check if the provided array is one of the default arrays.
        if self.observatory is not None:
            if self.observatory in self.observatories:
                array_layout = self.observatories[self.observatory]
                array_info = utilities.readyaml(array_layout)
                antlocations = array_info['antlocations']
            elif self.observatory.startswith('vla-'):
                array_layout = self.observatories['vla']
                array_info = utilities.readyaml(array_layout)
                antlocations = array_info['antlocations'][self.observatory]
               
        
        #if antennas positions file is given as input as a yaml file, read the file
        elif self.observatory is None and self.data_file is not None:
            array_info = utilities.readyaml(self.data_file)
            antlocations = array_info['antlocations']
                
        #if both cases fail, raise an error.
        else:
            raise ValueError('Either name of the array or antenna positions and center should be provided.')


        #Other attributes
        #antnames = array_info['antnames']
        centre = array_info['centre']
       # mount = array_info['mount']
        #coord_sys = array_info['coord_sys']
    

        longitude, latitude, altitude = zip(*antlocations)
        
        ref_longitude = centre[0][0]
        ref_latitude  = centre[0][1]
        ref_altitude = centre[0][2]

        if self.degrees:
            longitude = np.deg2rad(longitude)
            latitude = np.deg2rad(latitude)
            ref_longitude = np.deg2rad(ref_longitude)
            ref_latitude = np.deg2rad(ref_latitude)
    
        return longitude, latitude, altitude, [ref_longitude, ref_latitude, ref_altitude]


    def geodetic2global(self):
        """
        Convert the antenna positions from the geodetic frame to the global frame

        Returns
        ----
        An array of the antennas XYZ positions and an array of the array center in XYZ
        """
            
        #Earth's semi major axis.
        a = 6378137. #[m]

        #Earth's first numerical eccentricity
        esq = 0.00669437999014
        
        #flattening of the ellipsoid
        f = 1 / 298.257223563
        longitude,latitude,altitude, centre = self.get_arrayinfo()
        ref_longitude,ref_latitude,ref_altitude = centre

        Np = a/np.sqrt(1-esq*np.sin(latitude)**2)
        N0 = a/np.sqrt(1-esq*np.sin(ref_latitude**2))

        #calculating the global coordinates of the antennas.
        x = (Np+altitude)*np.cos(latitude)*np.cos(longitude)
        y = (Np+altitude)*np.cos(latitude)*np.sin(longitude)
        z = ((1-esq)*Np+altitude)*np.sin(latitude)

        #calculating the global coordinates of the array center.
        x0 = (N0+ref_altitude)*np.cos(ref_latitude)*np.cos(ref_longitude)
        y0 = (N0+ref_altitude)*np.cos(ref_latitude)*np.sin(ref_longitude)
        z0 = ((1-esq)*N0+ref_altitude)*np.sin(ref_latitude)

        xyz = np.column_stack((x,y,z))
        xyz0 = np.column_stack((x0,y0,z0))

        return xyz,xyz0
        

    def geodetic2local(self):    
        """
        Converts the antenna positions from the geodetic frame to the local frame

        Returns
        ---
        An array of the antenna positions in the local frame ENU
        """  
        longitude,latitude,_,_ = self.get_arrayinfo()
        
        #we need the positions in xyz
        xyz,xyz0 = self.geodetic2global()
        x,y,z = zip(*xyz)
        x0,y0,z0 = xyz0[0,0],xyz0[0,1],xyz0[0,2]

        #calculate the vector from the origin to the antenna position
        delta_x = x-x0
        delta_y = y-y0
        delta_z = z-z0

        #local frame components.
        E = -np.sin(longitude) * delta_x + np.cos(longitude) * delta_y + 0 * delta_z
        N = -np.sin(latitude)*np.cos(longitude) * delta_x + -np.sin(latitude)*np.sin(longitude) * delta_y + np.cos(latitude) * delta_z
        U =  np.cos(latitude)*np.cos(longitude) * delta_x + np.cos(latitude)*np.sin(longitude) * delta_y + np.sin(latitude) * delta_z
        
        #arranging the components into an array.
        enu = np.column_stack((E,N,U))

        return enu
    


    def global2uvw(self, h0,longitude,latitude,
                   pointing_direction=None,
                   date = None,
                   dtime = None,
                   ntimes = None,
                    start_freq = None,
                     dfreq = None,
                      nchan = None
                       ):

        """
        Converts the antenna positions from the global frame to the uvw space

        Parameters
        ---

        h0: Union[float,int]
                    : hour angle range
        longitude: float
                    : longitude of the observer
        latitude: float
                    : latitude of the observer
        pointing_direction: List[str]
                    : pointing direction. Default is ['J200','0deg','-30deg']
        date: str
                    : starting time of the observation. Default is 2023/10/25 12:0:0
        dtime: int
                    : integration time. Default is 10
        ntimes: int
                    : number of times the sky should be snapped.
                    Default is 10
        start_freq: Union[str,float]
                    : starting freq of the observation. Defualt is 900MHz
        dfreq: Union[str,float]
                    : frequency interval. Default is 2MHz
        nchan: int
                    : number of channels. Default is 10

        Returns
        ---
        An array of the uvw time dependent positions, the time array and the frequency array
        """
        #casacore direction and epoch measures
        dm = measures()

        #xyz coordinates of the array
        positions_global,_ = self.geodetic2global()

        #handle the pointing direction
        if pointing_direction is None:
            pointing_direction =  ['J2000','0deg','-30deg']
        else:
            pointing_direction = pointing_direction

        #convert the direction using measures
        ra_dec = dm.direction(*pointing_direction)
        ra = ra_dec['m0']['value']
        dec = ra_dec['m1']['value']
        
        #handle the date/starting time
        if date is None:
            date = ['UTC','2023/10/25 12:0:0']
        else: 
            date = date
        
        #handle the ntimes
        if ntimes is None:
            ntimes = 10
        else:
            ntimes = ntimes
        
        #handle the dtime
        if dtime is None:
            dtime = 10.0
        else:
            dtime = dtime
        
        #hour angle 
        ih0,date_read,H0,altitude = self.source_info(longitude,latitude,date[1])
        
        h0 = ih0 + np.linspace(h0[0], h0[1], ntimes)*np.pi/12.
        

        #the transformation matrix
        transform_matrix = np.array([
            [np.sin(h0), np.cos(h0), 0],
            [-np.sin(dec)*np.cos(h0), np.sin(dec)
             * np.sin(h0), np.cos(dec)],
            [np.cos(dec)*np.cos(h0), -np.cos(dec)
             * np.sin(h0), np.sin(dec)]
        ])

        #calculating the baselines
        bl = self.baselines(antennas=positions_global)

        u = np.outer(transform_matrix[0,0],bl[:,0]) + np.outer(transform_matrix[0,1],bl[:,1]) + np.outer(transform_matrix[0,2],bl[:,2])
        v = np.outer(transform_matrix[1,0],bl[:,0]) + np.outer(transform_matrix[1,1],bl[:,1]) + np.outer(transform_matrix[1,2],bl[:,2])
        w = np.outer(transform_matrix[2,0],bl[:,0]) + np.outer(transform_matrix[2,1],bl[:,1]) + np.outer(transform_matrix[2,2],bl[:,2])
        
        u, v, w = [ x.flatten() for x in (u, v, w) ]
        uvw = np.column_stack((u,v,w))

        #starting time of the observation in seconds(sunce 1970) 
        start_time_rad = dm.epoch(*date)['m0']['value']
        start_time_rad = start_time_rad * 24 * 3600

        #total time of observation
        total_time = start_time_rad + ntimes*dtime

        #the time table
        time_entries = np.arange(start_time_rad,total_time,dtime)

        #process the starting frequency value 
        if start_freq is None:
            start_freq =  900.0e6
        else:
            start_freq = start_freq
        
        start_freq = self.process_frequency(start_freq)

        #process the dfreq 
        if dfreq is None:
            dfreq =  2.0e6
        else: 
            dfreq = dfreq

        dfreq = self.process_frequency(dfreq)

        #number of channels
        if nchan is None:
            nchan =  10
        else:
            nchan = nchan

        total_bandwidth = start_freq + dfreq * nchan

        frequency_entries = np.arange(start_freq,total_bandwidth,dfreq)
       

        return uvw,time_entries,frequency_entries


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
    


    #stolen from https://github.com/SpheMakh/uvgen/blob/master/uvgen.py
    def source_info(self,longitude,latitude,
                    pointing_direction = None,
                    date=None):
        
        dm = measures()
        longitude = np.rad2deg(longitude)
        latitude = np.deg2rad(latitude)
       
        # Set up observer        
        obs = ephem.Observer()
        obs.lon, obs.lat = longitude,latitude

        if pointing_direction is None:
            pointing_direction =  ['J2000','0deg','-30deg']
        else:
            pointing_direction = pointing_direction

        ra_dec = dm.direction(*['J2000','0deg','-30deg'])
        ra = ra_dec['m0']['value']
        dec = ra_dec['m1']['value']

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
        

        obs.date = date or "2023/10/25 12:0:0"#%(time.localtime()[:3])
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



    def process_frequency(self,frequency):
        """
        Process the frequency value.

        Parameters
        ---
        
        frequency: Union[float,str]

        Returns
        ---
        Returns the numeric part of the frequency 
        """
        #if given freuquency as a string
        if isinstance(frequency, str):
            # extract the numeric part of the frequency.
            
            #use regular expression
            match = re.search(r'(\d+(\.\d*)?)', frequency)
            if match:
                numeric_part = match.group(1)
                
                return float(numeric_part)*1e6
            else:
                # Handle the case when the string doesn't contain a valid numeric part.
                raise ValueError("Invalid frequency string format")
            
        #if frequency given as either a float or integer
        elif isinstance(frequency,(float,int)):
             return frequency
        
        else:
            # Handle other data types or unexpected inputs.
            raise ValueError("Invalid input type")



obj = Array(observatory='meerkat')
glo = obj.geodetic2global()

uvw = obj.global2uvw(h0=[-1,1],longitude = -107.0,latitude=34.5)

