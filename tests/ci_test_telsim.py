from simms.telescope.generate_ms import create_ms
from daskms import xds_from_table
import numpy as np

def check_max_bl():
    
    create_ms(ms='test-mk.ms', 
              telescope_name='meerkat', 
              pointing_direction=['J2000','0deg','0deg'], 
              dtime=1,
              ntimes=10, 
              start_freq='1420MHz', 
              dfreq='1MHz', 
              nchan=4,
              correlations=['XX','XY'], 
              row_chunks=10000,
              sefd=425,
              column='DATA', 
              start_time='2025-03-06T12:25:00')
    
    ds = xds_from_table('../tests/test-mk.ms')[0]
    uvw = ds.UVW.values
    bl = np.sqrt(uvw[:,0] ** 2 + uvw[:,1] ** 2)
    max_bl = bl.max()
    min_bl = bl.min()
    
    real_max_bl = 7700
    real_min_bl = 30
    
    assert np.isclose(max_bl, real_max_bl, atol=100)  
    assert np.isclose(min_bl, real_min_bl, atol=100)
    
if __name__ == "__main__":
    check_max_bl()