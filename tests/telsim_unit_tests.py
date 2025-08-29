from simms.telescope.generate_ms import create_ms
from daskms import xds_from_table
import numpy as np
from simms.telescope import array_utilities
from scipy.optimize import least_squares


def check_max_bl():
    
    create_ms(ms='test-mk.ms', 
              telescope_name='meerkat', 
              pointing_direction=['J2000','0deg','0deg'], 
              dtime=1,
              ntimes=10, 
              start_freq='1420MHz', 
              dfreq='1MHz', 
              nchan=2,
              correlations=['XX','YY'], 
              row_chunks=10000,
              sefd=425,
              column='DATA', 
              start_time='2025-03-06T12:25:00',
              smooth=None,
              fit_order=None)

    ds = xds_from_table('../tests/test-mk.ms')[0]
    uvw = ds.UVW.values
    bl = np.sqrt(uvw[:,0] ** 2 + uvw[:,1] ** 2)
    max_bl = bl.max()
    min_bl = bl.min()
    
    real_max_bl = 7700
    real_min_bl = 30
    
    assert np.isclose(max_bl, real_max_bl, atol=100)  
    assert np.isclose(min_bl, real_min_bl, atol=100)
    


def check_uv_coverage():
    """
    Test to check if the uv coverage shape is what is expected. 
    e.g circles at the poles and lines at the equator.
    
    """
    
    array = array_utilities.Array('kat-7')
    array_data_line = array.uvgen(pointing_direction=['J2000','0deg','0deg'],dtime=2,ntimes=5000,start_freq='1420MHz',
                  dfreq='2MHz',nchan=10)
    
    line = check_circle_or_ellipse(array_data_line.uvw[::21,0], array_data_line.uvw[::21,1])
    assert line['is_line'] == True and line['is_circle'] == False and line['is_ellipse'] == False
    
    array_data_circle = array.uvgen(pointing_direction=['J2000','0deg','-90deg'],dtime=2,ntimes=5000,start_freq='1420MHz',
                  dfreq='2MHz',nchan=10)
    
    circle = check_circle_or_ellipse(array_data_circle.uvw[::21,0], array_data_circle.uvw[::21,1])
    assert circle['is_circle'] == True and circle['is_ellipse'] == False and circle['is_line'] == False
    
    array_data_circle_n = array.uvgen(pointing_direction=['J2000','0deg','90deg'],dtime=2,ntimes=5000,start_freq='1420MHz',
                  dfreq='2MHz',nchan=10)
    
    circle_n = check_circle_or_ellipse(array_data_circle_n.uvw[::21,0], array_data_circle_n.uvw[::21,1])
    assert circle_n['is_circle'] == True and circle_n['is_ellipse'] == False and circle_n['is_line'] == False
    
    
    
    array_data_ellipse = array.uvgen(pointing_direction=['J2000','0deg','-30deg'],dtime=2,ntimes=5000,start_freq='1420MHz',
                  dfreq='2MHz',nchan=10)
    
    ellipse = check_circle_or_ellipse(array_data_ellipse.uvw[::21,0], array_data_ellipse.uvw[::21,1])
    assert ellipse['is_circle'] == False and ellipse['is_ellipse'] == True and ellipse['is_line'] == False
    
    
    
    
def check_circle_or_ellipse(u, v):
    """
    Check if UV points form a circle, ellipse or line.
    """
 
    center_u = np.mean(u)
    center_v = np.mean(v)
   
    distances = np.sqrt((u - center_u)**2 + (v - center_v)**2)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)  
    
    def conic_equation(params, x, y):
        a, b, c, d, e, f = params
        return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
    
    def residuals(params, x, y):
        return conic_equation(params, x, y)
    
    initial_guess = [1, 0, 1, 0, 0, -mean_distance**2]
    result = least_squares(residuals, initial_guess, args=(u, v))
    a, b, c, d, e, f = result.x

    #For ellipse: b^2 - 4ac < 0; for circle, a ≈ c and b ≈ 0
    discriminant = b**2 - 4 * a * c
    is_circle = abs(a - c) < 0.01 * abs(a) and abs(b) < 0.01 * abs(a)
    is_ellipse = discriminant < 0 and not is_circle
    
    coeffs = np.polyfit(u, v, 1)
    m, b = coeffs
    v_pred = m * u + b
    residual_v = v - v_pred
    residual_std = np.std(residual_v)
    ss_tot = np.sum((v - np.mean(v))**2)
    ss_res = np.sum(residual_v**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    u_range = np.max(u) - np.min(u) if len(u) > 1 else 1.0
    v_std = np.std(v)
    is_horizontal_line = abs(m) < 0.01 and v_std < 0.05 * u_range  
    is_line = (r_squared > 0.95 and residual_std < 0.05 * np.std(np.sqrt(u**2 + v**2))) or is_horizontal_line
    
    return {
        'is_circle': is_circle,
        'is_ellipse': is_ellipse,
        'is_line': is_line,
    }

def check_stupid_things():
    ds = xds_from_table('../tests/test-mk.ms')[0]    
    ds_spw = xds_from_table('../tests/test-mk.ms::SPECTRAL_WINDOW')[0]
    num_chan = ds_spw.NUM_CHAN.values
    assert np.isclose(num_chan, 2)  
    
    freq = ds_spw.CHAN_FREQ.values[0][0]
    assert np.isclose(freq, 1420e6)
    
    dfreq = ds_spw.CHAN_FREQ.values[0][1] - freq
    assert np.isclose(dfreq, 1e6)
    
    time = np.unique(ds.TIME).shape
    assert np.isclose(time, 10)
    
    dtime = ds.INTERVAL.values[0]
    assert np.isclose(dtime, 1)
    
    nbl = ds.DATA.shape[0] / time[0]
    assert np.isclose(nbl, 2016)

    ds_point = xds_from_table('../tests/test-mk.ms::POINTING')[0]
    direction = ds_point.TARGET.values[0][0]
    print(direction)

    assert np.isclose(direction[0], 0)
    assert np.isclose(direction[1], 0)
    
    ds_pol = xds_from_table('../tests/test-mk.ms::POLARIZATION')[0]
    corr = ds_pol.CORR_TYPE.values[0]
    assert np.isclose(corr[0], 9)
    assert np.isclose(corr[1], 12)
    
    ds_ant = xds_from_table('../tests/test-mk.ms::ANTENNA')[0]
    mount = ds_ant.MOUNT.values[0]
    size = ds_ant.DISH_DIAMETER.values[0]
    assert mount == "ALT-AZ"
    assert np.isclose(size, 13.5)

if __name__ == "__main__":
    check_max_bl()
    check_uv_coverage()
    check_stupid_things()
    