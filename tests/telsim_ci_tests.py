import numpy as np
import astropy
from simms.telescope.generate_ms import create_ms
import subprocess
from astropy.io import fits
from simms.constants import C
from scipy.optimize import curve_fit
import os


def check_noise_image():
    
    """
    Test if the noise is what is expected in the image.
    """
    
    # create_ms(ms='test_telsim_ci.ms',
    #           telescope_name='meerkat',
    #           pointing_direction=['J2000','0deg','0deg'], 
    #           dtime=1,
    #           ntimes=5, 
    #           start_freq='1420MHz', 
    #           dfreq='1MHz', 
    #           nchan=2,
    #           correlations=['XX','XY'], 
    #           sefd=425,
    #           column='DATA', 
    #           start_time='2025-03-06T12:25:00',
    #           )
    
    mspath = '../tests/test-mk.ms'
    nchan = 2
    df = 1e6
    dt =1
    nt = 10
    nbaselines =2016
    sefd = 425
    ncorr = 2
    noise = sefd/(np.sqrt((nbaselines*ncorr*df*nchan*dt*nt)))
    
    output_filename = 'test-mk-pfb-result'
    log_directory = 'pfb-imaging/logs'
    output_folder = 'pfb-imaging'
   
    
    fov  =  1.22 * (C /1420e6)/(13.5)
    fov  = fov * 180/np.pi
    
    
    subprocess.run([
       "pfb","init",
       "--output-filename", output_filename,
       "--product", "I",
       "--ms", mspath,
       "--log-directory", log_directory,
       "--fits-output-folder",output_folder,
      #  "--freq-range", '1e9:2e9',
       "--data-column", "DATA",
       "--flag-column", 'FLAG',
       "--overwrite",
       "--channels-per-image", "1",
       "--no-fits-cubes",
       "--fits-mfs",
       "--scans","[1]",
       "--ddids","[0]",
       "--fields","[0]",
       "--freq-range","1.4e9:1.44e9",
       "--max-field-of-view", str(fov),  
        ])
    
    subprocess.run([
    "pfb","grid",
    "--output-filename", output_filename,  
    "--fits-output-folder",output_folder,
    "--log-directory", log_directory,
    "--field-of-view" , str(fov),
    "--no-do-wgridding",
    "--fits-mfs" ,
    "--no-fits-cubes",
    "--overwrite",
    "--nthreads" ," 1",
    "--super-resolution-factor", "0",
    "--cell-size","3",
    "--nx", "256",
    "--ny", "256",
   "--product" , "I",
   "--filter-counts-level", "0"
    ])
    
    base_dir = os.path.dirname(__file__) 
    file_path = os.path.abspath(os.path.join(base_dir, 'pfb-imaging', f'{output_filename}_I_main_dirty_time0000_mfs.fits'))
    hdu = fits.open(file_path)
   
    data = hdu[0].data[0,0,:,:]
    img_noise = np.std(data)  * np.sqrt(2) #In pfb-imaging, they did not account for the account of 2
    
    assert np.isclose(noise,img_noise, atol=1e-3)
    
if __name__ == "__main__":
    check_noise_image()
   
    