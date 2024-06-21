from typing import Dict, List, Union

import dask
import dask.array as da
import numpy as np
from casacore.tables import table
from daskms import Dataset, xds_to_table
from scabha.basetypes import File

from simms.telescope import array_utilities as autils


def create_ms(ms_name: str, telescope_name: Union[str, File],
              pointing_direction: List[str], dtime: int, ntimes: int,
              start_freq: Union[str, float], dfreq: Union[str, float],
              nchan: int, start_time: Union[str, List[str]] = None,
              start_ha: float = None, horizon_limit: Union[float, str] = None):
    
    "Creates an empty Measurement Set for an observation using given observation parameters"

    #Obtain the array information using the specified array name or file.
    telescope_array = autils.Array(telescope_name)

    #Generate the uv coverage of the observation along with the TIME and CHAN_FREQ columns.
    uvcoverage_data = telescope_array.uvgen(pointing_direction, dtime, ntimes,
                                            start_freq, dfreq, nchan, start_time,
                                            start_ha)

    #The number of rows of the measurementset main table, given by ntimes x nbaselines
    num_rows = uvcoverage_data.times.shape[0]
    #Defines number of chunks to divide the data into, larger number of
    #chunks improves the computation speed. Should find a way to specifiy it better
    num_row_chunks = num_rows // 4


    #Number of channels being used for the observation
    num_chans = len(uvcoverage_data.freqs)

    # Number of correlations, assuming all four correlations here
    num_corr = 4

    #Indices of the pair of antennas forming a baseline. These baselines
    #are used ntimes during the observation
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes

    #Data description id
    ddid = da.zeros(num_rows, chunks=num_row_chunks)

    #Data shape, used in tandem with the ddid to define the shape of the data in the ms
    data = da.zeros((num_rows, num_chans, num_corr),
                    chunks=(num_row_chunks, num_chans, num_corr))

    data_vars = {'DATA_DESC_ID': (("row",), ddid),
                 'DATA': (("row", "chan", "corr"), data)}
    
    #Create the MS with all of the columns
    writes = xds_to_table([Dataset(data_vars)], f"{ms_name}.ms", "ALL")
    dask.compute(writes)
   
    #Open the created MS file main table
    main_table = table(f"{ms_name}.ms", readonly=False, lockoptions="auto")

    #Update the columns
    main_table.putcol("UVW", uvcoverage_data.uvw)
    main_table.putcol("ANTENNA1", ant1)
    main_table.putcol("ANTENNA2", ant2)
    main_table.putcol("TIME", uvcoverage_data.times)
    main_table.putcol("INTERVAL", np.full(num_rows,dtime))
    main_table.putcol("EXPOSURE", np.full(num_rows,dtime))
    main_table.putcol("TIME_CENTROID", uvcoverage_data.times)
    main_table.putcol("SIGMA", np.full((num_rows,num_corr),1.))
    main_table.putcol("WEIGHT", np.full((num_rows,num_corr),1.))
    
    #Open the Spectral window table of the MS
    spw_table = table(f"{ms_name}.ms::SPECTRAL_WINDOW",
                      readonly=False, lockoptions="auto")
    

    #Update the columns
    spw_table.putcol("CHAN_FREQ", uvcoverage_data.freqs)
    

    print(spw_table.getcol('CHAN_FREQ'))

    # Close the table when done
    spw_table.close()
    main_table.close()

    # return 0
