from typing import Dict, List, Union

import dask
import xarray as xr
import dask.array as da
import numpy as np
from casacore.tables import table
from daskms import Dataset, xds_to_table
from scabha.basetypes import File


from simms.telescope import array_utilities as autils
from casacore.measures import measures
dm = measures()


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

    #Number of frequency channels 
    num_chans = len(uvcoverage_data.freqs)

    # Number of correlations, assuming all four correlations here
    num_corr = 4

    #Indices of the pair of antennas forming a baseline. These baselines
    #are used ntimes during the observation
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes
    num_spws = 1.

    ddid = da.zeros(num_rows, chunks=num_row_chunks)
    data = da.zeros((num_rows, num_chans, num_corr),chunks=(num_row_chunks, num_chans, num_corr))
    times = da.from_array(uvcoverage_data.times,chunks=num_row_chunks)
    uvw = da.from_array(uvcoverage_data.uvw,chunks=(num_row_chunks,3))
    antenna1 = da.from_array(ant1,chunks=num_row_chunks)
    antenna2 = da.from_array(ant2,chunks=num_row_chunks)
    interval = da.full(num_rows,dtime)
    interval = da.rechunk(interval,chunks=num_row_chunks)
    sigma = da.full((num_rows,num_corr),1.)
    sigma = da.rechunk(sigma,chunks=(num_row_chunks,num_corr))

    freqs = da.from_array(uvcoverage_data.freqs,chunks=num_chans)
    ref_freq = dm.frequency(v0=start_freq)["m0"]["value"]
    ref_freq = da.from_array(ref_freq, chunks=1).compute()
    # ref_freq = da.asarray(ref_freq).compute()
    chan_width = dm.frequency(v0=dfreq)["m0"]["value"]
    chan_width = da.from_array(np.asarray(chan_width), chunks=1).compute()

    main_table = xr.Dataset(
    {
        'DATA_DESC_ID': (("row",), ddid),
        'DATA': (("row", "chan", "corr"), data),
        'CORRECTED_DATA': (("row", "chan", "corr"), data),
        'MODEL_DATA': (("row", "chan", "corr"), data),
        "UVW": (("row", "uvw_dim"), uvw),
        "TIME": (("row"),times),
        "TIME_CENTROID": (("row"),times),
        "INTERVAL":(("row"),interval),
        "EXPOSURE":(("row"),interval),
        "ANTENNA1":(("row"), antenna1),
        "ANTENNA2":(("row"), antenna2),
        "SIGMA":(("row","corr"),sigma),
        "WEIGHT":(("row","corr"),sigma),
        "SIGMA_SPECTRUM":(("row","corr"),sigma),
        "WEIGHT_SPECTRUM":(("row","corr"),sigma),
    },
    coords={"row": da.arange(num_rows)}
    )

    

    # spectral_window_table = xr.Dataset(
    # {   
    #     "CHAN_FREQ": (("chan"), freqs),
    #     # "REF_FREQUENCY": (("chan"), ref_freq),
    #     # "CHAN_WIDTH":(("chan"),chan_width),
    #     # "RESOLUTION":(("chan"),chan_width),
    #     # "EFFECTIVE_BW":(("chan"),chan_width),
    # },
    # coords={"chan": da.arange(num_chans)}
    # )


    write_main = xds_to_table([main_table], f"{ms_name}.ms")
    # write_spw = xds_to_table([spectral_window_table],f"{ms_name}.ms::SPECTRAL_WINDOW")
    dask.compute(write_main)
    # dask.compute(write_spw)





    # freqs = da.from_array(uvcoverage_data.freqs,chunks=num_chans)
    # ref_freq = dm.frequency(v0=start_freq)["m0"]["value"]
    # ref_freq = da.from_array(ref_freq, chunks=1).compute()
    # print(f'ref-freq:{ref_freq}')
    # # ref_freq = da.asarray(ref_freq).compute()
    # chan_width = dm.frequency(v0=dfreq)["m0"]["value"]
    # chan_width = da.from_array(np.asarray(chan_width), chunks=1).compute()
    # chan_width = da.asarray(chan_width,dtype=float).compute()

    # spectral_window_table = xr.Dataset(
    # {
    #     # "CHAN_FREQ": (("chan"), freqs),
    #     # "REF_FREQUENCY": (("chan"), ref_freq),
    #     # "CHAN_WIDTH":(("chan"),chan_width),
    #     # "RESOLUTION":(("chan"),chan_width),
    #     # "EFFECTIVE_BW":(("chan"),chan_width),
    # },
    # coords={"chan": np.arange(num_chans)}
    # )

    # write_spw = xds_to_table([spectral_window_table], f"{ms_name}.ms::SPECTRAL_WINDOW",["CHAN_FREQ"])
    # dask.compute(write_spw)






    # #Data description id
    # ddid = da.zeros(num_rows, chunks=num_row_chunks)

    # #Data shape, used in tandem with the ddid to define the shape of the data in the ms
    # data = da.zeros((num_rows, num_chans, num_corr),
    #                 chunks=(num_row_chunks, num_chans, num_corr))

    # data_vars = {'DATA_DESC_ID': (("row",), ddid),
    #              'DATA': (("row", "chan", "corr"), data)}
    
    # #Create the MS with all of the columns
    # writes = xds_to_table([Dataset(data_vars)], f"{ms_name}.ms", "ALL")
    # dask.compute(writes)
   
    # #Open the created MS file main table
    # main_table = table(f"{ms_name}.ms", readonly=False, lockoptions="auto")

    # #Update the columns
    # main_table.putcol("UVW", uvcoverage_data.uvw)
    # main_table.putcol("ANTENNA1", ant1)
    # main_table.putcol("ANTENNA2", ant2)
    # main_table.putcol("TIME", uvcoverage_data.times)
    # main_table.putcol("INTERVAL", np.full(num_rows,dtime))
    # main_table.putcol("EXPOSURE", np.full(num_rows,dtime))
    # main_table.putcol("TIME_CENTROID", uvcoverage_data.times)
    # main_table.putcol("SIGMA", np.full((num_rows,num_corr),1.))
    # main_table.putcol("WEIGHT", np.full((num_rows,num_corr),1.))

    # main_table.close()
    
    # #Open the Spectral window table of the MS
    # spw_table = table(f"{ms_name}.ms::SPECTRAL_WINDOW",
    #                   readonly=False, lockoptions="auto")
    
    # start_freq = dm.frequency(v0=start_freq)["m0"]["value"]

    # # channel bandwidth of the observation
    # dfreq = dm.frequency(v0=dfreq)["m0"]["value"]
    # print(f"Chan_freq:{spw_table.getcol('CHAN_FREQ').shape}")

    # #Update the columns
    # spw_table.putcol("CHAN_FREQ", uvcoverage_data.freqs)
    # spw_table.putcol("REF_FREQUENCY", np.array([start_freq],dtype=float))
    # spw_table.putcol("NUM_CHAN", np.array([nchan], dtype=int))
    # spw_table.putcol("CHAN_WIDTH", np.full(nchan,dfreq,dtype=float))
    # spw_table.putcol("EFFECTIVE_BW", np.full(nchan,dfreq,dtype=float))
    # spw_table.putcol("RESOLUTION", np.full(nchan,dfreq,dtype=float))
    # total_bandwidth = nchan * dfreq
    # spw_table.putcol("TOTAL_BANDWIDTH",np.array(total_bandwidth,dtype=float))
    


    # #Close the table when done
    # spw_table.close()
   

    # # return 0
