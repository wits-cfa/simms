import copy
from typing import Dict, List, Union

import dask
import dask.array as da
import daskms
import numpy as np
import xarray as xr
from casacore.measures import measures
from casacore.tables import table
from daskms import Dataset, xds_from_table, xds_to_table
from scabha.basetypes import File

from simms.telescope import array_utilities as autils

dm = measures()


def create_ms(ms_name: str, telescope_name: Union[str, File],
              pointing_direction: List[str], dtime: int, ntimes: int,
              start_freq: Union[str, float], dfreq: Union[str, float],
              nchan: int, correlations: List[str], start_time: Union[str, List[str]] = None,
              start_ha: float = None, horizon_limit: Union[float, str] = None):
    "Creates an empty Measurement Set for an observation using given observation parameters"

    # Obtain the array information using the specified array name or file.
    telescope_array = autils.Array(telescope_name)
    telescope_array.set_arrayinfo()
    # size = telescope_array.size
    mount = telescope_array.mount
    antlocation, _ = telescope_array.geodetic2global()

    # Generate the uv coverage of the observation along with the TIME and CHAN_FREQ columns.
    uvcoverage_data = telescope_array.uvgen(pointing_direction, dtime, ntimes,
                                            start_freq, dfreq, nchan, start_time,
                                            start_ha)

    # The number of rows of the measurementset main table, given by ntimes x nbaselines
    num_rows = uvcoverage_data.times.shape[0]
    # Defines number of chunks to divide the data into, larger number of
    # chunks improves the computation speed. Should find a way to specifiy it better
    num_row_chunks = num_rows // 1

    # Number of frequency channels
    num_chans = len(uvcoverage_data.freqs)

    # Number of correlations, assuming all four correlations here
    num_corr = len(correlations)

    # Indices of the pair of antennas forming a baseline. These baselines
    # are used ntimes during the observation
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes
    num_spws = 1.

    ddid = da.zeros(num_rows, chunks=num_row_chunks)
    data = da.zeros((num_rows, num_chans, num_corr),
                    chunks=(num_row_chunks, num_chans, num_corr))
    times = da.from_array(uvcoverage_data.times, chunks=num_row_chunks)
    uvw = da.from_array(uvcoverage_data.uvw, chunks=(num_row_chunks, 3))
    antenna1 = da.from_array(ant1, chunks=num_row_chunks)
    antenna2 = da.from_array(ant2, chunks=num_row_chunks)
    interval = da.full(num_rows, dtime)
    interval = da.rechunk(interval, chunks=num_row_chunks)
    sigma = da.full((num_rows, num_corr), 1.)
    sigma = da.rechunk(sigma, chunks=(num_row_chunks, num_corr))

    freqs = da.from_array([uvcoverage_data.freqs], chunks=(1, num_chans))
    ref_freq = dm.frequency(v0=start_freq)["m0"]["value"]
    ref_freq = da.from_array(ref_freq, chunks=1).compute()
    # ref_freq = da.asarray(ref_freq).compute()
    chan_width = dm.frequency(v0=dfreq)["m0"]["value"]
    chan_width = da.from_array(np.asarray(chan_width), chunks=1).compute()
    n_chans = da.from_array(num_chans)

    main_table = xr.Dataset(
        {
            'DATA_DESC_ID': (("row",), ddid),
            'DATA': (("row", "chan", "corr"), data),
            'CORRECTED_DATA': (("row", "chan", "corr"), data),
            'MODEL_DATA': (("row", "chan", "corr"), data),
            "UVW": (("row", "uvw_dim"), uvw),
            "TIME": (("row"), times),
            "TIME_CENTROID": (("row"), times),
            "INTERVAL": (("row"), interval),
            "EXPOSURE": (("row"), interval),
            "ANTENNA1": (("row"), antenna1),
            "ANTENNA2": (("row"), antenna2),
            "SIGMA": (("row", "corr"), sigma),
            "WEIGHT": (("row", "corr"), sigma),
            "SIGMA_SPECTRUM": (("row", "corr"), sigma),
            "WEIGHT_SPECTRUM": (("row", "corr"), sigma),
        },
        coords={"ROWID": ("row", da.arange(num_rows))}
    )

    write_main = xds_to_table([main_table], f"{ms_name}.ms")

    dask.compute(write_main)

    # spw_tab = table(f"{ms_name}.ms::SPECTRAL_WINDOW",
    #                 readonly=False, lockoptions='auto')
    # # spw_tab.addrows(1)
    # spw_tab.putcol("CHAN_FREQ", np.array([freqs]), startrow=0, nrow=-1)
    # # spw_tab.close()
