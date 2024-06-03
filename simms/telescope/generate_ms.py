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

    telescope_array = autils.Array(telescope_name)

    uvcoverage_data = telescope_array.uvgen(pointing_direction, dtime, ntimes,
                                            start_freq, dfreq, nchan, start_time,
                                            start_ha)

    print(f"Freqs:: {uvcoverage_data.freqs}")

    num_rows = len(uvcoverage_data.times[0])
    num_row_chunks = num_rows / 4

    num_chans = len(uvcoverage_data.freqs)

    ddid = da.zeros(num_rows, chunks=num_row_chunks)

    data = da.zeros((num_rows, num_chans, 4),
                    chunks=(num_row_chunks, num_chans, 4))

    data_vars = {'DATA_DESC_ID': (("row",), ddid),
                 'DATA': (("row", "chan", "corr"), data)}

    writes = xds_to_table([Dataset(data_vars)], f"{ms_name}.ms", "ALL")
    dask.compute(writes)

    ms_table = table(f"{ms_name}.ms", readonly=False, lockoptions="auto")

    ms_table.putcol("TIME", uvcoverage_data.times[0], nrow=num_rows)
    ms_table.putcol("UVW", uvcoverage_data.uvw, nrow=num_rows)
    ms_table.close()

    spw_table = table(f"{ms_name}.ms::SPECTRAL_WINDOW",
                      readonly=False, lockoptions="auto")
    # spw_table.putcol("CHAN_FREQ", uvcoverage_data.freqs)

    print("Column names:", spw_table.colnames())
    print("Initial CHAN_FREQ:", spw_table.getcol("CHAN_FREQ"))

    print("Shape of freqs:", uvcoverage_data.freqs.shape)

    # Insert the data into the CHAN_FREQ column
    spw_table.putcol("CHAN_FREQ", uvcoverage_data.freqs[0:5], nrow=-1)
    print("Updated CHAN_FREQ:", spw_table.getcol("CHAN_FREQ"))

    # Close the table when done
    spw_table.close()

    # return 0
