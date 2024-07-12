import os
import shutil
from typing import List, Union

import dask
import dask.array as da
import daskms
import numpy as np
from casacore.measures import measures
from casacore.tables import table
from daskms import xds_to_table
from scabha.basetypes import File

from simms.telescope import array_utilities as autils

dm = measures()


def remove_ms(ms_name: str):
    path = os.path.abspath(f"{ms_name}.ms")
    name = os.path.basename(path)
    if os.path.exists(name):
        shutil.rmtree(name, ignore_errors=True)
        print(
            f"The existing MS file {name} was successfully deleted. It is now being recreated...")
    else:
        print(
            f"MS file {name} does not exist. A new file will be created.")


def create_ms(ms_name: str, telescope_name: Union[str, File],
            pointing_direction: str, dtime: int, ntimes: int,
            start_freq: Union[str, float], dfreq: Union[str, float],
            nchan: int, correlations: List[str], row_chunks: int,
            addnoise: bool, sefd: float, column: str,
            start_time: Union[str, List[str]] = None,
            start_ha: float = None, horizon_limit: Union[float, str] = None,
            ):
    "Creates an empty Measurement Set for an observation using given observation parameters"

    remove_ms(ms_name)
    telescope_array = autils.Array(telescope_name)
    telescope_array.set_arrayinfo()
    size = telescope_array.size
    mount = telescope_array.mount
    antlocation, _ = telescope_array.geodetic2global()

    uvcoverage_data = telescope_array.uvgen(pointing_direction, dtime, ntimes,
                                            start_freq, dfreq, nchan, start_time,
                                            start_ha)

    num_rows = uvcoverage_data.times.shape[0]
    num_row_chunks = row_chunks

    num_chans = len(uvcoverage_data.freqs)
    num_corr = len(correlations)

    num_ants = antlocation.shape[0]
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes

    ra_dec = dm.direction(*pointing_direction)
    ra = ra_dec["m0"]["value"]
    dec = ra_dec["m1"]["value"]
    phase_dir = np.array([[[ra, dec]]])

    ddid = da.zeros(num_rows, chunks=num_row_chunks)
    data = da.zeros((num_rows, num_chans, num_corr),
                    chunks=(num_row_chunks, num_chans, num_corr))
    times = da.from_array(uvcoverage_data.times, chunks=num_row_chunks)
    time_range = np.array(
        [[uvcoverage_data.times[0], uvcoverage_data.times[-1]]])
    uvw = da.from_array(uvcoverage_data.uvw, chunks=(num_row_chunks, 3))
    antenna1 = da.from_array(ant1, chunks=num_row_chunks)
    antenna2 = da.from_array(ant2, chunks=num_row_chunks)
    interval = da.full(num_rows, dtime)
    interval = da.rechunk(interval, chunks=num_row_chunks)
    sigma = da.full((num_rows, num_corr), 1.)
    sigma = da.rechunk(sigma, chunks=(num_row_chunks, num_corr))
    flag = da.zeros((num_rows, num_chans, num_corr),
                    dtype=bool, chunks=(num_row_chunks, num_chans, num_corr))

    freqs = uvcoverage_data.freqs
    freqs = freqs.reshape(1, freqs.shape[0])
    ref_freq = dm.frequency(v0=start_freq)["m0"]["value"]
    chan_width = dm.frequency(v0=dfreq)["m0"]["value"]
    channel_widths = np.full(freqs.shape, chan_width)
    total_bandwidth = nchan * chan_width

    noise = sefd / np.sqrt(abs(2*chan_width*dtime))
    dummy_data = np.random.randn(num_rows, num_chans, num_corr) + \
        1j*np.random.randn(num_rows, num_chans, num_corr)
    noisy_data = da.array(dummy_data * noise)

    ds = {
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
        "FLAG": (("row", "chan", "corr"), flag)
    }

    if addnoise:
        ds[column] = (("row", "chan", "corr"), noisy_data)

    main_table = daskms.Dataset(
        ds, coords={"ROWID": ("row", da.arange(num_rows))})

    write_main = xds_to_table(main_table, f"{ms_name}.ms")
    dask.compute(write_main)

    spw_tab = table(f"{ms_name}.ms::SPECTRAL_WINDOW",
                    readonly=False, lockoptions='user', ack=False)

    try:
        spw_tab.lock(write=True)
        spw_tab.addrows(1)
        spw_tab.putcol("CHAN_FREQ", freqs)
        spw_tab.putcol("CHAN_WIDTH", channel_widths)
        spw_tab.putcol("EFFECTIVE_BW", channel_widths)
        spw_tab.putcol("RESOLUTION", channel_widths)
        spw_tab.putcol("REF_FREQUENCY", ref_freq)
        spw_tab.putcol("MEAS_FREQ_REF", ref_freq)
        spw_tab.putcol("TOTAL_BANDWIDTH", total_bandwidth)
        spw_tab.putcol("NUM_CHAN", nchan)
        spw_tab.putcol("NAME", "00")
        spw_tab.putcol("NET_SIDEBAND", [1])
    finally:
        spw_tab.unlock()
        spw_tab.close()

    dish_diameter = [size] * num_ants
    ant_mount = [mount] * num_ants
    teltype = ["GROUND_BASED"] * num_ants

    ant_table = table(f"{ms_name}.ms::ANTENNA",
                      readonly=False, lockoptions='user', ack=False)
    try:
        ant_table.lock(write=True)
        ant_table.addrows(num_ants)
        ant_table.putcol("DISH_DIAMETER", dish_diameter)
        ant_table.putcol("MOUNT", ant_mount)
        ant_table.putcol("POSITION", antlocation)
        ant_table.putcol("TYPE", teltype)

    finally:
        ant_table.unlock()
        ant_table.close()

    fld_tab = table(f"{ms_name}.ms::FIELD",
                    readonly=False, lockoptions='user', ack=False)

    try:
        fld_tab.lock(write=True)
        fld_tab.addrows(1)
        fld_tab.putcol("PHASE_DIR", phase_dir)
        fld_tab.putcol("DELAY_DIR", phase_dir)
        fld_tab.putcol("REFERENCE_DIR", phase_dir)
    finally:
        fld_tab.unlock()
        fld_tab.close()

    dd_tab = table(f"{ms_name}.ms::DATA_DESCRIPTION",
                   readonly=False, lockoptions='user', ack=False)
    try:
        dd_tab.lock(write=True)
        dd_tab.addrows(1)
    finally:
        dd_tab.unlock()
        dd_tab.close()

    obs_tab = table(f"{ms_name}.ms::OBSERVATION",
                    readonly=False, lockoptions='user', ack=False)
    try:
        obs_tab.lock(write=True)
        obs_tab.addrows(1)
        obs_tab.putcol("TIME_RANGE", time_range)
        obs_tab.putcol("OBSERVER", 'simms simulator')
        obs_tab.putcol("TELESCOPE_NAME", telescope_name)
    finally:
        obs_tab.unlock()
        obs_tab.close()

    pntng_tab = table(f"{ms_name}.ms::POINTING",
                      readonly=False, lockoptions='user', ack=False)
    try:
        pntng_tab.lock(write=True)
        pntng_tab.addrows(1)
    finally:
        pntng_tab.unlock()
        pntng_tab.close()

    pol_tab = table(f"{ms_name}.ms::POLARIZATION",
                    readonly=False, lockoptions='user', ack=False)
    try:
        pol_tab.lock(write=True)
        pol_tab.addrows(1)
        pol_tab.putcol("NUM_CORR", num_corr)
    finally:
        pol_tab.unlock()
        pol_tab.close()

    feed_tab = table(f"{ms_name}.ms::FEED",
                     readonly=False, lockoptions='user', ack=False)
    try:
        feed_tab.lock(write=True)
        feed_tab.addrows(num_ants)
    finally:
        feed_tab.unlock()
        feed_tab.close()
