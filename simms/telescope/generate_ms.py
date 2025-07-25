import os
import shutil
from itertools import combinations
from typing import List, Union
import dask
import dask.array as da
import daskms
import numpy as np
from casacore.measures import measures
from daskms import xds_to_table
from omegaconf import OmegaConf
from scabha.basetypes import File
from tqdm.dask import TqdmCallback

import simms
from simms.constants import PI
from simms.telescope import array_utilities as autils

CORR_TYPES = OmegaConf.load(f"{simms.PCKGDIR}/telescope/ms_corr_types.yaml").CORR_TYPES
dm = measures()

# Numpy does the correct type setting for strings
nda = lambda items: da.asarray(np.array(items))

log = simms.get_logger(name="telsim")


def remove_ms(ms: Union[File, str]):

    if os.path.exists(ms):
        shutil.rmtree(ms, ignore_errors=True)
        log.debug(f"MS file {ms} exists. It will be overriden.")
    else:
        log.debug(f"MS file {ms} is being created.")


def create_ms(
    ms: str,
    telescope_name: Union[str, File],
    pointing_direction: str,
    dtime: int,
    ntimes: int,
    start_freq: Union[str, float],
    dfreq: Union[str, float],
    nchan: int,
    correlations: str,
    row_chunks: int,
    sefd: float,
    tsys_over_eta: float,
    column: str,
    start_time: Union[str, List[str]] = None,
    start_ha: float = None,
    freq_range: str = None,
    sfile: File = None,
    subarray_list: List[str] = None,
    subarray_range: List[int] = None,
    subarray_file: File = None,
    low_source_limit: Union[float, str] = None,
    high_source_limit: Union[float, str] = None,
    low_antenna_limit: Union[float,str] = None,
    high_antenna_limit: Union[float,str] = None,
):
    "Creates an empty Measurement Set for an observation using given observation parameters"

    remove_ms(ms)
    telescope_array = autils.Array(telescope_name, 
                                   sefd=sefd,
                                   tsys_over_eta=tsys_over_eta, 
                                   sensitivity_file=sfile, 
                                   subarray_list=subarray_list, subarray_range=subarray_range,
                                   subarray_file=subarray_file)
    size = telescope_array.size
    mount = telescope_array.mount
    antnames = telescope_array.names
    antlocation = telescope_array.antlocations
    
    if freq_range:
        start_freq = dm.frequency(v0=freq_range[0])["m0"]["value"]
        end_freq = dm.frequency(v0=freq_range[1])["m0"]["value"]
        nchan = int(freq_range[2])
        dfreq = (end_freq - start_freq) / (nchan - 1)
    else: 
        start_freq = dm.frequency(v0=start_freq)["m0"]["value"]
        dfreq = dm.frequency(v0=dfreq)["m0"]["value"]
        end_freq = start_freq + dfreq * (nchan -1)
        
    freqs = np.linspace(start_freq, end_freq, nchan)

    uvcoverage_data = telescope_array.uvgen(
        pointing_direction,
        dtime,
        ntimes,
        start_time,
        start_ha,
    )

    num_rows = uvcoverage_data.times.shape[0]
    num_row_chunks = row_chunks

    num_chans = len(freqs)
    num_corr = len(correlations)
    corr_types = np.array([[CORR_TYPES[x] for x in correlations]])
    # TODO(sphe) use casacore to determine this
    if num_corr == 2:
        corr_products = np.array([([0, 0], [1, 1])])
    else:
        corr_products = np.array([([0, 0], [0, 1], [1, 0], [1, 1])])

    num_ants = antlocation.shape[0]
    ant1 = uvcoverage_data.antenna1 * ntimes
    ant2 = uvcoverage_data.antenna2 * ntimes

    ra_dec = dm.direction(*pointing_direction)
    ra = ra_dec["m0"]["value"]
    dec = ra_dec["m1"]["value"]
    phase_dir = np.array([[[ra, dec]]])

    data = da.zeros(
        (num_rows, num_chans, num_corr), chunks=(num_row_chunks, num_chans, num_corr)
    )
    times = da.from_array(uvcoverage_data.times, chunks=num_row_chunks)
    time_range = np.array([[uvcoverage_data.times[0], uvcoverage_data.times[-1]]])
    uvw = da.from_array(uvcoverage_data.uvw, chunks=(num_row_chunks, 3))
    antenna1 = da.from_array(ant1, chunks=num_row_chunks)
    antenna2 = da.from_array(ant2, chunks=num_row_chunks)
    interval = da.full(num_rows, dtime)
    interval = da.rechunk(interval, chunks=num_row_chunks)
    sigma = da.full((num_rows, num_corr), 1.0)
    sigma = da.rechunk(sigma, chunks=(num_row_chunks, num_corr))
    sigma_spec = da.full((num_rows, num_chans, num_corr), 1.0)
    sigma_spec = da.rechunk(sigma_spec, chunks=(num_row_chunks, num_chans, num_corr))
    flag = da.zeros(
        (num_rows, num_chans, num_corr),
        dtype=bool,
        chunks=(num_row_chunks, num_chans, num_corr),
    )

    freqs = freqs.reshape(1, freqs.shape[0])
    channel_widths = da.full(freqs.shape, dfreq)
    total_bandwidth = nchan * dfreq
    
    ds = {
        "DATA": (("row", "chan", "corr"), data),
        "UVW": (("row", "uvw_dim"), uvw),
        "TIME": (("row"), times),
        "TIME_CENTROID": (("row"), times),
        "INTERVAL": (("row"), interval),
        "EXPOSURE": (("row"), interval),
        "ANTENNA1": (("row"), antenna1),
        "ANTENNA2": (("row"), antenna2),
        "SIGMA": (("row", "corr"), sigma),
        "WEIGHT": (("row", "corr"), sigma),
        "SIGMA_SPECTRUM": (("row", "chan", "corr"), sigma_spec),
        "WEIGHT_SPECTRUM": (("row", "chan", "corr"), sigma_spec),
        "FLAG": (("row", "chan", "corr"), flag),
        "FLAG_CATEGORY": (("row", "flagcat", "chan", "corr"), flag[:, None, :, :]),
    }
    
    dd_id = 0
    ddid = da.rechunk(da.full(num_rows, dd_id), chunks=num_row_chunks)
    
    ds["DATA_DESC_ID"] = ("row",), ddid
    
    field_id = 0
    ds["FIELD_ID"] = ("row"), da.full_like(ddid, fill_value=field_id)
    
    scan_number = 1
    ds["SCAN_NUMBER"] = ("row"), da.full_like(ddid, fill_value=scan_number)
    
    state_id = 0
    ds["STATE_ID"] = ("row"), da.full_like(ddid, fill_value=state_id)
    
    array_id = 0
    ds["ARRAY_ID"] = ("row"), da.full_like(ddid, fill_value=array_id)
    
    obs_id = 0
    ds["OBSERVATION_ID"] = ("row"), da.full_like(ddid, fill_value=obs_id)
    
    proc_id = 0
    ds["PROCESSOR_ID"] = ("row"), da.full_like(ddid, fill_value=proc_id)
    
    sefd = telescope_array.sefd
    
    if sefd:
        noise = get_noise(sefd, ntimes, dtime, dfreq)
        dummy_data = np.random.randn(
            num_rows, num_chans, num_corr
        ) + 1j * np.random.randn(num_rows, num_chans, num_corr)
        if isinstance(noise, (float, int)):
            noisy_data = da.array(dummy_data * noise, like=data).rechunk(
                chunks=data.chunks
            )
        else:
            noise = np.array(noise)
            noisy_data = da.array(dummy_data * noise[:, None, None], like=data).rechunk(
                chunks=data.chunks
            )

        ds[column] = (("row", "chan", "corr"), noisy_data)

    nbaselines = num_ants * (num_ants - 1) // 2
    src_elevs = uvcoverage_data.source_elevations
    expanded_src_elevations = []
    
    for elevation in src_elevs:
        all_baselines_elevation_per_time = [elevation] * nbaselines
        expanded_src_elevations.append(all_baselines_elevation_per_time)
        
    expanded_src_elevations = np.array(expanded_src_elevations).flatten()

    flag_row = np.zeros(num_rows, dtype=bool)
        
    if low_source_limit:
        for i in range(num_rows):
            if expanded_src_elevations[i] < low_source_limit:
                flag_row[i] = True
                    
    if high_source_limit:
        for i in range(num_rows):
            if expanded_src_elevations[i] > high_source_limit:
                flag_row[i] = True    
    
    ds["FLAG_ROW"] = (("row",), da.from_array(flag_row, chunks=num_row_chunks))        

    main_table = daskms.Dataset(ds, coords={"ROWID": ("row", da.arange(num_rows))})

    write_main = xds_to_table(main_table, ms, columns="ALL", descriptor="ms(False)")
    with TqdmCallback(desc=f"Writing the Main Table to {ms}"):
        dask.compute(write_main)

    pol_response = da.array([[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]])

    receptor_angles = np.tile([0, 0.5 * PI], (num_ants, 1))

    feed_ds = {
        "ANTENNA_ID": (("row"), da.arange(num_ants)),
        "BEAM_ID": (("row"), da.from_array(np.full(num_ants, -1))),
        "BEAM_OFFSET": (
            ("row", "receptors", "radec"),
            da.from_array(np.zeros((num_ants, 2, 2), dtype=float)),
        ),
        "INTERVAL": (("row"), da.from_array(np.full(num_ants, 1.0e30))),
        "NUM_RECEPTORS": (("row"), da.from_array(np.full(num_ants, 2))),
        "SPECTRAL_WINDOW_ID": (("row"), da.from_array(np.full(num_ants, -1))),
        # TODO: @Allycan Account for circular polarisation
        "POLARIZATION_TYPE": (
            ("row", "receptors"),
            da.from_array(np.full((num_ants, 2), (["X", "Y"]))),
        ),
        "RECEPTOR_ANGLE": (
            ("row", "receptors"),
            da.from_array(receptor_angles),
        ),
        "POL_RESPONSE": (
            ("row", "receptors", "receptors-2"),
            da.from_array(np.full((num_ants, 2, 2), pol_response)),
        ),
    }

    feed_table = daskms.Dataset(feed_ds)
    ms_desc = "ms_subtable(True)"
    write_feed = xds_to_table(feed_table, f"{ms}::FEED", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the FEED table to {ms}"):
        dask.compute(write_feed)

    spw_ds = {
        "CHAN_FREQ": (("row", "chan"), da.from_array(freqs)),
        "CHAN_WIDTH": (("row", "chan"), channel_widths),
        "EFFECTIVE_BW": (("row", "chan"), channel_widths),
        "RESOLUTION": (("row", "chan"), channel_widths),
        "REF_FREQ": (("row"), da.from_array([start_freq])),
        "MEAS_RES_FREQ": (("row"), da.from_array([start_freq])),
        "TOTAL_BANDWIDTH": (("row"), da.from_array([total_bandwidth])),
        "NUM_CHAN": (("row"), da.from_array([nchan])),
        "NAME": (("row"), da.from_array(["00"])),
        "NET_SIDEBAND": (("row"), da.array([1])),
        "FREQ_GROUP_NAME": (("row"), da.from_array(["GROUP 1"])),
    }

    spw_table = daskms.Dataset(spw_ds)

    write_spw = xds_to_table(spw_table, f"{ms}::SPECTRAL_WINDOW")
    with TqdmCallback(desc=f"Writing the SPECTRAL_WINDOW table to {ms}"):
        dask.compute(write_spw)

    
    if isinstance(size, (int, float)):
        dish_diameter = [size] * num_ants
    else:
        dish_diameter = np.array(size)
        
    if isinstance(mount, str):
        ant_mount = [mount] * num_ants
    else:
        ant_mount = np.array(mount)

    names = np.array(antnames)
    teltype = ["GROUND_BASED"] * num_ants
    ant_ds = {
        "DISH_DIAMETER": (("row"), da.from_array(dish_diameter)),
        "MOUNT": (("row"), da.from_array(ant_mount)),
        "POSITION": (("row", "xyz"), da.from_array(antlocation)),
        "NAME": (("row"), da.from_array(names)),
        "STATION": (("row"), da.from_array(names)),
        "TYPE": (("row"), da.from_array(teltype)),
    }

    ant_table = daskms.Dataset(ant_ds)

    write_ant = xds_to_table(ant_table, f"{ms}::ANTENNA", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the ANTENNA table to {ms}"):
        dask.compute(write_ant)

    field_name = "TARGET"
    ftimes = [0.0]

    fld_ds = {
        "NAME": (("row"), nda([field_name])),
        "PHASE_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "DELAY_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "REFERENCE_DIR": (("row", "field-poly", "field-dir"), da.from_array(phase_dir)),
        "TIME": (("row"), da.from_array(ftimes)),
        "SOURCE_ID": (("row"), da.from_array([field_id])),
    }

    fld_table = daskms.Dataset(fld_ds)

    write_fld = xds_to_table(fld_table, f"{ms}::FIELD", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the FIELD table to {ms}"):
        dask.compute(write_fld)

    obs_ds = {
        "TIME_RANGE": (("row", "obs-exts"), da.from_array(time_range)),
        "OBSERVER": (("row"), da.from_array(np.array(["simms simulator"]))),
        "PROJECT": (("row"), da.from_array(np.array(["simms simulation"]))),
        "TELESCOPE_NAME": (("row"), da.from_array(np.array([telescope_name]))),
    }

    obs_table = daskms.Dataset(obs_ds)

    write_obs = xds_to_table(obs_table, f"{ms}::OBSERVATION", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the OBSERVATION table to {ms}"):
        dask.compute(write_obs)

    pol_ds = {
        "NUM_CORR": (("row"), da.from_array(np.array([num_corr]))),
        "CORR_PRODUCT": (("row", "corr", "corrprod_idx"), da.from_array(corr_products)),
        "CORR_TYPE": (("row", "corr"), da.from_array(corr_types)),
    }

    pol_table = daskms.Dataset(pol_ds)

    write_pol = xds_to_table(pol_table, f"{ms}::POLARIZATION", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the POLARIZATION table to {ms}"):
        dask.compute(write_pol)

    phase_arr = da.from_array(np.full((num_rows, 1, 2), phase_dir))

    pntng_ds = {
        "TARGET": (("row", "point-poly", "radec"), phase_arr),
        "TIME": (("row"), da.from_array(uvcoverage_data.times)),
        "INTERVAL": (("row"), da.from_array(np.full(num_rows, dtime))),
        "TRACKING": (("row"), da.from_array(np.full(num_rows, True))),
    }

    pntng_table = daskms.Dataset(pntng_ds)

    write_pntng = xds_to_table(pntng_table, f"{ms}::POINTING", descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the POINTING table to {ms}"):
        dask.compute(
            write_pntng,
        )

    dir_ds = {
        "DIRECTION": (("row", "point-poly", "radec"), phase_arr),
    }

    dir_table = daskms.Dataset(dir_ds)

    write_dir = xds_to_table(dir_table, f"{ms}::POINTING", columns=["DIRECTION"], descriptor=ms_desc)
    with TqdmCallback(desc=f"Writing the DIRECTION column to POINTING table to {ms}"):
        dask.compute(write_dir)
        
    # add PROCESSOR table
    processor_table = daskms.Dataset({
        "TYPE": (("row",), nda(["CORRELATOR"])),
        "SUB_TYPE": (("row",), nda(["UNSET"])),
        "TYPE_ID": (("row",), nda([proc_id])),
        "MODE_ID": (("row",), nda([proc_id])),
        "FLAG_ROW": (("row",), nda([False])),
    })
    
    with TqdmCallback(desc=f"Writing the PROCESSOR table to {ms}"):
        dask.compute(
            xds_to_table(processor_table, f"{ms}::PROCESSOR"),
        )
    
    # add DATA_DESC table
    datadesc_table = daskms.Dataset( {
        "SPECTRAL_WINDOW_ID": (("row",), da.array([dd_id])),
        "POLARIZATION_ID": (("row",), da.array([dd_id])),
        "LAG_ID": (("row",), da.array([0])),
        "FLAG_ROW": (("row",), da.array([False])),
    })

    with TqdmCallback(desc=f"Writing the DATA_DESCRIPTION table to {ms}"):
        dask.compute(
            xds_to_table(datadesc_table, f"{ms}::DATA_DESCRIPTION"),
        )

        
    # add state table
    state_table = daskms.Dataset( {
        "SIG": (("row",), da.array([True])),
        "REF": (("row",), da.array([False])),
        "CAL": (("row",), da.array([0.0])),
        "LOAD": (("row",), da.array([0.0])),
        "SUB_SCAN": (("row",), da.array([state_id])),
        "OBS_MODE": (("row",), nda(['OBSERVE_TARGET.ON_SOURCE'])),
        "FLAG_ROW": (("row",), da.array([False],dtype=bool)),
    })

    with TqdmCallback(desc=f"Writing the STATE table to {ms}"):
        dask.compute(
            xds_to_table(state_table, f"{ms}::STATE", descriptor=ms_desc),
        )

    log.info(f"{ms} successfully generated.")


def get_noise(sefds: Union[List, float], ntime: int, dtime: int, dfreq: float):
    """
    This function computes the noise given an SEFD/s.
    """

    if isinstance(sefds, (int, float)):
        noise = sefds / np.sqrt(2 * dfreq * dtime)
        return noise

    sefd_pairs = list(combinations(sefds, 2))
    noises = []
    for sefd1, sefd2 in sefd_pairs:
        prod = sefd1 * sefd2
        den = 2 * dfreq * dtime
        noise = np.sqrt(prod / den)
        noises.append(noise)

    return noises * ntime
