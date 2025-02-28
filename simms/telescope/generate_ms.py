import os
import shutil
from itertools import combinations
from typing import List, Union
from tqdm.dask import TqdmCallback
import dask
import dask.array as da
import daskms
import numpy as np
from casacore.measures import measures
from casacore.tables import table
from daskms import xds_to_table,xds_from_table
from scabha.basetypes import File
from simms.telescope import array_utilities as autils
from omegaconf import OmegaConf
import simms

CORR_TYPES = OmegaConf.load(f"{simms.PCKGDIR}/telescope/ms_corr_types.yaml").CORR_TYPES
dm = measures()

log = simms.get_logger(name="telsim")

def remove_ms(ms: Union[File,str]):
    
    if os.path.exists(ms):
        shutil.rmtree(ms, ignore_errors=True)
        log.debug(
            f"MS file {ms} exists. It will be overriden.")
    else:
        log.debug(
            f"MS file {ms} is being created.")


def create_ms(ms: str, telescope_name: Union[str, File],
            pointing_direction: str, dtime: int, ntimes: int,
            start_freq: Union[str, float], dfreq: Union[str, float],
            nchan: int, correlations: str, row_chunks: int,
            sefd: float, column: str,
            start_time: Union[str, List[str]] = None,
            start_ha: float = None, horizon_limit: Union[float, str] = None,
            ):
    "Creates an empty Measurement Set for an observation using given observation parameters"

    remove_ms(ms)
    telescope_array = autils.Array(telescope_name,sefd=sefd)
    telescope_array.set_arrayinfo()
    size = telescope_array.size
    mount = telescope_array.mount
    antnames = telescope_array.names
    antlocation, _ = telescope_array.geodetic2global()

    uvcoverage_data = telescope_array.uvgen(pointing_direction, dtime, ntimes,
                                            start_freq, dfreq, nchan, start_time,
                                            start_ha)

    num_rows = uvcoverage_data.times.shape[0]
    num_row_chunks = row_chunks

    num_chans = len(uvcoverage_data.freqs)
    num_corr = len(correlations)
    corr_types = np.array([[CORR_TYPES[x] for x in correlations]])
    # TODO(sphe) use casacore to determine this
    if num_corr == 2: 
        corr_products = np.array([([0,0],[1,1])])
    else:
        corr_products = np.array([([0,0], [0,1], [1,0], [1,1])])

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
    duration = ntimes * dtime
    uvw = da.from_array(uvcoverage_data.uvw, chunks=(num_row_chunks, 3))
    antenna1 = da.from_array(ant1, chunks=num_row_chunks)
    antenna2 = da.from_array(ant2, chunks=num_row_chunks)
    interval = da.full(num_rows, dtime)
    interval = da.rechunk(interval, chunks=num_row_chunks)
    sigma = da.full((num_rows, num_corr), 1.)
    sigma = da.rechunk(sigma, chunks=(num_row_chunks, num_corr))
    sigma_spec = da.full((num_rows, num_chans,num_corr), 1.)
    sigma_spec = da.rechunk(sigma_spec, chunks=(num_row_chunks, num_chans,num_corr))
    flag = da.zeros((num_rows, num_chans, num_corr),
                    dtype=bool, chunks=(num_row_chunks, num_chans, num_corr))
    scan_number = da.rechunk(da.full(num_rows,1),chunks=num_row_chunks)

    freqs = uvcoverage_data.freqs
    freqs = freqs.reshape(1, freqs.shape[0])
    ref_freq = dm.frequency(v0=start_freq)["m0"]["value"]
    chan_width = dm.frequency(v0=dfreq)["m0"]["value"]
    channel_widths = da.full(freqs.shape, chan_width)
    total_bandwidth = nchan * chan_width

    

    ds = {
        'DATA_DESC_ID': (("row",), ddid),
        'DATA': (("row", "chan", "corr"), data),
        "UVW": (("row", "uvw_dim"), uvw),
        "TIME": (("row"), times),
        "TIME_CENTROID": (("row"), times),
        "INTERVAL": (("row"), interval),
        "EXPOSURE": (("row"), interval),
        "ANTENNA1": (("row"), antenna1),
        "ANTENNA2": (("row"), antenna2),
        "SIGMA": (("row", "corr"), sigma),
        "WEIGHT": (("row", "corr"), sigma),
        "SIGMA_SPECTRUM": (("row", "chan","corr"), sigma_spec),
        "WEIGHT_SPECTRUM": (("row","chan", "corr"), sigma_spec),
        "FLAG": (("row", "chan", "corr"), flag),
        "FLAG_CATEGORY":(("row","flagcat","chan","corr"),flag[:,None,:,:]),
        "SCAN_NUMBER":(("row"),scan_number)
        
    }

    sefd = telescope_array.sefd
   
    if sefd:
        noise = get_noise(sefd, ntimes, dtime, chan_width)
        dummy_data = np.random.randn(num_rows, num_chans, num_corr) + \
            1j*np.random.randn(num_rows, num_chans, num_corr)
        if isinstance(noise, (float,int)):
            noisy_data = da.array(dummy_data * noise,like=data).rechunk(chunks=data.chunks)
        else:
            noise = np.array(noise)
            noisy_data = da.array(dummy_data * noise[:,None,None],like=data).rechunk(chunks=data.chunks)
       
        ds[column] = (("row", "chan", "corr"), noisy_data)

    main_table = daskms.Dataset(
        ds, coords={"ROWID": ("row", da.arange(num_rows))})

    write_main = xds_to_table(main_table, ms)
    with TqdmCallback(desc=f'Writing the Main Table to {ms}'):
        dask.compute(write_main)
        
        
 
    pol_response = da.array([[[1.+0.j, 0.+0.j],
         [0.+0.j, 1.+0.j]]])
    
    feed_ds = {
        "ANTENNA_ID":(("row"),da.arange(num_ants)),
        "BEAM_ID":(("row"),da.from_array(np.full(num_ants,-1))),
        "BEAM_OFFSET":(("row","receptors", "radec"),da.from_array(np.zeros((num_ants,2,2),dtype=float))),
        "INTERVAL":(("row"),da.from_array(np.full(num_ants,1.e30))),
        "NUM_RECEPTORS":(("row"),da.from_array(np.full(num_ants,2))),
        "SPECTRAL_WINDOW_ID":(("row"),da.from_array(np.full(num_ants,-1))),
        "POLARIZATION_TYPE":(("row", "receptors"), da.from_array(np.full((num_ants,2),(['X','Y'])))),
        "POL_RESPONSE":(("row", "receptors", "receptors-2"),da.from_array(np.full((num_ants,2,2),pol_response)))
    }

    feed_table = daskms.Dataset(
        feed_ds)

    write_feed = xds_to_table(feed_table, f"{ms}::FEED")
    with TqdmCallback(desc=f'Writing the FEED table to {ms}'):
        dask.compute(write_feed)

    spw_ds = {
        "CHAN_FREQ":(("row","chan"),da.from_array(freqs)),
        "CHAN_WIDTH":(("row","chan"),channel_widths),
        "EFFECTIVE_BW":(("row","chan"),channel_widths),
        "RESOLUTION":(("row","chan"), channel_widths),
        "REF_FREQ":(("row"),da.from_array([ref_freq])),
        "MEAS_RES_FREQ":(("row"),da.from_array([ref_freq])),
        "TOTAL_BANDWIDTH":(("row"),da.from_array([total_bandwidth])),
        "NUM_CHAN":(("row"),da.from_array([nchan])),
        "NAME":(("row"),da.from_array(["00"])),
        "NET_SIDEBAND":(("row"),da.array([1])),
        "FREQ_GROUP_NAME":(("row"),da.from_array(["GROUP 1"]))
      
    }
    
    spw_table = daskms.Dataset(
        spw_ds)

    write_spw = xds_to_table(spw_table, f"{ms}::SPECTRAL_WINDOW")
    with TqdmCallback(desc=f'Writing the SPECTRAL_WINDOW table to {ms}'):
        dask.compute(write_spw)

        
    if not isinstance(size, (int, float)):
        dish_diameter = np.array(size)
        ant_mount = np.array(mount)
    else:
        dish_diameter = [size] * num_ants
        ant_mount = [mount] * num_ants
        # names = [f"ANT-{x}" for x in range(num_ants)]
        
    names = np.array(antnames) 
    teltype = ["GROUND_BASED"] * num_ants
    
    ant_ds = {
        "DISH_DIAMETER":(("row"),da.from_array(dish_diameter)),
        "MOUNT":(("row"),da.from_array(ant_mount)),
        "POSITION":(("row","xyz"),da.from_array(antlocation)),
        "NAME":(("row"),da.from_array(names)),
        "STATION":(("row"),da.from_array(names)),
        "TYPE":(("row"),da.from_array(teltype))
    }
    
    ant_table = daskms.Dataset(
        ant_ds)

    write_ant = xds_to_table(ant_table, f"{ms}::ANTENNA")
    with TqdmCallback(desc=f'Writing the ANTENNA table to {ms}'):
        dask.compute(write_ant)

    fld_ds = {
        "PHASE_DIR":(("row", "field-poly", "field-dir"),da.from_array(phase_dir)),
        "DELAY_DIR":(("row", "field-poly", "field-dir"),da.from_array(phase_dir)),
        "REFERENCE_DIR":(("row", "field-poly", "field-dir"),da.from_array(phase_dir)),
        "TIME":(("row"),da.from_array(np.array([0.0]))),
        "SOURCE_ID":(("row"),da.from_array(np.array([0])))
    }
    
    fld_table = daskms.Dataset(
        fld_ds)

    write_fld = xds_to_table(fld_table, f"{ms}::FIELD")
    with TqdmCallback(desc=f'Writing the FIELD table to {ms}'):
        dask.compute(write_fld)
    
    
    
    autils.ms_addrow(ms,"DATA_DESCRIPTION",1)
    
    obs_ds = {
        "TIME_RANGE":(("row","obs-exts"),da.from_array(time_range)),
        "OBSERVER":(("row"),da.from_array(np.array(["simms simulator"]))),
        "PROJECT":(("row"),da.from_array(np.array(["simms simulation"]))),
        "TELESCOPE_NAME":(("row"),da.from_array(np.array([telescope_name]))),
    }
    
    obs_table = daskms.Dataset(
        obs_ds)

    write_obs = xds_to_table(obs_table, f"{ms}::OBSERVATION")
    with TqdmCallback(desc=f'Writing the OBSERVATION table to {ms}'):
        dask.compute(write_obs)
    
    pol_ds = {
        "NUM_CORR":(("row"),da.from_array(np.array([num_corr]))),
        "CORR_PRODUCT":(("row", "corr", "corrprod_idx"),da.from_array(corr_products)),
        "CORR_TYPE":(("row","corr"),da.from_array(corr_types))
    }
    
    pol_table = daskms.Dataset(
        pol_ds)

    write_pol = xds_to_table(pol_table, f"{ms}::POLARIZATION")
    with TqdmCallback(desc=f'Writing the POLARIZATION table to {ms}'):
        dask.compute(write_pol)
    
    phase_arr = da.from_array(np.full((num_rows,1,2),phase_dir))
    
    pntng_ds = {
        "TARGET":(("row", "point-poly", "radec"),phase_arr),
        "TIME":(("row"),da.from_array(uvcoverage_data.times)),
        "INTERVAL":(("row"),da.from_array(np.full(num_rows,dtime))),
        "TRACKING":(("row"),da.from_array(np.full(num_rows,True))),
        
    }
    
    pntng_table = daskms.Dataset(
        pntng_ds)

    write_pntng = xds_to_table(pntng_table, f"{ms}::POINTING")
    with TqdmCallback(desc=f'Writing the POINTING table to {ms}'):
        dask.compute(write_pntng,)

    
    dir_ds = {
        "DIRECTION":(("row", "point-poly", "radec"),phase_arr),
    }
    
    dir_table = daskms.Dataset(
        dir_ds)

    write_dir = xds_to_table(dir_table, f"{ms}::POINTING",columns=["DIRECTION"])
    with TqdmCallback(desc=f'Writing the DIRECTION column to POINTING table to {ms}'):
        dask.compute(write_dir)
    

    log.info(f"{ms} successfully generated.")

def get_noise(sefds: Union[List,float], ntime: int, dtime: int, chan_width: float):
        """
        This function computes the noise given an SEFD/s.
        """
       
        if isinstance(sefds, (int, float)):
            noise = np.sqrt(sefds/(2*chan_width*dtime))
            return noise
        
        
        sefd_pairs = list(combinations(sefds, 2))
        noises = []
        for sefd1, sefd2 in sefd_pairs:
            prod = sefd1 * sefd2
            den = 2*chan_width*dtime
            noise = np.sqrt(prod/den)
            noises.append(noise)
        
        
        return noises * ntime