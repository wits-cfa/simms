from typing import List, Optional, Tuple, Union

import numpy as np
from africanus.dft import im_to_vis as dft_im_to_vis
from daskms import xds_from_ms, xds_from_table
from ducc0.wgridder import dirty2ms
from scabha.basetypes import MS

from simms.skymodel.converters import radec2lm
from simms.skymodel.source_factory import Source


def vis_noise_from_sefd_and_ms(ms: Union[MS, str], sefd: float, spw_id: int = 0, field_id: int = 0):
    """
    Compute per visibility noise from an SEFD and MS

    Args:
        ms (Union[MS,str]): MS path
        sefd (float): Antenna SEFD in Jy
        spw_id (int, optional): Data description ID. Defaults to 0.
        field_id (int, optional): Field ID. Defaults to 0.

    Returns:
        float: noise per visibility
    """

    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"], taql_where=f"FIELD_ID=={field_id}")[spw_id]

    df = spw_ds.CHAN_WIDTH.data[spw_id][0]

    dt = msds.EXPOSURE.data[0]
    noise_vis = sefd / np.sqrt(2 * dt * df)

    return noise_vis


def sim_noise(dshape: Union[List, Tuple], vis_noise: float) -> np.ndarray:
    """AI is creating summary for simnoise

    Args:
        dshape (Union[List, Tuple]): [description]
        vis_noise (float): [description]

    Returns:
        np.ndarray: [description]
    """

    return np.random.randn(*dshape) * vis_noise + np.random.randn(*dshape) * vis_noise * 1j


def add_noise(vis: Union[np.ndarray, float], vis_noise: float):
    """AI is creating summary for add_noise

    Args:
        vis (Union[np.ndarray,float]): Data to add noise to. Set to zero to compute a noise only visibility
        noise_vis (float): Noise per visibility
    """
    return vis + sim_noise(vis.shape, vis_noise)


def add_to_vis(vis0: np.ndarray, vis1: np.ndarray, subtract: bool = False) -> np.ndarray:
    """Add/subtract two visibility arrays

    Args:
        vis0 (np.ndarray): Add/subtract to/from
        vis1 (np.ndarray): data to add/subtract
        subtract (bool, optional): Should the data be subtracted. Defaults to False.

    Returns:
        np.ndarray: the added/subtracted visibility data
    """
    if subtract:
        return vis0 - vis1
    else:
        return vis0 + vis1


def stack_unpolarised_vis(vis: np.ndarray, ncorr: int) -> np.ndarray:
    """
    Takes XX or RR visibilities and creates visibility array with shape (nrows, nchan, ncorr).
    Used to avoid double computation of identical correlations.

    Args:
        vis: numpy array of shape (nrows, nchan) containing the visibility data
        ncorr: number of correlations (2 or 4)
    Returns:
        vis: numpy array of shape (nrows, nchan, ncorr) containing the visibility data
    """
    if ncorr == 2:
        vis = np.stack([vis, vis], axis=2)
    elif ncorr == 4:
        vis = np.stack([vis, np.zeros_like(vis), np.zeros_like(vis), vis], axis=2)
    else:
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    return vis


def fft_im_to_vis(
    uvw: np.ndarray,
    chan_freq: np.ndarray,
    image: np.ndarray,
    pixsize_x: float,
    pixsize_y: float,
    epsilon: Optional[float] = 1e-7,
    nthreads: Optional[int] = 8,
    do_wstacking: Optional[bool] = True,
) -> np.ndarray:
    """
    ducc0.wgridder.dirty2ms wrapper to add squeezing and conjugation.
    NB: Image should be 2D.
    """

    result = dirty2ms(
        uvw,
        chan_freq,
        image,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wstacking=True,
        nthreads=nthreads,
    )

    return np.conj(np.squeeze(result))


def compute_vis(
    sources: List[Source],
    uvw: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray = None,
    ncorr: int = 2,
    polarisation: bool = False,
    pol_basis: str = "linear",
    ra0: float = None,
    dec0: float = None,
    noise_vis: Optional[float] = None,
):
    """
    Computes visibilities

    Args:
        srcs (list):                List of Source objects
        uvw (numpy.ndarray):        Array of shape (3, nrows) containing the UVW coordinates
        freqs (numpy.ndarray):      Array of shape (nchan,) containing the frequencies
        ncorr (int):                Number of correlations
        polarisation (bool):        True if polarisation information is present, False otherwise
        pol_basis (str):            Polarisation basis ("linear" or "circular")
        times:                     Number of unique times
        mod_data (numpy.ndarray):   Array of shape (nrows, nchan, ncorr) containing the model data
            to/from which computed visibilities should be added/subtracted
        noise (float):              RMS noise
        subtract (bool):            True if visibilities should be subtracted from the model data, False otherwise

    Returns:
        vis (numpy.ndarray):        Visibility array of shape (nrows, nchan, ncorr)
    """

    wavs = 2.99e8 / freqs
    uvw_scaled = uvw.T[..., np.newaxis] / wavs

    # helper function to calculate phase factor
    def calculate_phase_factor(src):
        el, em = radec2lm(ra0, dec0, src.ra, src.dec)
        n_term = np.sqrt(1 - el * el - em * em) - 1
        arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term

        if not src.emaj and not src.emin:
            # point source
            return np.exp(2 * np.pi * 1j * arg)
        else:
            # extended source
            ell = src.emaj * np.sin(src.pa)
            emm = src.emaj * np.cos(src.pa)
            ecc = src.emin / (1.0 if src.emaj == 0.0 else src.emaj)

            fu1 = (uvw_scaled[0] * emm - uvw_scaled[1] * ell) * ecc
            fv1 = uvw_scaled[0] * ell + uvw_scaled[1] * emm

            shape_phase = fu1 * fu1 + fv1 * fv1
            return np.exp(2 * np.pi * 1j * arg - shape_phase)

    vis_xx = 0
    vis_xy = 0
    vis_yx = 0
    vis_yy = 0

    for source in sources:
        phase = calculate_phase_factor(source)
        bmatrix = source.stokes.get_brightness_matrix(ncorr, pol_basis == "linear")
        if source.is_transient:
            if isinstance(times, type(None)):
                raise ValueError("Times must be provided for transient sources")
            _, time_index_mapper = np.unique(times, return_inverse=True)
            bmatrix = bmatrix[:, time_index_mapper, ...]
        vis_xx += bmatrix[0, ...] * phase
        if ncorr == 2:
            if polarisation:
                vis_yy += bmatrix[ncorr - 1, ...] * phase
            else:
                vis_yy = vis_xx
        else:
            if polarisation:
                vis_xy += bmatrix[1, ...] * phase
                vis_yx += bmatrix[2, ...] * phase
                vis_yy += bmatrix[3, ...] * phase
            else:
                vis_xy = np.zeros_like(vis_xx)
                vis_yx = np.zeros_like(vis_xx)
                vis_yy += vis_xx

    if ncorr == 2:
        vis = np.stack((vis_xx, vis_yy), axis=-1)
    else:
        vis = np.stack((vis_xx, vis_xy, vis_yx, vis_yy), axis=-1)

    if noise_vis:
        vis = add_noise(vis, noise_vis)

    return vis


def augmented_im_to_vis(
    image: np.ndarray,
    uvw: np.ndarray,
    lm: Union[None, np.ndarray],
    chan_freqs: np.ndarray,
    polarisation: bool,
    expand_freq_dim: bool,
    use_dft: bool,
    ncorr: int,
    delta_ra: Optional[int] = None,
    delta_dec: Optional[int] = None,
    do_wstacking: Optional[bool] = True,
    epsilon: Optional[float] = 1e-7,
    noise: Optional[float] = None,
    nthreads: Optional[int] = 8,
):
    """
    Augmented version of im_to_vis
    Args:
        image: image array
        uvw: UVW coordinates
        lm: (l, m) coordinates (used for DFT)
        chan_freqs: frequency array
        polarisation: True if polarisation information is present, False otherwise (used for FFT)
        use_dft: True if DFT should be used, False if FFT should be used
        mode: 'add' or 'subtract' to specify whether to add or subtract model data
        mod_data: model data to/from which computed visibilities should be added/subtracted
        ncorr: number of correlations (must be 2 or 4; used for FFT)
        delta_ra: pixel size in RA direction (used for FFT)
        delta_dec: pixel size in Dec direction (used for FFT)
        epsilon: numerical precision for FFT
        noise: RMS noise
        nthreads: number of threads to use for FFT
    Returns:
        vis: visibility array
    """
    if expand_freq_dim:
        predict_nchan = 1
        predict_freqs = chan_freqs[:1]
    else:
        predict_nchan = chan_freqs.size
        predict_freqs = chan_freqs

    # if sparse, use DFT
    if use_dft:
        if polarisation:
            vis = dft_im_to_vis(image, uvw, lm, predict_freqs, convention="casa")
        else:
            image = image[..., 0]
            vis = dft_im_to_vis(image[..., np.newaxis], uvw, lm, predict_freqs, convention="casa")
            vis = stack_unpolarised_vis(vis[..., 0], ncorr)
    else:
        image = np.transpose(image, axes=(3, 2, 0, 1))
        if polarisation:
            vis = np.zeros((uvw.shape[0], predict_nchan, ncorr), dtype=np.complex128)
            for corr in range(ncorr):
                for chan in range(predict_nchan):
                    vis[:, chan, corr] = fft_im_to_vis(
                        uvw,
                        np.array([predict_freqs[chan]]),
                        image[corr, chan],
                        pixsize_x=np.abs(delta_ra),
                        pixsize_y=delta_dec,
                        epsilon=epsilon,
                        do_wstacking=do_wstacking,
                        nthreads=nthreads,
                    )
        else:
            vis = np.zeros((uvw.shape[0], predict_nchan), dtype=np.complex128)
            for chan in range(predict_nchan):
                vis[:, chan] = fft_im_to_vis(
                    uvw,
                    np.array([predict_freqs[chan]]),
                    image[0, chan],
                    pixsize_x=np.abs(delta_ra),
                    pixsize_y=delta_dec,
                    epsilon=epsilon,
                    do_wstacking=do_wstacking,
                    nthreads=nthreads,
                )

            vis = stack_unpolarised_vis(vis, ncorr)

    if expand_freq_dim:
        vis = np.repeat(vis, chan_freqs.size, axis=1)

    if noise:
        vis = add_noise(vis, noise)

    return vis
