from typing import List, Optional, Tuple, Union

import numpy as np
from africanus.dft import im_to_vis as dft_im_to_vis
from daskms import xds_from_ms, xds_from_table
from ducc0.wgridder import dirty2ms
from scabha.basetypes import MS

from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.utilities import radec2lm


def vis_noise_from_sefd_and_ms(ms: Union[MS, str], sefd: float, spw_id: int = 0, field_id: int = 0):
    """
    Compute per-visibility thermal noise from an SEFD and an MS.

    Parameters
    ----------
    ms : MS or str
        Measurement Set path.
    sefd : float
        Antenna System Equivalent Flux Density in Jy.
    spw_id : int, optional
        DATA_DESC_ID (spectral window) to use. Default is 0.
    field_id : int, optional
        FIELD_ID to filter rows. Default is 0.

    Returns
    -------
    float
        RMS noise per visibility (Jy).
    """
    spw_ds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    msds = xds_from_ms(ms, group_cols=["DATA_DESC_ID"], taql_where=f"FIELD_ID=={field_id}")[spw_id]

    df = spw_ds.CHAN_WIDTH.data[spw_id][0]
    dt = msds.EXPOSURE.data[0]
    noise_vis = sefd / np.sqrt(2 * dt * df)
    return noise_vis


def sim_noise(dshape: Union[List, Tuple], vis_noise: float) -> np.ndarray:
    """
    Simulate complex Gaussian visibility noise.

    Parameters
    ----------
    dshape : list or tuple
        Desired output shape (e.g., (nrows, nchan, ncorr)).
    vis_noise : float
        RMS per visibility (Jy).

    Returns
    -------
    numpy.ndarray
        Complex noise array of shape `dshape`.
    """
    return np.random.randn(*dshape) * vis_noise + np.random.randn(*dshape) * vis_noise * 1j


def add_noise(vis: Union[np.ndarray, float], vis_noise: float):
    """
    Add complex Gaussian noise to visibilities.

    Parameters
    ----------
    vis : numpy.ndarray or float
        Visibility data. If scalar 0, returns pure noise.
    vis_noise : float
        RMS per visibility (Jy).

    Returns
    -------
    numpy.ndarray
        Noisy visibilities.
    """
    return vis + sim_noise(vis.shape, vis_noise)


def add_to_vis(vis0: np.ndarray, vis1: np.ndarray, subtract: bool = False) -> np.ndarray:
    """
    Add or subtract two visibility arrays.

    Parameters
    ----------
    vis0 : numpy.ndarray
        Base visibility array.
    vis1 : numpy.ndarray
        Array to add or subtract.
    subtract : bool, optional
        If True, perform vis0 - vis1; otherwise vis0 + vis1. Default False.

    Returns
    -------
    numpy.ndarray
        Resulting visibility array.
    """
    if subtract:
        return vis0 - vis1
    else:
        return vis0 + vis1


def stack_unpolarised_vis(vis: np.ndarray, ncorr: int) -> np.ndarray:
    """
    Replicate unpolarised visibilities across correlation dimension.

    Parameters
    ----------
    vis : numpy.ndarray
        Array of shape (nrows, nchan) with Stokes I or single correlation.
    ncorr : int
        Number of output correlations (2 or 4).

    Returns
    -------
    numpy.ndarray
        Stacked array of shape (nrows, nchan, ncorr).

    Raises
    ------
    ValueError
        If `ncorr` is not 2 or 4.
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
    Predict visibilities via FFT gridding (dirty2ms wrapper).

    Parameters
    ----------
    uvw : numpy.ndarray
        UVW coordinates (nrows, 3) or (3, nrows) accepted by dirty2ms.
    chan_freq : numpy.ndarray
        1D array of channel frequencies (Hz).
    image : numpy.ndarray
        2D image (n_l, n_m).
    pixsize_x : float
        Pixel size along RA (radians).
    pixsize_y : float
        Pixel size along Dec (radians).
    epsilon : float, optional
        Accuracy parameter for wgridder. Default 1e-7.
    nthreads : int, optional
        Number of threads. Default 8.
    do_wstacking : bool, optional
        Enable w-stacking. Default True.

    Returns
    -------
    numpy.ndarray
        Complex visibilities of shape (nrows,).
    """
    result = dirty2ms(
        uvw,
        chan_freq,
        image,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wstacking=do_wstacking,
        nthreads=nthreads,
    )
    return np.conj(np.squeeze(result))


def compute_vis(
    skymodel: ASCIISkymodel,
    uvw: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray = None,
    ncorr: int = 2,
    polarisation: bool = False,
    linear_basis: bool = True,
    ra0: float = None,
    dec0: float = None,
    noise_vis: Optional[float] = None,
):
    """
    Compute model visibilities for an ASCII sky model.

    Parameters
    ----------
    skymodel : ASCIISkymodel
        Parsed sky model object.
    uvw : numpy.ndarray
        UVW coordinates of shape (3, nrows) or (nrows, 3).
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    times : numpy.ndarray, optional
        Time stamps per row if transient sources present.
    ncorr : int, optional
        Number of correlations (2 or 4). Default 2.
    polarisation : bool, optional
        If True, include cross-hands when available. Default False.
    linear_basis : bool, optional
        Use linear (True) or circular (False) basis. Default True.
    ra0 : float, optional
        Phase centre right ascension (radians).
    dec0 : float, optional
        Phase centre declination (radians).
    noise_vis : float, optional
        RMS noise per visibility (Jy). If provided, noise is added.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan or nchan*ntime, ncorr).

    Raises
    ------
    ValueError
        If transient sources exist and `times` is None.
    """
    wavs = 2.99e8 / freqs
    uvw_scaled = uvw.T[..., np.newaxis] / wavs

    def calculate_phase_factor(src):
        el, em = radec2lm(ra0, dec0, src.ra, src.dec)
        n_term = np.sqrt(1 - el * el - em * em) - 1
        arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
        if not src.emaj and not src.emin:
            return np.exp(2 * np.pi * 1j * arg)
        else:
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

    if skymodel.has_transient:
        if isinstance(times, type(None)):
            raise ValueError("parameter 'times' must be provided for skymodels with transient source")
        unique_times, time_index_mapper = np.unique(times, return_inverse=True)
    else:
        unique_times = time_index_mapper = None

    for source in skymodel.sources:
        phase = calculate_phase_factor(source)
        bmatrix = source.get_brightness_matrix(
            freqs,
            ncorr,
            unique_times=unique_times,
            time_index_mapper=time_index_mapper,
            linear_basis=linear_basis,
        )
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
    Predict visibilities from an image (DFT or FFT path).

    Parameters
    ----------
    image : numpy.ndarray
        Image cube:
        - DFT: shape (N_nonzero, N_freq, ncorr) or (N_nonzero, N_freq, 1).
        - FFT: shape (n_l, n_m, N_freq, ncorr).
    uvw : numpy.ndarray
        UVW coordinates (nrows, 3).
    lm : numpy.ndarray or None
        (l, m) direction cosines for DFT workflow; None for FFT.
    chan_freqs : numpy.ndarray
        Full MS frequency grid (Hz).
    polarisation : bool
        If True, use all correlations; if False, replicate Stokes I.
    expand_freq_dim : bool
        If True, single-frequency input is expanded to all MS channels.
    use_dft : bool
        True for sparse DFT mode; False for FFT mode.
    ncorr : int
        Number of correlations (2 or 4).
    delta_ra : float, optional
        Pixel size along RA (radians) for FFT.
    delta_dec : float, optional
        Pixel size along Dec (radians) for FFT.
    do_wstacking : bool, optional
        Enable w-stacking in FFT mode. Default True.
    epsilon : float, optional
        Accuracy parameter for FFT gridding. Default 1e-7.
    noise : float, optional
        RMS noise per visibility (Jy). If provided, added to output.
    nthreads : int, optional
        Threads for FFT gridding. Default 8.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan, ncorr).

    Notes
    -----
    - DFT path uses africanus.dft.im_to_vis.
    - FFT path uses ducc0.wgridder.dirty2ms per channel.
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
