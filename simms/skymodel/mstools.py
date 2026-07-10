from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from africanus.dft import im_to_vis as dft_im_to_vis
from astropy import constants as c
from daskms import xds_from_ms, xds_from_table
from ducc0.wgridder import dirty2ms
from scabha.basetypes import MS

from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.kernels import is_uniform_grid, predict_vis
from simms.utilities import radec2lm


def vis_noise_from_sefd_and_ms(ms: MS | str, sefd: float, spw_id: int = 0, field_id: int = 0):
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
    # Reduce to a plain float here. A lazy dask scalar would otherwise be carried
    # into every predict task, where evaluating it re-opens the MS.
    noise_vis = float((sefd / np.sqrt(2 * dt * df)).compute())
    return noise_vis


def sim_noise(dshape: list | tuple, vis_noise: float, dtype: np.dtype = np.complex128) -> np.ndarray:
    """
    Simulate complex Gaussian visibility noise.

    Parameters
    ----------
    dshape : list or tuple
        Desired output shape (e.g., (nrows, nchan, ncorr)).
    vis_noise : float
        RMS per visibility (Jy).
    dtype : numpy.dtype, optional
        Complex dtype of the output. Default complex128.

    Returns
    -------
    numpy.ndarray
        Complex noise array of shape ``dshape``.
    """
    rng = np.random.default_rng()
    real_dtype = np.finfo(dtype).dtype
    # Draw the real and imaginary parts as one array and reinterpret it as
    # complex, so the noise costs a single allocation of the output size.
    noise = rng.standard_normal((*dshape, 2), dtype=real_dtype)
    noise *= vis_noise
    return noise.view(dtype).reshape(dshape)


def sim_noise_block(
    uvw: np.ndarray,
    freqs: np.ndarray,
    ncorr: int,
    vis_noise: float,
    out_dtype: np.dtype = np.complex128,
):
    """
    Noise-only visibilities for one row block.

    ``uvw`` and ``freqs`` are used only for their shapes; they give ``da.blockwise``
    the row and channel extents it cannot otherwise infer. The output dtype is
    named ``out_dtype`` because ``da.blockwise`` consumes any ``dtype`` kwarg
    itself and would never forward it.
    """
    return sim_noise((uvw.shape[0], freqs.size, ncorr), vis_noise, dtype=out_dtype)


def add_noise(vis: np.ndarray, vis_noise: float):
    """
    Add complex Gaussian noise to visibilities in place.

    Parameters
    ----------
    vis : numpy.ndarray
        Visibility data.
    vis_noise : float
        RMS per visibility (Jy).

    Returns
    -------
    numpy.ndarray
        `vis`, with noise added.
    """
    vis += sim_noise(vis.shape, vis_noise, dtype=vis.dtype)
    return vis


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


@dataclass
class PreparedSky:
    """An ASCII sky model reduced to flat arrays ready for :func:`predict_vis`.

    Built once per simulation rather than once per row block, and shared by all
    blocks. Its memory footprint is dominated by ``bmat``, which is
    ``nsrc * nspec * nchan`` complex values.
    """

    lmn: np.ndarray
    gauss_shape: np.ndarray
    is_gauss: np.ndarray
    bmat: np.ndarray
    lightcurve: np.ndarray
    unique_times: np.ndarray | None
    freqs: np.ndarray
    uniform_freqs: bool
    ncorr: int
    polarisation: bool

    @property
    def nspec(self) -> int:
        """Number of correlations actually carried through the kernel."""
        return self.bmat.shape[1]


def prepare_skymodel(
    skymodel: ASCIISkymodel,
    freqs: np.ndarray,
    ra0: float,
    dec0: float,
    ncorr: int = 2,
    polarisation: bool = False,
    linear_basis: bool = True,
    unique_times: np.ndarray = None,
    dtype: np.dtype = np.complex128,
) -> PreparedSky:
    """
    Flatten an ASCII sky model into the arrays the prediction kernel consumes.

    Parameters
    ----------
    skymodel : ASCIISkymodel
        Parsed sky model object.
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    ra0, dec0 : float
        Phase centre (radians).
    ncorr : int, optional
        Number of correlations (2 or 4). Default 2.
    polarisation : bool, optional
        If True, carry every correlation. If False, only Stokes I is predicted
        and the remaining correlations are filled in afterwards. Default False.
    linear_basis : bool, optional
        Use linear (True) or circular (False) basis. Default True.
    unique_times : numpy.ndarray, optional
        Sorted unique time stamps spanning the *whole* observation. Required if
        the model contains transient sources: the lightcurve is referenced to
        the start of the observation, not to the start of a row block.
    dtype : numpy.dtype, optional
        Complex dtype of the brightness matrix (and hence of the visibilities).

    Returns
    -------
    PreparedSky

    Raises
    ------
    ValueError
        If transient sources are present and `unique_times` is None, or if
        `ncorr` is not 2 or 4.
    """
    if ncorr not in (2, 4):
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    has_transient = skymodel.has_transient
    if has_transient and unique_times is None:
        raise ValueError("parameter 'unique_times' must be provided for skymodels with transient sources")

    sources = skymodel.sources
    nsrc = len(sources)
    nchan = freqs.size
    # Unpolarised runs only ever need Stokes I; the other correlations are
    # derived from it once, after the sources have been summed.
    nspec = ncorr if polarisation else 1
    ntime = unique_times.size if has_transient else 1

    lmn = np.zeros((nsrc, 3), dtype=np.float64)
    gauss_shape = np.zeros((nsrc, 3), dtype=np.float64)
    is_gauss = np.zeros(nsrc, dtype=np.bool_)
    bmat = np.zeros((nsrc, nspec, nchan), dtype=dtype)
    lightcurve = np.ones((nsrc, ntime), dtype=np.float64)

    for i, source in enumerate(sources):
        el, em = radec2lm(ra0, dec0, source.ra, source.dec)
        lmn[i] = el, em, np.sqrt(1 - el * el - em * em) - 1

        emaj = source.value_or_default("emaj")
        emin = source.value_or_default("emin")
        if emaj or emin:
            pa = source.value_or_default("pa")
            is_gauss[i] = True
            gauss_shape[i] = (
                emaj * np.sin(pa),
                emaj * np.cos(pa),
                emin / (1.0 if emaj == 0.0 else emaj),
            )

        bmat[i] = source.get_brightness_matrix(freqs, ncorr, linear_basis=linear_basis)[:nspec]

        if source.is_transient:
            lightcurve[i] = source.get_lightcurve(unique_times)

    return PreparedSky(
        lmn=lmn,
        gauss_shape=gauss_shape,
        is_gauss=is_gauss,
        bmat=bmat,
        lightcurve=lightcurve,
        unique_times=unique_times if has_transient else None,
        freqs=freqs,
        uniform_freqs=is_uniform_grid(freqs),
        ncorr=ncorr,
        polarisation=polarisation,
    )


def predict_block(
    prepared: PreparedSky,
    uvw: np.ndarray,
    times: np.ndarray = None,
    noise_vis: float | None = None,
    out_dtype: np.dtype = None,
) -> np.ndarray:
    """
    Predict visibilities for one block of rows.

    Parameters
    ----------
    prepared : PreparedSky
        Sky model arrays from :func:`prepare_skymodel`.
    uvw : numpy.ndarray
        UVW coordinates of shape (nrows, 3), in metres.
    times : numpy.ndarray, optional
        Time stamp per row. Required if the model contains transient sources.
    noise_vis : float, optional
        RMS noise per visibility (Jy). If provided, noise is added.
    out_dtype : numpy.dtype, optional
        Complex dtype to cast the result to. Sources are summed in the (higher)
        precision of ``prepared.bmat``, so a single-precision output column does
        not degrade the accumulation. Named ``out_dtype`` because ``da.blockwise``
        consumes any ``dtype`` kwarg itself and would never forward it.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan, ncorr).
    """
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    nrow = uvw.shape[0]
    ncorr = prepared.ncorr
    nspec = prepared.nspec

    if prepared.unique_times is None:
        time_index = np.zeros(nrow, dtype=np.int64)
    else:
        if times is None:
            raise ValueError("parameter 'times' must be provided for skymodels with transient sources")
        time_index = np.searchsorted(prepared.unique_times, times).astype(np.int64)

    vis = np.zeros((nrow, prepared.freqs.size, nspec), dtype=prepared.bmat.dtype)
    predict_vis(
        uvw,
        prepared.freqs,
        prepared.uniform_freqs,
        prepared.lmn,
        prepared.gauss_shape,
        prepared.is_gauss,
        prepared.bmat,
        prepared.lightcurve,
        time_index,
        vis,
    )

    if nspec != ncorr:
        # Unpolarised: XX == YY == Stokes I, cross-hands vanish.
        vis = stack_unpolarised_vis(vis[..., 0], ncorr)

    if noise_vis:
        vis = add_noise(vis, noise_vis)

    if out_dtype is not None:
        vis = vis.astype(out_dtype, copy=False)
    return vis


def compute_vis(
    skymodel: ASCIISkymodel,
    uvw: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray = None,
    ncorr: int = 2,
    polarisation: bool = False,
    linear_basis: bool = True,
    ra0: float | None = None,
    dec0: float | None = None,
    noise_vis: float | None = None,
    unique_times: np.ndarray = None,
    dtype: np.dtype = np.complex128,
):
    """
    Compute model visibilities for an ASCII sky model.

    Convenience wrapper that prepares the sky model and predicts a single block.
    Callers looping over row blocks should call :func:`prepare_skymodel` once and
    :func:`predict_block` per block instead: transient lightcurves are referenced
    to the first of `unique_times`, which must therefore span the whole
    observation rather than a single block.

    Parameters
    ----------
    skymodel : ASCIISkymodel
        Parsed sky model object.
    uvw : numpy.ndarray
        UVW coordinates of shape (nrows, 3), in metres.
    freqs : numpy.ndarray
        Channel centre frequencies (Hz).
    times : numpy.ndarray, optional
        Time stamps per row if transient sources are present.
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
    unique_times : numpy.ndarray, optional
        Sorted unique time stamps of the whole observation. Defaults to the
        unique values of `times`, which is only correct when `uvw` covers every
        row of the observation.
    dtype : numpy.dtype, optional
        Complex dtype of the output. Default complex128.

    Returns
    -------
    numpy.ndarray
        Visibility array of shape (nrows, nchan, ncorr).
    """
    if unique_times is None and times is not None and skymodel.has_transient:
        unique_times = np.unique(times)
    prepared = prepare_skymodel(
        skymodel,
        freqs,
        ra0,
        dec0,
        ncorr=ncorr,
        polarisation=polarisation,
        linear_basis=linear_basis,
        unique_times=unique_times,
        dtype=dtype,
    )
    return predict_block(prepared, uvw, times=times, noise_vis=noise_vis)


def fft_im_to_vis(
    uvw: np.ndarray,
    chan_freq: np.ndarray,
    image: np.ndarray,
    pixsize_x: float,
    pixsize_y: float,
    epsilon: float | None = 1e-7,
    nthreads: int | None = 8,
    do_wstacking: None | bool = True,
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


def augmented_im_to_vis(
    image: np.ndarray,
    uvw: np.ndarray,
    lm: None | np.ndarray,
    chan_freqs: np.ndarray,
    polarisation: bool,
    expand_freq_dim: bool,
    use_dft: bool,
    ncorr: int,
    ref_freq: np.ndarray | None = None,
    delta_ra: int | None = None,
    delta_dec: int | None = None,
    do_wstacking: bool | None = True,
    epsilon: float | None = 1e-7,
    noise: float | None = None,
    nthreads: int | None = 8,
    dtype: np.dtype | None = None,
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
    ref_freq : np.ndarray, optional
        One-element array containing frequency (Hz) at which the input image is defined.
        Used if `expand_freq_dim` is True.
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
        predict_freqs = ref_freq
    else:
        predict_nchan = chan_freqs.size
        predict_freqs = chan_freqs

    # determine output dtype
    if dtype is not None:
        vis_dtype = np.complex128 if np.finfo(dtype).precision == 15 else np.complex64
    else:
        vis_dtype = np.complex128 if np.finfo(image.dtype).precision == 15 else np.complex64

    # if sparse, use DFT
    if use_dft:
        if polarisation:
            vis = dft_im_to_vis(image, uvw, lm, predict_freqs, convention="casa")
        else:
            image = image[..., 0]
            vis = dft_im_to_vis(image[..., np.newaxis], uvw, lm, predict_freqs, convention="casa")
            vis = stack_unpolarised_vis(vis[..., 0], ncorr)
    # otherwise, use FFT
    else:
        image = np.transpose(image, axes=(3, 2, 0, 1))
        if polarisation:
            vis = np.zeros((uvw.shape[0], predict_nchan, ncorr), dtype=vis_dtype)
            for corr in range(ncorr):
                vis[:, :, corr] = fft_im_to_vis(
                    uvw,
                    predict_freqs,
                    image[corr],
                    pixsize_x=np.abs(delta_ra),
                    pixsize_y=delta_dec,
                    epsilon=epsilon,
                    do_wstacking=do_wstacking,
                    nthreads=nthreads,
                )
        else:
            vis = fft_im_to_vis(
                uvw,
                predict_freqs,
                image[0],
                pixsize_x=np.abs(delta_ra),
                pixsize_y=delta_dec,
                epsilon=epsilon,
                do_wstacking=do_wstacking,
                nthreads=nthreads,
            )

            vis = stack_unpolarised_vis(vis, ncorr)

    if expand_freq_dim:
        amps, phases = np.abs(vis), np.angle(vis)
        wavs = c.c.value / chan_freqs
        ref_wav = c.c.value / ref_freq[0]
        phase_scale_factors = ref_wav / wavs
        phases = phases * phase_scale_factors[np.newaxis, :, np.newaxis]
        vis = amps * np.exp(1j * phases)

    if noise:
        vis = add_noise(vis, noise)

    return vis
