from typing import Optional, Union, List
import numpy as np
from africanus.dft import im_to_vis as dft_im_to_vis
from ducc0.wgridder import dirty2ms
from simms.skymodel.source_factory import Source, compute_brightness_matrix

def add_noise(vis:np.ndarray, noise_vis:float):
    """AI is creating summary for add_noise

    Args:
        vis (np.ndarray): [description]
        noise_vis (float): Noise per visibility
    """
    noise =  np.random.randn(vis.shape)*noise_vis + np.random.randn(vis.shape)*noise_vis*1j
    return vis +  noise 

def add_to_vis(vis0:np.ndarray, vis1:np.ndarray, subtract:bool=False) -> np.ndarray:
    """ Add/subtract two visibility arrays

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


def fft_im_to_vis(uvw: np.ndarray, chan_freq: np.ndarray, image: np.ndarray, pixsize_x: float, pixsize_y: float,
                epsilon: Optional[float]=1e-7, nthreads: Optional[int]=8, do_wstacking: Optional[bool]=True) -> np.ndarray:
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
        nthreads=nthreads
    )
    
    return np.conj(np.squeeze(result))



def compute_vis(srcs: List[Source], uvw: np.ndarray, freqs: np.ndarray, ncorr: int, polarisation: bool, basis: str,
                mode: Union[None, str], input_vis: Union[None, np.ndarray], noise: Optional[float] = None):
    """
    Computes visibilities

    Args:
        srcs (list):                List of Source objects
        uvw (numpy.ndarray):        Array of shape (3, nrows) containing the UVW coordinates
        freqs (numpy.ndarray):      Array of shape (nchan,) containing the frequencies
        ncorr (int):                Number of correlations
        polarisation (bool):        True if polarisation information is present, False otherwise
        basis (str):                Polarisation basis ("linear" or "circular")
        mod_data (numpy.ndarray):   Array of shape (nrows, nchan, ncorr) containing the model data 
            to/from which computed visibilities should be added/subtracted
        noise (float):              RMS noise
        subtract (bool):            True if visibilities should be subtracted from the model data, False otherwise

    Returns:
        vis (numpy.ndarray):        Visibility array of shape (nrows, nchan, ncorr)
    """

    wavs = 2.99e8 / freqs
    uvw_scaled = uvw.T[...,np.newaxis] / wavs
    
    # helper function to calculate phase factor
    def calculate_phase_factor(source, uvw_scaled):
        el, em = source.l, source.m
        n_term = np.sqrt(1 - el*el - em*em) - 1
        arg = uvw_scaled[0] * el + uvw_scaled[1] * em + uvw_scaled[2] * n_term
        
        if source.emaj in [None, "null"] and source.emin in [None, "null"]:
            # point source
            return np.exp(2 * np.pi * 1j * arg)
        else:
            # extended source
            ell = source.emaj * np.sin(source.pa)
            emm = source.emaj * np.cos(source.pa)
            ecc = source.emin / (1.0 if source.emaj == 0.0 else source.emaj)
            
            fu1 = (uvw_scaled[0]*emm - uvw_scaled[1]*ell) * ecc
            fv1 = (uvw_scaled[0]*ell + uvw_scaled[1]*emm)
            
            shape_phase = fu1 * fu1 + fv1 * fv1
            return np.exp(2 *np.pi * 1j * arg - shape_phase)
    
    # if polarisation is detected, we need to compute different correlations separately
    if polarisation:
        xx, yy = 0j, 0j
        if ncorr == 2:  # if ncorr is 2, we only need compute XX and YY correlations
            for source in srcs:
                phase_factor = calculate_phase_factor(source, uvw_scaled)
                source_xx, source_yy = compute_brightness_matrix(source.spectrum, 'diagonal', basis)
                xx += source_xx * phase_factor
                yy += source_yy * phase_factor
                
            vis = np.stack([xx, yy], axis=2)

        elif ncorr == 4:  # if ncorr is 4, we need to compute all correlations
            xy, yx = 0j, 0j
            for source in srcs:
                phase_factor = calculate_phase_factor(source, uvw_scaled)
                source_xx, source_xy, source_yx, source_yy = compute_brightness_matrix(source.spectrum, 'all', basis)
                xx += source_xx * phase_factor
                xy += source_xy * phase_factor
                yx += source_yx * phase_factor
                yy += source_yy * phase_factor
            
            vis = np.stack([xx, xy, yx, yy], axis=2)

        else:
            raise ValueError(
                f"Only two or four correlations allowed, but {ncorr} were requested."
            )

    # if no polarisation is detected, we only need compute XX and duplicate to YY
    else:
        vis = 0j
        for source in srcs:
            phase_factor = calculate_phase_factor(source, uvw_scaled)
            vis += source.spectrum * phase_factor
            
        vis = stack_unpolarised_vis(vis, ncorr)
    
    if noise:
        vis = add_noise(vis, noise)
    if isinstance(input_vis, np.ndarray):
        vis = add_to_vis(input_vis, vis, subtract= mode == "subtract")
    
    return vis
    
    
def augmented_im_to_vis(image: np.ndarray, uvw: np.ndarray, lm: Union[None, np.ndarray], chan_freqs: np.ndarray,
                        polarisation: bool, use_dft: bool, mode: Union[None, str], input_vis: Union[None, np.ndarray],
                        ncorr: int, delta_ra: Optional[int]=None, delta_dec: Optional[int]=None, do_wstacking: Optional[bool]=True,
                        epsilon: Optional[float]=1e-7, noise: Optional[float]=None, nthreads: Optional[int]=8,):
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
    # if sparse, use DFT
    if use_dft:
        if polarisation:
            vis = dft_im_to_vis(image, uvw, lm, chan_freqs, convention='casa')
        else:
            image = image[..., 0]
            vis = dft_im_to_vis(image[..., np.newaxis], uvw, lm, chan_freqs, convention='casa')
            vis = stack_unpolarised_vis(vis[...,0], ncorr)

    # else, use FFT
    else:
        image = np.transpose(image, axes=(3, 2, 0, 1))
        if polarisation:
            vis = np.zeros((uvw.shape[0], chan_freqs.size, ncorr), dtype=np.complex128)
            for corr in range(ncorr):
                for chan in range(chan_freqs.size):
                    vis[:, chan, corr] = fft_im_to_vis(
                        uvw,
                        np.array([chan_freqs[chan]]),
                        image[corr, chan],
                        pixsize_x=np.abs(delta_ra),
                        pixsize_y=delta_dec,
                        epsilon=epsilon,
                        do_wstacking=do_wstacking,
                        nthreads=nthreads
                    )
        else:
            vis = np.zeros((uvw.shape[0], chan_freqs.size), dtype=np.complex128)
            for chan in range(chan_freqs.size):
                vis[:, chan] = fft_im_to_vis(
                    uvw,
                    np.array([chan_freqs[chan]]),
                    image[0, chan],
                    pixsize_x=np.abs(delta_ra),
                    pixsize_y=delta_dec,
                    epsilon=epsilon,
                    do_wstacking=do_wstacking,
                    nthreads=nthreads
                )
            
            vis = stack_unpolarised_vis(vis, ncorr)

    if noise:
        vis = add_noise(vis, noise)
    if isinstance(input_vis, np.ndarray):
        vis = add_to_vis(input_vis, vis, subtract= mode == "subtract")

    return vis
