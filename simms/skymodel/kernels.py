"""Numba kernels for predicting visibilities from a discrete sky model.

The kernels accumulate directly into the output visibility buffer, so a
prediction over ``nsrc`` sources allocates no per-source temporaries.

On a uniformly spaced channel grid the phase ``2*pi*(u.l + v.m + w.n)*nu/c``
is linear in ``nu``, so the per-channel phasor is a fixed rotation of the
previous one. That replaces one ``sincos`` per channel with one complex
multiply. The rotation is renormalised every ``RENORM_INTERVAL`` channels to
stop the modulus drifting away from unity.
"""

import numpy as np
from numba import njit

from simms.constants import PI, C

TWO_PI = 2.0 * PI

# Channels between renormalisations of the recurrence phasor.
RENORM_INTERVAL = 256

_JIT = dict(cache=True, nogil=True, fastmath=True)


@njit(inline="always", **_JIT)
def _accumulate_point_uniform(vis_row, bmat_s, base, f0, df, nchan, nspec, amp):
    """Point source, uniform channel grid: one sincos, then a rotation per channel."""
    re = np.cos(base * f0)
    im = np.sin(base * f0)
    cos_d = np.cos(base * df)
    sin_d = np.sin(base * df)

    for f in range(nchan):
        phasor = amp * (re + 1j * im)
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor

        re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
        if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
            inv = 1.0 / np.sqrt(re * re + im * im)
            re *= inv
            im *= inv


@njit(inline="always", **_JIT)
def _accumulate_point_general(vis_row, bmat_s, base, freqs, nchan, nspec, amp):
    """Point source, arbitrary channel grid."""
    for f in range(nchan):
        phase = base * freqs[f]
        phasor = amp * (np.cos(phase) + 1j * np.sin(phase))
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor


@njit(inline="always", **_JIT)
def _accumulate_gaussian(vis_row, bmat_s, base, freqs, nchan, nspec, amp, gauss_arg):
    """Gaussian source. The envelope exp(-gauss_arg * (nu/c)**2) needs a real
    exponential per channel, which dominates, so no rotation trick here."""
    for f in range(nchan):
        scale = freqs[f] / C
        envelope = amp * np.exp(-gauss_arg * scale * scale)
        phase = base * freqs[f]
        phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor


@njit(**_JIT)
def predict_vis(uvw, freqs, uniform, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index, vis):
    """
    Accumulate model visibilities into ``vis``.

    Parameters
    ----------
    uvw : (nrow, 3) float64
        Baseline coordinates in metres.
    freqs : (nchan,) float64
        Channel centre frequencies in Hz.
    uniform : bool
        True if ``freqs`` is uniformly spaced, enabling the rotation recurrence.
    lmn : (nsrc, 3) float64
        Direction cosines ``l``, ``m`` and ``n - 1`` per source.
    gauss_shape : (nsrc, 3) float64
        Per-source ``ell``, ``emm``, ``ecc`` describing the Gaussian envelope.
    is_gauss : (nsrc,) bool
        Whether each source has a non-zero extent.
    bmat : (nsrc, nspec, nchan) complex
        Brightness matrix per source and correlation, excluding any lightcurve.
    lightcurve : (nsrc, ntime) float64
        Time-dependent flux scaling. All ones (and ``ntime == 1``) when the model
        holds no transients.
    time_index : (nrow,) int64
        Index into the time axis of ``lightcurve`` for each row.
    vis : (nrow, nchan, nspec) complex
        Output buffer, accumulated into.
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]
    nspec = vis.shape[2]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C

            if is_gauss[s]:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                _accumulate_gaussian(vis_row, bmat[s], base, freqs, nchan, nspec, amp, fu1 * fu1 + fv1 * fv1)
            elif uniform:
                _accumulate_point_uniform(vis_row, bmat[s], base, f0, df, nchan, nspec, amp)
            else:
                _accumulate_point_general(vis_row, bmat[s], base, freqs, nchan, nspec, amp)

    return vis


@njit(**_JIT)
def predict_vis_beam(
    uvw,
    freqs,
    uniform,
    lmn,
    gauss_shape,
    is_gauss,
    bmat,
    lightcurve,
    time_index,
    vis,
    antenna1,
    antenna2,
    ant_type,
    beam_grid,
    pa_lo,
    pa_wt,
    corr_feed_p,
    corr_feed_q,
):
    """Accumulate model visibilities with a per-antenna primary beam applied.

    Like :func:`predict_vis`, but each correlation ``c`` of source ``s`` is scaled by
    ``g_p[fp(c)] * conj(g_q[fq(c)])``, where ``g_p``/``g_q`` are the interpolated feed
    voltages of the two antennas on the baseline.

    Extra parameters
    ----------------
    antenna1, antenna2 : (nrow,) int
        Antenna indices per row.
    ant_type : (nant,) int
        Beam-type index per antenna, indexing the first axis of ``beam_grid``.
    beam_grid : (ntype, n_pa, nsrc, nchan, 2) complex
        Feed voltages sampled on the parallactic-angle grid (last axis 0=H, 1=V).
    pa_lo : (nrow,) int
        Lower PA-grid index bracketing each row's timestamp.
    pa_wt : (nrow,) float
        Interpolation weight in ``[0, 1]`` toward ``pa_lo + 1``.
    corr_feed_p, corr_feed_q : (ncorr,) int
        Feed index (0=H, 1=V) of the first/second antenna for each correlation.
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]
    ncorr = vis.shape[2]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]
        tp = ant_type[antenna1[r]]
        tq = ant_type[antenna2[r]]
        k = pa_lo[r]
        wt = pa_wt[r]

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C

            gaussian = is_gauss[s]
            if gaussian:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                gauss_arg = fu1 * fu1 + fv1 * fv1

            re = np.cos(base * f0)
            im = np.sin(base * f0)
            cos_d = np.cos(base * df)
            sin_d = np.sin(base * df)

            for f in range(nchan):
                if gaussian:
                    scale = freqs[f] / C
                    envelope = amp * np.exp(-gauss_arg * scale * scale)
                    phase = base * freqs[f]
                    phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
                elif uniform:
                    phasor = amp * (re + 1j * im)
                else:
                    phase = base * freqs[f]
                    phasor = amp * (np.cos(phase) + 1j * np.sin(phase))

                # Linearly interpolate the two feed voltages of each antenna in PA.
                gp0 = beam_grid[tp, k, s, f, 0] * (1.0 - wt) + beam_grid[tp, k + 1, s, f, 0] * wt
                gp1 = beam_grid[tp, k, s, f, 1] * (1.0 - wt) + beam_grid[tp, k + 1, s, f, 1] * wt
                gq0 = beam_grid[tq, k, s, f, 0] * (1.0 - wt) + beam_grid[tq, k + 1, s, f, 0] * wt
                gq1 = beam_grid[tq, k, s, f, 1] * (1.0 - wt) + beam_grid[tq, k + 1, s, f, 1] * wt

                for c in range(ncorr):
                    gpc = gp0 if corr_feed_p[c] == 0 else gp1
                    gqc = gq0 if corr_feed_q[c] == 0 else gq1
                    vis_row[f, c] += bmat[s, c, f] * phasor * gpc * np.conj(gqc)

                if uniform and not gaussian:
                    re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
                    if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
                        inv = 1.0 / np.sqrt(re * re + im * im)
                        re *= inv
                        im *= inv

    return vis


@njit(**_JIT)
def predict_vis_jones(
    uvw,
    freqs,
    uniform,
    lmn,
    gauss_shape,
    is_gauss,
    bmat,
    lightcurve,
    time_index,
    vis,
    antenna1,
    antenna2,
    ant_type,
    beam_grid,
    pa_lo,
    pa_wt,
):
    """Accumulate visibilities with a full 2x2 Jones primary beam.

    Like :func:`predict_vis_beam`, but applies the complete E-Jones per source:
    ``V_pq = (E_p · B · E_q^H) · phasor`` with 2x2 complex matrices, where ``B`` is the
    source coherency ``[[XX, XY], [YX, YY]]`` from ``bmat[s, :, f]`` (linear feed basis)
    and ``E_p``/``E_q`` are the PA-interpolated beam Jones. The basis transform to the MS
    correlation frame is already folded into ``beam_grid`` (see ``build_beam_grid_jones``).
    Requires ``ncorr == 4``.

    beam_grid : (ntype, n_pa, nsrc, nchan, 2, 2) complex
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]
        tp = ant_type[antenna1[r]]
        tq = ant_type[antenna2[r]]
        k = pa_lo[r]
        wt = pa_wt[r]
        wt0 = 1.0 - wt

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C

            gaussian = is_gauss[s]
            if gaussian:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                gauss_arg = fu1 * fu1 + fv1 * fv1

            re = np.cos(base * f0)
            im = np.sin(base * f0)
            cos_d = np.cos(base * df)
            sin_d = np.sin(base * df)

            for f in range(nchan):
                if gaussian:
                    scale = freqs[f] / C
                    envelope = amp * np.exp(-gauss_arg * scale * scale)
                    phase = base * freqs[f]
                    phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
                elif uniform:
                    phasor = amp * (re + 1j * im)
                else:
                    phase = base * freqs[f]
                    phasor = amp * (np.cos(phase) + 1j * np.sin(phase))

                # PA-interpolated 2x2 Jones for each antenna.
                ep00 = beam_grid[tp, k, s, f, 0, 0] * wt0 + beam_grid[tp, k + 1, s, f, 0, 0] * wt
                ep01 = beam_grid[tp, k, s, f, 0, 1] * wt0 + beam_grid[tp, k + 1, s, f, 0, 1] * wt
                ep10 = beam_grid[tp, k, s, f, 1, 0] * wt0 + beam_grid[tp, k + 1, s, f, 1, 0] * wt
                ep11 = beam_grid[tp, k, s, f, 1, 1] * wt0 + beam_grid[tp, k + 1, s, f, 1, 1] * wt
                eq00 = beam_grid[tq, k, s, f, 0, 0] * wt0 + beam_grid[tq, k + 1, s, f, 0, 0] * wt
                eq01 = beam_grid[tq, k, s, f, 0, 1] * wt0 + beam_grid[tq, k + 1, s, f, 0, 1] * wt
                eq10 = beam_grid[tq, k, s, f, 1, 0] * wt0 + beam_grid[tq, k + 1, s, f, 1, 0] * wt
                eq11 = beam_grid[tq, k, s, f, 1, 1] * wt0 + beam_grid[tq, k + 1, s, f, 1, 1] * wt

                # B (coherency) and M = E_p @ B.
                b00 = bmat[s, 0, f]
                b01 = bmat[s, 1, f]
                b10 = bmat[s, 2, f]
                b11 = bmat[s, 3, f]
                m00 = ep00 * b00 + ep01 * b10
                m01 = ep00 * b01 + ep01 * b11
                m10 = ep10 * b00 + ep11 * b10
                m11 = ep10 * b01 + ep11 * b11

                # V = (M @ E_q^H) * phasor;  E_q^H = conj(transpose(E_q)).
                cq00 = np.conj(eq00)
                cq01 = np.conj(eq01)
                cq10 = np.conj(eq10)
                cq11 = np.conj(eq11)
                vis_row[f, 0] += (m00 * cq00 + m01 * cq01) * phasor
                vis_row[f, 1] += (m00 * cq10 + m01 * cq11) * phasor
                vis_row[f, 2] += (m10 * cq00 + m11 * cq01) * phasor
                vis_row[f, 3] += (m10 * cq10 + m11 * cq11) * phasor

                if uniform and not gaussian:
                    re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
                    if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
                        inv = 1.0 / np.sqrt(re * re + im * im)
                        re *= inv
                        im *= inv

    return vis


def is_uniform_grid(freqs: np.ndarray, rtol: float = 1e-9) -> bool:
    """True if ``freqs`` is uniformly spaced to within ``rtol`` of the channel width."""
    if freqs.size < 3:
        return True
    steps = np.diff(freqs)
    return bool(np.all(np.abs(steps - steps[0]) <= rtol * np.abs(steps[0])))
