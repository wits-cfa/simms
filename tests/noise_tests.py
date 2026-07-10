"""Tests for the seeded thermal-noise array."""

import numpy as np

from simms.skymodel.mstools import noise_visibilities


def test_seed_makes_noise_reproducible():
    shape, chunks = (200, 8, 2), (100, 4, 2)
    a = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=7).compute()
    b = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=7).compute()
    c = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=8).compute()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_statistics_and_dtype():
    shape, chunks = (4000, 16, 2), (1000, 16, 2)
    sigma = 0.3
    noise = noise_visibilities(shape, chunks, sigma, np.complex64, seed=1).compute()
    assert noise.dtype == np.complex64
    np.testing.assert_allclose(noise.real.std(), sigma, rtol=0.05)
    np.testing.assert_allclose(noise.imag.std(), sigma, rtol=0.05)
    assert abs(noise.mean()) < 0.02
