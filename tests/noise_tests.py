"""Tests for the seeded thermal-noise array."""

import numpy as np

from simms.skymodel.mstools import add_noise, noise_visibilities, sim_noise


def test_seed_makes_noise_reproducible():
    shape, chunks = (200, 8, 2), (100, 4, 2)
    a = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=7).compute()
    b = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=7).compute()
    c = noise_visibilities(shape, chunks, 0.5, np.complex64, seed=8).compute()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seeded_noise_depends_on_the_chunking():
    """Pins the actual reproducibility contract, which is weaker than "same seed, same array".

    ``dask.array.random`` spawns a bit generator per chunk, keyed to that chunk's position in
    the grid, so rechunking re-keys every block and changes the whole realisation. That is not
    a bug, but it is easy to assume otherwise -- the docstring claimed chunk-independence for
    a while -- so assert the dependence explicitly rather than leaving it accidental.
    """
    shape, sigma = (200, 8, 2), 0.5
    coarse = noise_visibilities(shape, (200, 8, 2), sigma, np.complex64, seed=7).compute()
    fine = noise_visibilities(shape, (50, 8, 2), sigma, np.complex64, seed=7).compute()

    assert not np.array_equal(coarse, fine)
    # Not even the leading rows survive: the first chunk is re-keyed too, so this is a
    # different realisation rather than the same stream cut at different offsets.
    assert not np.array_equal(coarse[:50], fine[:50])
    # Both are still valid noise, so a rechunked rerun is wrong only if you needed the
    # *same* realisation.
    np.testing.assert_allclose(fine.real.std(), sigma, rtol=0.1)


def test_noise_statistics_and_dtype():
    shape, chunks = (4000, 16, 2), (1000, 16, 2)
    sigma = 0.3
    noise = noise_visibilities(shape, chunks, sigma, np.complex64, seed=1).compute()
    assert noise.dtype == np.complex64
    np.testing.assert_allclose(noise.real.std(), sigma, rtol=0.05)
    np.testing.assert_allclose(noise.imag.std(), sigma, rtol=0.05)
    assert abs(noise.mean()) < 0.02


def test_sim_noise_is_unseeded_by_default():
    """The default stays fresh entropy, so two calls must differ."""
    a = sim_noise((64, 4, 2), 0.5)
    b = sim_noise((64, 4, 2), 0.5)
    assert not np.array_equal(a, b)


def test_sim_noise_seed_makes_it_reproducible():
    a = sim_noise((64, 4, 2), 0.5, seed=11)
    b = sim_noise((64, 4, 2), 0.5, seed=11)
    c = sim_noise((64, 4, 2), 0.5, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sim_noise_shared_generator_gives_independent_blocks():
    """The documented way to get reproducible *and* independent per-block noise.

    A repeated integer seed gives every block the same realisation, which is the trap; one
    shared Generator advances its state across calls, so blocks differ but the whole
    sequence replays from the same seed.
    """
    shape = (64, 4, 2)
    repeated = [sim_noise(shape, 0.5, seed=3) for _ in range(2)]
    assert np.array_equal(*repeated), "a repeated integer seed should repeat the realisation"

    rng = np.random.default_rng(3)
    blocks = [sim_noise(shape, 0.5, seed=rng) for _ in range(2)]
    assert not np.array_equal(*blocks)

    rng_again = np.random.default_rng(3)
    replay = [sim_noise(shape, 0.5, seed=rng_again) for _ in range(2)]
    assert all(np.array_equal(x, y) for x, y in zip(blocks, replay, strict=True))


def test_add_noise_threads_the_seed():
    vis = np.zeros((32, 4, 2), dtype=np.complex128)
    a = add_noise(vis.copy(), 0.5, seed=5)
    b = add_noise(vis.copy(), 0.5, seed=5)
    c = add_noise(vis.copy(), 0.5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
