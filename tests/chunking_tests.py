"""Row-chunk sizing (`simms.skymodel.mstools.auto_row_chunks`)."""

from simms.skymodel.mstools import (
    DEFAULT_ROW_CHUNK_CAP,
    MIN_ROW_CHUNK,
    ROW_TASKS_PER_WORKER,
    auto_row_chunks,
)


def chunk_count(nrows, chunk):
    return -(-nrows // chunk)


def test_single_worker_keeps_the_cap():
    """Nothing to gain from splitting further when there is one worker."""
    assert auto_row_chunks(1_000_000, 1) == DEFAULT_ROW_CHUNK_CAP
    assert auto_row_chunks(1_000_000, 0) == DEFAULT_ROW_CHUNK_CAP


def test_short_track_is_split_across_workers():
    """The regression this guards: 76608 rows (5-min MeerKAT track) at the fixed
    10000-row default is only 8 chunks, so `--nworkers 32` could never use more
    than 8 cores."""
    nrows, nworkers = 76608, 32
    assert chunk_count(nrows, DEFAULT_ROW_CHUNK_CAP) == 8  # the old behaviour

    chunk = auto_row_chunks(nrows, nworkers)
    assert chunk == 599  # ceil(76608 / (4 * 32))
    # every worker gets work, and no more chunks than we asked for
    assert nworkers <= chunk_count(nrows, chunk) <= ROW_TASKS_PER_WORKER * nworkers


def test_long_ms_is_left_at_the_cap():
    """A long MS already has plenty of chunks; the cap still bounds memory per task."""
    chunk = auto_row_chunks(50_000_000, 32)
    assert chunk == DEFAULT_ROW_CHUNK_CAP


def test_never_exceeds_the_cap():
    """The size is only ever reduced, so peak memory per task cannot grow."""
    for nrows in (1, 1_000, 76_608, 10_000_000):
        for nworkers in (1, 4, 32, 256):
            assert auto_row_chunks(nrows, nworkers) <= DEFAULT_ROW_CHUNK_CAP


def test_respects_a_lowered_cap():
    """An explicit small --row-chunks is honoured rather than overridden upwards."""
    assert auto_row_chunks(10_000_000, 32, cap=500) == 500


def test_tiny_ms_is_not_shattered():
    """Below MIN_ROW_CHUNK the per-task overhead outweighs the work, so stop splitting."""
    chunk = auto_row_chunks(1_000, 64)
    assert chunk == MIN_ROW_CHUNK
    assert chunk_count(1_000, chunk) < 64


def test_targets_several_tasks_per_worker():
    """Enough chunks to keep the pool fed and absorb uneven task cost."""
    nrows, nworkers = 4_000_000, 16
    chunk = auto_row_chunks(nrows, nworkers, cap=10**9)
    assert chunk_count(nrows, chunk) == ROW_TASKS_PER_WORKER * nworkers


def test_zero_or_negative_cap_falls_back_to_default():
    assert auto_row_chunks(10_000_000, 1, cap=0) == DEFAULT_ROW_CHUNK_CAP
    assert auto_row_chunks(10_000_000, 1, cap=-5) == DEFAULT_ROW_CHUNK_CAP


def test_empty_selection_is_safe():
    """An empty FIELD/DDID selection must not divide by zero."""
    assert auto_row_chunks(0, 32) == DEFAULT_ROW_CHUNK_CAP
