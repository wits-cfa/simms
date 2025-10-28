import subprocess

import pytest

from . import TESTDIR, InitTest


@pytest.fixture
def params():
    return InitTest()


def test_telsim_defaults(params):
    subprocess.check_call(
        [
            "simms",
            "telsim",
            "--telescope",
            "kat-7",
            params.random_named_directory(suffix=".ms"),
        ]
    )


def test_telsim_subarray(params):
    subprocess.check_call(
        [
            "simms",
            "telsim",
            "--telescope",
            "skamid-aastar",
            "--ntime",
            "1",
            params.random_named_directory(suffix=".ms"),
        ]
    )


def test_telsim_skysim_chain(params):
    main_args = [
        "simms",
        "--chain",
        "--ms",
        params.random_named_directory(suffix=".ms"),
    ]

    telsim_args = [
        "telsim",
        "--telescope",
        "kat-7",
    ]

    simms_args = [
        "skysim",
        "--ascii-sky",
        f"{TESTDIR}/testsky.txt",
    ]

    subprocess.check_call(main_args + telsim_args + simms_args)


def test_error_telsim_chain_double_ms_spec(params):
    ms = params.random_named_directory(suffix=".ms")
    main_args = [
        "simms",
        "--chain",
        "--ms",
        ms,
    ]

    telsim_args = [
        "telsim",
        "--telescope",
        "kat-7",
    ]

    # this should work
    subprocess.check_output(main_args + telsim_args)

    # now it should fail because the MS is being specified when chained is true
    with pytest.raises(subprocess.CalledProcessError) as exception:
        subprocess.check_call(main_args + telsim_args + [ms])

    assert exception.type is subprocess.CalledProcessError


# TODO(mukundi,galefang)
## Add more tests
