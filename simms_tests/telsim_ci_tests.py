import subprocess

import pytest

from . import InitTest


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


# TODO(mukundi,galefang)
## Add more tests
