import os.path
import shutil

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from daskms import xds_from_table
from scipy.optimize import least_squares

from simms.telescope import array_utilities
from simms.telescope.generate_ms import create_ms
from simms_tests import TESTDIR


class InitTest:
    def __init__(self):
        """Set up common test parameters before each test method runs."""
        # Common parameters used across tests
        self.ms = os.path.join(TESTDIR, "test-mk.ms")
        self.telescope = "kat-7"
        self.max_bl = 185
        self.min_bl = 26
        self.nant = 7
        self.ntimes = 10
        self.dtime = 1
        self.start_freq = "1420MHz"
        self.dfreq = "1MHz"
        self.nchan = 2
        self.direction = ["J2000", "0h0m20s", "-30deg"]
        self.start_time = "2025-03-06T12:25:00"
        self.sefd = 821
        self.column = "DATA"

        # Store temporary files to be cleaned up
        self.test_files = []
        self.make_ms()

    def __del__(self):
        """Clean up after all tests in this class."""
        # Remove any temporary files created
        for file in self.test_files:
            if os.path.exists(file):
                shutil.rmtree(file)

    def make_ms(self):
        create_ms(
            self.ms,
            telescope_name=self.telescope,
            pointing_direction=self.direction,
            dtime=self.dtime,
            ntimes=self.ntimes,
            start_freq=self.start_freq,
            dfreq=self.dfreq,
            nchan=self.nchan,
            correlations=["XX", "YY"],
            row_chunks=10000,
            sefd=self.sefd,
            column=self.column,
            start_time=self.start_time,
            smooth=None,
            fit_order=None,
        )

        self.test_files.append(self.ms)


@pytest.fixture
def params():
    return InitTest()


def test_max_bl(params):
    ds = xds_from_table(params.ms)[0]
    uvw = ds.UVW.values
    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)

    assert np.isclose(bl.max(), params.max_bl, atol=10)
    assert np.isclose(bl.min(), params.min_bl, atol=4)


def test_uv_coverage(params):
    """
    Test to check if the uv coverage shape is what is expected.
    e.g circles at the poles and lines at the equator.

    """

    array = array_utilities.Array(params.telescope)
    array_data_line = array.uvgen(
        pointing_direction=["J2000", "0deg", "0deg"],
        dtime=2,
        ntimes=500,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    line = check_circle_or_ellipse(array_data_line.uvw[::21, 0], array_data_line.uvw[::21, 1])
    assert line["is_line"] == True

    array_data_circle = array.uvgen(
        pointing_direction=["J2000", "0deg", "-90deg"],
        dtime=2,
        ntimes=5000,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    circle = check_circle_or_ellipse(array_data_circle.uvw[::21, 0], array_data_circle.uvw[::21, 1])
    assert circle["is_circle"] == True

    array_data_circle_n = array.uvgen(
        pointing_direction=["J2000", "0deg", "90deg"],
        dtime=2,
        ntimes=5000,
        start_freq="1420MHz",
        dfreq="2MHz",
        nchan=10,
    )

    circle_n = check_circle_or_ellipse(array_data_circle_n.uvw[::21, 0], array_data_circle_n.uvw[::21, 1])
    assert circle_n["is_circle"] == True

    array_data_ellipse = array.uvgen(
        pointing_direction=["J2000", "0deg", "-30deg"],
        dtime=2,
        ntimes=5000,
        start_freq=params.start_freq,
        dfreq="2MHz",
        nchan=10,
    )

    ellipse = check_circle_or_ellipse(array_data_ellipse.uvw[::21, 0], array_data_ellipse.uvw[::21, 1])
    assert ellipse["is_circle"] == False and ellipse["is_ellipse"] == True


def test_visdata_configuration_info(params):
    ds = xds_from_table(params.ms)[0]
    ds_spw = xds_from_table(f"{params.ms}::SPECTRAL_WINDOW")[0]
    num_chan = ds_spw.NUM_CHAN.values
    assert np.isclose(num_chan, 2)

    freq = ds_spw.CHAN_FREQ.values[0][0]
    assert np.isclose(freq, 1420e6)

    dfreq = ds_spw.CHAN_FREQ.values[0][1] - freq
    assert np.isclose(dfreq, 1e6)

    time = np.unique(ds.TIME).shape
    assert np.isclose(time, 10)

    dtime = ds.INTERVAL.values[0]
    assert np.isclose(dtime, 1)

    nbl = ds.DATA.shape[0] / time[0]
    tel_nbl = params.nant * (params.nant - 1) // 2
    assert np.isclose(nbl, tel_nbl)

    ds_point = xds_from_table(f"{params.ms}::POINTING")[0]
    direction = ds_point.TARGET.values[0][0]

    orig_direction = SkyCoord(*params.direction[1:])
    ra0 = orig_direction.ra.to("rad").value
    dec0 = orig_direction.dec.to("rad").value
    assert np.isclose(direction[0], ra0)
    assert np.isclose(direction[1], dec0)

    ds_pol = xds_from_table(f"{params.ms}::POLARIZATION")[0]
    corr = ds_pol.CORR_TYPE.values[0]
    assert np.isclose(corr[0], 9)
    assert np.isclose(corr[1], 12)

    ds_ant = xds_from_table(f"{params.ms}::ANTENNA")[0]
    mount = ds_ant.MOUNT.values[0]
    size = ds_ant.DISH_DIAMETER.values[0]
    assert mount == "ALT-AZ"
    assert np.isclose(size, 12)


def check_circle_or_ellipse(u, v):
    """
    Check if UV points form a circle, ellipse or line.
    """

    center_u = np.mean(u)
    center_v = np.mean(v)

    distances = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    def conic_equation(params, x, y):
        a, b, c, d, e, f = params
        return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f

    def residuals(params, x, y):
        return conic_equation(params, x, y)

    initial_guess = [1, 0, 1, 0, 0, -(mean_distance**2)]
    result = least_squares(residuals, initial_guess, args=(u, v))
    a, b, c, d, e, f = result.x

    discriminant = b**2 - 4 * a * c
    is_circle = abs(a - c) < 0.01 * abs(a) and abs(b) < 0.01 * abs(a)
    is_ellipse = discriminant < 0 and not is_circle

    coeffs = np.polyfit(u, v, 1)
    m, b = coeffs
    v_pred = m * u + b
    residual_v = v - v_pred
    residual_std = np.std(residual_v)
    ss_tot = np.sum((v - np.mean(v)) ** 2)
    ss_res = np.sum(residual_v**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    u_range = np.max(u) - np.min(u) if len(u) > 1 else 1.0
    v_std = np.std(v)
    is_horizontal_line = abs(m) < 0.01 and v_std < 0.05 * u_range
    is_line = (r_squared > 0.95 and residual_std < 0.05 * np.std(np.sqrt(u**2 + v**2))) or is_horizontal_line

    return {
        "is_circle": is_circle,
        "is_ellipse": is_ellipse,
        "is_line": is_line,
    }
