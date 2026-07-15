"""Tests for simms.utilities helpers, in particular quantity_to_value, which
backs every schema-driven unit conversion in the ASCII sky model reader."""

import math

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Angle, Latitude, Longitude

from simms.exceptions import SkymodelSchemaError
from simms.utilities import (
    get_noise,
    is_numeric,
    is_range_in_range,
    quantity_to_value,
    radec2lm,
)


def test_is_numeric():
    assert is_numeric("1.5")
    assert is_numeric("-65")
    assert is_numeric("1e9")
    assert is_numeric(3)
    assert not is_numeric("4h30m")
    assert not is_numeric(None)
    assert not is_numeric("")


def test_quantity_to_value_null():
    assert quantity_to_value(Angle, None, null_value=0) == 0
    assert quantity_to_value(Angle, "null", null_value=None) is None


def test_quantity_to_value_numeric_with_units():
    # plain numbers are scaled by the schema units and converted to the target
    assert quantity_to_value(Angle, 180, "deg", target_units="rad") == pytest.approx(math.pi)
    assert quantity_to_value(Angle, "180", "deg", target_units="rad") == pytest.approx(math.pi)
    assert quantity_to_value(units.Quantity, 2.0, "GHz", target_units="Hz") == pytest.approx(2e9)


def test_quantity_to_value_numeric_without_units():
    assert quantity_to_value(units.Quantity, "8") == pytest.approx(8.0)
    assert quantity_to_value(units.Quantity, 8) == pytest.approx(8.0)


def test_quantity_to_value_string_with_units():
    # strings that carry their own units bypass the schema units
    assert quantity_to_value(Angle, "45 deg", target_units="rad") == pytest.approx(math.radians(45))
    assert quantity_to_value(Longitude, "4h8m20.38s", target_units="rad") == pytest.approx(
        math.radians((4 + 8 / 60 + 20.38 / 3600) * 15)
    )
    assert quantity_to_value(Latitude, "-65d45m9.08s", target_units="rad") == pytest.approx(
        math.radians(-(65 + 45 / 60 + 9.08 / 3600))
    )


def test_quantity_to_value_string_passthrough():
    assert quantity_to_value(str, "SrcA") == "SrcA"


def test_quantity_to_value_unknown_units():
    with pytest.raises(SkymodelSchemaError, match="parsecs_per_fortnight"):
        quantity_to_value(Angle, 1.0, "parsecs_per_fortnight")


def test_get_noise_scalar():
    assert get_noise(400.0, 8, 1e6) == pytest.approx(400.0 / math.sqrt(2 * 1e6 * 8))


def test_get_noise_per_baseline():
    sefds = [400.0, 500.0, 600.0]
    dtime, dfreq = 8, 1e6
    noises = get_noise(sefds, dtime, dfreq)
    # one noise per baseline pair, ordered as combinations(sefds, 2)
    expected = [
        math.sqrt(400.0 * 500.0 / (2 * dfreq * dtime)),
        math.sqrt(400.0 * 600.0 / (2 * dfreq * dtime)),
        math.sqrt(500.0 * 600.0 / (2 * dfreq * dtime)),
    ]
    assert noises == pytest.approx(expected)


def test_is_range_in_range():
    assert is_range_in_range((2, 3), (1, 4))
    assert not is_range_in_range((0, 3), (1, 4))
    assert is_range_in_range((1, 4), (1, 4))
    # reversed tuples are normalised before comparison
    assert is_range_in_range((3, 2), (4, 1))
    assert not is_range_in_range((5, 2), (1, 4))


def test_radec2lm():
    # at the phase centre, (l, m) is (0, 0)
    l_coord, m_coord = radec2lm(0.3, -0.5, 0.3, -0.5)
    assert l_coord == pytest.approx(0.0)
    assert m_coord == pytest.approx(0.0)

    # with the phase centre on the equator at ra0=0: l = cos(dec) sin(ra), m = sin(dec)
    ra = np.array([0.01, -0.02])
    dec = np.array([0.03, -0.04])
    l_coord, m_coord = radec2lm(0.0, 0.0, ra, dec)
    np.testing.assert_allclose(l_coord, np.cos(dec) * np.sin(ra), rtol=1e-12)
    np.testing.assert_allclose(m_coord, np.sin(dec), rtol=1e-12)

    # pure declination offset from a general phase centre moves m only
    l_coord, m_coord = radec2lm(1.0, 0.5, 1.0, 0.5 + 0.01)
    assert l_coord == pytest.approx(0.0)
    assert m_coord == pytest.approx(0.01, rel=1e-4)
