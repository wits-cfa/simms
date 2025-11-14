import math
import os

import pytest
from astropy.coordinates import Angle, Latitude, Longitude
from omegaconf import OmegaConf

from simms.exceptions import ASCIISkymodelError, ASCIISourceError
from simms.skymodel.ascii_skies import (
    ASCIISkymodel,
    ASCIISource,
    ASCIISourceSchema,
)
from simms.skymodel.source_factory import SourceType

from . import InitTest


class InitThisTest(InitTest):
    def __init__(self):
        # important to have this first, else files/dirs created before it will not be tracked
        self.test_files = []

    def write_temp_file(self, content: str, suffix: str = ".txt") -> str:
        path = self.random_named_file(suffix=suffix)
        with open(path, "w") as f:
            f.write(content)
        return path


@pytest.fixture
def params():
    return InitThisTest()


def load_default_schema() -> ASCIISourceSchema:
    """Load the default schema via ASCIISkymodel helper but without reading a file of sources."""
    # Create a tiny valid model file to force schema load, then ignore parsing
    from simms import SCHEMADIR

    schema_path = os.path.join(SCHEMADIR, "source_schema.yaml")
    schema = ASCIISourceSchema(**OmegaConf.load(schema_path))
    return schema


def source_type_validation_test():
    src = SourceType(requires=["ra", "dec", "a|b|c", "e|f"])
    with pytest.raises(ASCIISourceError):
        src.is_valid(["ra", "dec", "a"], raise_exception=True)

    assert not src.is_valid(["ra", "dec", "a"])
    assert src.is_valid(["ra", "dec", "a", "e"])


def test_set_source_param_conversions():
    schema = load_default_schema()
    src = ASCIISource(schema)

    # RA/Dec should convert to radians using schema units (deg)
    src.set_source_param("ra", "180")
    src.set_source_param("dec", "-30")
    src.set_source_param("stokes_i", "1.5")
    src.set_source_param("name", "SrcA")

    assert pytest.approx(src.ra, rel=0, abs=1e-12) == math.pi
    assert pytest.approx(src.dec, rel=0, abs=1e-12) == Latitude("-30 deg").to("rad").value
    assert pytest.approx(src.stokes_i, rel=0, abs=1e-12) == 1.5
    assert src.name == "SrcA"


def test_alias_mapper():
    schema = load_default_schema()
    src = ASCIISource(schema)

    a2f = src.alias_to_field_mapper()

    # Default schema has no aliases set, so mapping is identity
    for key in ["ra", "dec", "stokes_i", "name"]:
        assert a2f[key] == key


def test_finalise_point_vs_extended_and_polarisation():
    schema = load_default_schema()

    # Point, unpolarised
    s1 = ASCIISource(schema)
    for k, v in {"ra": "0", "dec": "0", "stokes_i": 1.0}.items():
        s1.set_source_param(k, v)
    s1.finalise()
    assert s1.is_point is True
    assert s1.is_polarised is False

    # Extended, polarised (I and Q specified)
    s2 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 2.0, "stokes_q": 0.5, "emaj": "0.1", "emin": "0.05"}.items():
        s2.set_source_param(k, v)
    s2.finalise()
    assert s2.is_point is False
    assert s2.is_polarised is True


def test_circular_polarisation_errors():
    schema = load_default_schema()
    # Circular polarisation polarised (I, Q, U specified)
    src = ASCIISource(schema)
    for k, v in {
        "ra": 0,
        "dec": 0,
        "stokes_i": 2.0,
        "stokes_q": 0.5,
        "stokes_u": 0.1,
        "stokes_v": 0.01,
        "emaj": "0.1",
        "emin": "0.05",
    }.items():
        src.set_source_param(k, v)
    src.finalise()

    freqs = [1e6, 1.2e6, 1.3e6]
    bmatrix = src.get_brightness_matrix(chan_freqs=freqs, ncorr=2, linear_basis=True)
    assert pytest.approx(src.stokes.I, abs=1e-6) == src.stokes_i

    # check as if basis was circular. This should fail
    with pytest.raises(AssertionError):
        assert pytest.approx(src.stokes.V, abs=1e-6) == src.stokes_q
        assert pytest.approx(src.stokes.Q, abs=1e-6) == src.stokes_u
        assert pytest.approx(src.stokes.U, abs=1e-6) == src.stokes_v
    assert bmatrix.shape == (2, 3)

    # set correct basis, it should work now
    bmatrix = src.get_brightness_matrix(chan_freqs=freqs, ncorr=4, linear_basis=False)
    assert pytest.approx(src.stokes.V, abs=1e-6) == src.stokes_q
    assert pytest.approx(src.stokes.Q, abs=1e-6) == src.stokes_u
    assert pytest.approx(src.stokes.U, abs=1e-6) == src.stokes_v
    assert bmatrix.shape == (4, 3)


def test_finalise_transient_field_validation():
    schema = load_default_schema()

    # No transient fields -> not transient
    s1 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 1.0}.items():
        s1.set_source_param(k, v)
    s1.finalise()
    assert s1.is_transient is False

    # Partial transient fields -> error (ensure time values include units since schema has null units)
    s2 = ASCIISource(schema)
    for k, v in {"ra": 0, "dec": 0, "stokes_i": 1.0, "transient_start": "0 s", "transient_period": "10 s"}.items():
        s2.set_source_param(k, v)
    with pytest.raises(ASCIISourceError):
        s2.finalise()

    # All transient fields -> transient True
    s3 = ASCIISource(schema)
    for k, v in {
        "ra": 0,
        "dec": 0,
        "stokes_i": 1.0,
        "transient_start": "0 s",
        "transient_period": "10 s",
        "transient_ingress": "2 s",
        "transient_absorb": 0.1,
    }.items():
        s3.set_source_param(k, v)
    s3.finalise()
    assert s3.is_transient is True


def test_skymodel_header_required_and_format_errors(params):
    # Missing #format: line
    bad1 = params.write_temp_file("ra dec stokes_i\n0 0 1\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad1)

    # Missing required headers
    bad2 = params.write_temp_file("#format: ra dec\n0 0\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad2)

    # Row length mismatch
    bad3 = params.write_temp_file("#format: ra dec stokes_i\n0 0\n")
    with pytest.raises(ASCIISkymodelError):
        ASCIISkymodel(bad3)


def test_skymodel_parsing_space_delimited(params):
    content = """
#format: ra dec stokes_i name
0 0 1 Src1
# a comment that should be skipped
180 -30 2.5 Src2
""".strip()
    path = params.write_temp_file(content)

    model = ASCIISkymodel(path)
    assert len(model.sources) == 2

    s1, s2 = model.sources
    assert pytest.approx(s1.ra, abs=1e-12) == 0.0
    assert pytest.approx(s1.dec, abs=1e-12) == 0.0
    assert pytest.approx(s1.stokes_i, abs=1e-12) == 1.0
    assert s1.name == "Src1"

    assert pytest.approx(s2.ra, abs=1e-12) == math.pi
    assert pytest.approx(s2.dec, abs=1e-12) == Angle("-30 deg").to("rad").value
    assert pytest.approx(s2.stokes_i, abs=1e-12) == 2.5
    assert s2.name == "Src2"


def test_skymodel_parsing_csv_with_alias(params):
    # Custom schema with aliases (avoid tabs/spaces in YAML by building lines explicitly)
    schema_yaml = "\n".join(
        [
            "info: Test schema",
            "parameters:",
            "  ra:",
            "    info: Right ascension",
            "    alias: RA",
            "    units: deg",
            "    ptype: longitude",
            "    required: true",
            "  dec:",
            "    info: Declination",
            "    alias: DEC",
            "    units: deg",
            "    ptype: latitude",
            "    required: true",
            "  stokes_i:",
            "    info: Stokes I",
            "    alias: I",
            "    units: Jy",
            "    ptype: flux",
            "    required: true",
            "  name:",
            "    info: Name",
            "    alias: NAME",
            "    units: null",
            "    ptype: string",
        ]
    )
    schema_path = params.write_temp_file(schema_yaml, suffix=".yaml")

    csv_content = "\n".join(
        [
            "#format: RA,DEC,I,NAME",
            "0,0,1.0,S0",
            "45,-45,0.5,S1",
        ]
    )
    data_path = params.write_temp_file(csv_content, suffix=".csv")

    model = ASCIISkymodel(data_path, delimiter=",", source_schema_file=schema_path)
    assert len(model.sources) == 2

    s0, s1 = model.sources
    assert pytest.approx(s0.ra, abs=1e-12) == 0.0
    assert pytest.approx(s0.dec, abs=1e-12) == 0.0
    assert pytest.approx(s0.stokes_i, abs=1e-12) == 1.0
    assert s0.name == "S0"

    assert pytest.approx(s1.ra, abs=1e-12) == Longitude("45 deg").to("rad").value
    assert pytest.approx(s1.dec, abs=1e-12) == Latitude("-45 deg").to("rad").value
    assert pytest.approx(s1.stokes_i, abs=1e-12) == 0.5
    assert s1.name == "S1"
