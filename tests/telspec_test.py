from simms.telescope.teldef import Observation
from simms.config_spec import validate



print("Testing Antenna API")

observe = Observation(
    ms="testvis.ms",
    antennas="kat-7",
    direction=["J2000", "0deg", "-30deg"],
    ntimes=20,
    dtime=2,
    start_freq="1GHz",
    dfreq="10MHz",
    nchan=5,
    )

validate(observe)
print()