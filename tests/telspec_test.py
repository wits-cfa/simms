from simms.telescope.teldef import Antenna, ArrayTelescope, Observation
from simms.config_spec import validate

print("Testing Antenna API")
line = Antenna(1.3, 4, 10.2, restfreq=1.42e9)
validate(line)
print()

#print("Test ArrayTelescope API")
#line = ArrayTelescope(1.3, 4, 10.2)
#validate(line)
#print()
#
#print("Test Observation API")
#cont = Observation(5.2, 10.6, [5.1,2.2,3.4,1])
#validate(cont)
#print()