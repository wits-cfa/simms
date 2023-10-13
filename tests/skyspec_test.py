from simms.skymodel.skydef import Line, Cont, Extendedsource, Pointsource
from simms.config_spec import validate
from simms.skymodel.source_factory import singlegauss_2d
import matplotlib.pyplot as plt
import numpy as np

print("Testing base API")
line = Line(1.3, 4, restfreq=1.42e9)
validate(line)
print()

print("Test for case where optional parameters are not set")
line = Line(1.3, 4, 10.2)
validate(line)
print()

print("Test API with a list value")
cont = Cont(5.2, coeffs=[5.1,2.2,3.4,1])
validate(cont)
print()

print("Test API with default value not set")
cont = Cont(5.2)
validate(cont)
print()

print("Test API with List value given as a single value")
cont = Cont(5.2, 0.7)
validate(cont)

print("Test base API")
extsrc = Extendedsource([10, 0, 0, 0 ], 53, -28.1, 5, 2)
validate(extsrc)

print("Test API with single value given for stokes")
extsrc = Extendedsource(10, 53, -28.1, 5, 2)
validate(extsrc)

print("Test base API")
ptsrc = Pointsource([10, 0, 0, 0 ], 53, -28.1)
validate(ptsrc)

print("Test API with single value given for stokes")
ptsrc = Pointsource(10, 53, -28.1)
validate(ptsrc)

print("Testing gaussian with positive angle with central coords (0, 0)")
x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
gauss2d = singlegauss_2d(x, y, 10, 0, 0, 4, 2, 225)

plt.figure()
plt.imshow(gauss2d, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis')
plt.colorbar(label='flux')
plt.title('2D Gaussian with Elliptical Cross-Section')
plt.xlabel('X')
plt.ylabel('Y')
#plt.savefig('tests/testimage.png')
