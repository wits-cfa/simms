#import matplotlib
import numpy as np
from casacore.tables import table
#import matplotlib.pylab as plt
#import astropy.io.fits as fitsio
#from astropy.wcs import WCS
import csv
import math
#import pandas
#import astropy.coordinates as coord
#from matplotlib.patches import Circle
#from astropy.coordinates import angular_separation
import yaml
import os
from  simms.utilities import ValidationError, ListSpec, BASE_TYPES, singlegauss
from simms.config_spec import  getvals
from simms.skymodel.skydef import Line, Cont, Pointsource, Extendedsource, Catalogue


ms = "/workspaces/simms/tests/msdir/mytestsKAT-7.ms" #This should be replaced with the actual ms file˜

tab = table(ms, readonly=True)
data = tab.getcol("DATA")
uvw = tab.getcol("UVW")
fldtab = table(f"{ms}::FIELD") 
radec0 = fldtab.getcol("PHASE_DIR")
pi = np.pi
ra0= radec0[0,0][0] 
dec0= radec0[0,0][1]
nrow = tab.nrows()
spw_tab = table(f"{ms}::SPECTRAL_WINDOW")
freqs = spw_tab.getcol("CHAN_FREQ")[0]
nrows, nchan, ncorr = data.shape

class Source:
    def __init__(self, source, ra0, dec0, spectrum, tempo=None):
        self.source = source
        self.spectrum = spectrum
        self.tempo = tempo
        self.ra0 = ra0
        self.dec0 = dec0
        self.dra = self.source.ra - self.ra0

    @property
    def l(self):
        return np.cos(self.source.dec) * np.sin(self.dra) 
    @property
    def m(self):
        return np.sin(self.source.dec) * np.cos(self.dec0) - np.cos(self.source.dec) * np.sin(self.dec0) * np.cos(self.dra)

    def get_spectrum(self, nchan):
        
        return self.spectrum.spectrum(nchan)

def chan_to_freq(chan,f0,df):
    freqs = f0 + chan*df
    return freqs/1e6 ## returns frequency in MHz

def z_to_chan(z, nustart, nu0, chanwidth): #want to find the channel of the peak
    nu = nu0/ (1+z) #nu0 is the frequency where the peak is i.e 1420 MHz 
    chan = (nu-nustart)/chanwidth # nustart is the start of the cube i.e. 1304 MHz
    return round(chan)

def deg_to_rad (deg):
    rad = deg * (np.pi/180)
    return rad

def computevis(srcs, uvw, nchan):
    
    wavs = 2.99e8 / freqs[:nchan]
    uvw_scaled = uvw[:, None] / wavs 
    
    vis = 0j
    
    for src in srcs:
        l, m = src.radec2lm(ra0, dec0)
        n_term = np.sqrt(1 - l*l - m*m) - 1
        arg = uvw_scaled[0] * l + uvw_scaled[1] * m + uvw_scaled[2] * n_term
        vis += src.spectrum(nchan) * np.exp(2 * np.pi * 1j * arg)
    
    return vis

line = Line(1.3, 4, 10.2)#we need to read this in somehow, maybe this needs to all be one big code?
pointsource = Pointsource(1, 53, -28.5 )


ra_vals = getvals(pointsource, 'ra')#make these into arrays and fill them up (it should be fine with 1. val)
dec_vals = getvals(pointsource, 'dec')
flux_vals = getvals(pointsource, 'stokes_i')
nu0_vals= getvals(line, 'freq_peak')
width_vals= getvals(line, 'width')

ras =[]
decs =[]
fluxs =[]
widths = []
#if ra_vals, dec_vals, flux_vals, widths are not singular floats 
for i in range (len(ra_vals)):
    ra = ra_vals[i]
    dec = dec_vals[i]
    ras.append(ra)
    decs.append(dec)

if len(flux_vals) == len(ra_vals):
    for i in range (len(flux_vals)):
        flux = flux_vals[i]
        fluxs.append(flux)
if len(flux_vals) == 1:
    fluxs =np.ones(len(ra_vals))*flux_vals    
elif len(flux_vals) != 1 and  len(flux_vals) != len(ra_vals) :
    raise ValidationError("The number of flux values given does not match the number of sources")

if len(width_vals) == len(ra_vals):
    for i in range (len(width_vals)):
        width = width_vals[i]
        widths.append(width)
if len(width) == 1:
    widths =np.ones(len(ra_vals))*width_vals    
elif len(width_vals) != 1 and  len(width) != len(ra_vals) :
    raise ValidationError("The number of width values given does not match the number of sources")

for ra, dec, flux, nu0, width in zip(ras, decs, fluxs, nu0s, widths):# NB! add nu_0 as a parameter for line!!!
    src = Source(ra, dec, flux, nu0, width)
    srcs.append(src)

vischan = np.zeros_like(data)
for row in range(nrow):
    vischan[row,:,0] = computevis(srcs, uvw[row], nchan)  #adding the visibilites to the first diagonal
    vischan[row,:,3] = vischan[row,:,0] #adding the visibilities to the fourth diagonal

