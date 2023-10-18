import numpy as np
from simms.constants import PI, C


class Source:
    def __init__(self, source, spectrum, tempo=None, ra0=None, dec0=None):
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
        return np.sin(self.source.dec) * np.cos(self.dec0) -\
              np.cos(self.source.dec) * np.sin(self.dec0) * np.cos(self.dra)

    def get_spectrum(self, nchan):
        
        return self.spectrum.spectrum(nchan)

    def chan_to_freq(self, chan,f0,df):
        freqs = f0 + chan*df
        return freqs/1e6 ## returns frequency in MHz

    def z_to_chan(self, z, nustart, nu0, chanwidth): #want to find the channel of the peak
        nu = nu0/ (1+z) #nu0 is the frequency where the peak is i.e 1420 MHz 
        chan = (nu-nustart)/chanwidth # nustart is the start of the cube i.e. 1304 MHz
        return round(chan)

    def computevis(self, srcs, uvw, freqs):
    
        wavs = C / freqs
        nchan = len(freqs)
        uvw_scaled = uvw[:, None] / wavs 
    
        vis = 0j
    
        for src in srcs:
            l, m = src.radec2lm(self.ra0, self.dec0)
            n_term = np.sqrt(1 - l*l - m*m) - 1
            arg = uvw_scaled[0] * l + uvw_scaled[1] * m + uvw_scaled[2] * n_term
            vis += src.spectrum(nchan) * np.exp(2 * PI * 1j * arg)
    
        return vis