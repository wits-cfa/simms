import numpy as np
from simms.constants import PI, C, FWHM


class Source:
    def __init__(self, source source_type, spectrum, tempo=None, ra0=None, dec0=None):
        self.source = source
        self.source_type = source_type
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

def addsky(uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype):
    
    fwhminv = 1.0 / FWHM
    gauss_scale = fwhminv * np.sqrt(2.0) * np.pi / C

    nrow, nchan, ncorr = data.shape

    nsrc = sources.shape[0]
    n1 = lm.dtype.type(1) #make sure that n1 is the same dtype as lm

    scaled_freq = frequency * frequency.dtype.type(gauss_scale) #multiply the frequency by gaussscale

    vis = np.zeros((nrow, nchan, ncorr), dtype=dtype)

     for s in range(nsrc):
        l = lm[s, 0]  # noqa
        m = lm[s, 1]
        n = np.sqrt(n1 - l*l - m*m) - n1

        if source_type[s] == "POINT":
            for r in range(nrow):
                u = uvw[r, 0]
                v = uvw[r, 1]
                w = uvw[r, 2]

                real_phase = (2*PI/C)*(u*l + v*m + w*n)

                for f in range(nchan):
                    p = real_phase * frequency[f]
                    re = np.cos(p) * spectrum[s, f]
                    im = np.sin(p) * spectrum[s, f]

                    vis[r, f, 0] += re + im*1j
        elif source_type[s] == "GAUSSIAN":
            emaj, emin, angle = gauss_shape[s]

            # Convert to l-projection, m-projection, ratio
            el = emaj * np.sin(angle)
            em = emaj * np.cos(angle)
            er = emin / (1.0 if emaj == 0.0 else emaj)

            for r in range(nrow):
                u = uvw[r, 0]
                v = uvw[r, 1]
                w = uvw[r, 2]

                # Compute phase term
                real_phase = (2*PI/C)*(u*l + v*m + w*n)

                # Gaussian shape term bits
                u1 = (u*em - v*el)*er
                v1 = u*el + v*em

                for f in range(nchan):
                    p = real_phase * frequency[f]
                    re = np.cos(p) * spectrum[s, f]
                    im = np.sin(p) * spectrum[s, f]

                    # Calculate gaussian shape component and multiply in
                    fu1 = u1 * scaled_freq[f]
                    fv1 = v1 * scaled_freq[f]
                    shape = np.exp(-(fu1 * fu1 + fv1 * fv1))
                    re *= shape
                    im *= shape

                    vis[r, f, 0] += re + im*1j
    
        else:
            raise ValueError("source_type must be "
                             "POINT or GAUSSIAN")

    return vis

vischan = np.zeros_like(data)
for row in range(nrow):
    vischan[row,:,0] =addsky(uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype)
    vischan[row,:,3] = vischan[row,:,0] 





