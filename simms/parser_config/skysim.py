import os
import simms
from simms.parser_config.utils import load, load_sources, thisdir as skysimdir
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
import glob
from simms.utilities import CatalogueError, isnummber, ParameterError
from simms.skymodel.source_factory import singlegauss_1d, contspec
import numpy as np
from astropy.coordinates import Angle
from simms.skymodel.converters import *
from casacore.tables import table
from tqdm import tqdm

log = get_logger(BIN.skysim)

command = BIN.skysim
command = BIN.skysim
sources = load_sources(["library/sys_noise"])
thisdir = os.path.dirname(__file__)
config = load(command, use_sources=sources)

source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]



@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    cat = opts.catalogue
    ms = opts.ms
    map_path = opts.mapping
   
    if opts.mapping and opts.cat_species:
        raise ParameterError("Cannot use custom map and built-in map simultaneously")
    elif opts.mapping:
        map_path = opts.mapping
    else:
        map_path = f'{thisdir}/library/{opts.cat_species}.yaml'

        
    mapdata = OmegaConf.load(map_path)
    mapcols = OmegaConf.create({})
    column = opts.column
    delimiter = opts.cat_delim


    for key in mapdata:
        keymap = mapdata.get(key)
        if keymap:   
            mapcols[key] = (keymap.name, [], keymap.get("unit"))
        else:
            mapcols[key] = (key, [], None)

    with open(cat) as stdr:
        header = stdr.readline().strip()

        if not header.startswith("#format:"):
            raise CatalogueError("Catalogue needs to have a header starting with the string #format:")
        
        header = header.strip().replace("#format:", "").strip()
        header = header.split(delimiter)

        for line in stdr.readlines():
            if line.startswith("#"):
                continue
            rowdata = line.strip().split(delimiter)
            if len(rowdata) != len(header): #raise cat error
                raise CatalogueError("The number of elements in one or more rows does not equal the\
                                    number of expected elements based on the number of elements in the\
                                    header")
            for key in mapcols:
                catkey = mapcols[key][0]
                if catkey not in header:
                    continue
                
                index = header.index(catkey)
                value = rowdata[index]
                if isnummber(value) and mapcols[key][2]:
                    value += mapcols[key][2]
                
                mapcols[key][1].append(value)
                
    tab = table(ms, readonly=False) 
    data = tab.getcol(column)
    uvw = tab.getcol("UVW")
    fldtab = table(f"{ms}::FIELD") 
    radec0 = fldtab.getcol("PHASE_DIR")
    ra0= radec0[0,0][0] 
    dec0= radec0[0,0][1]
    nrow = tab.nrows()
    spw_tab = table(f"{ms}::SPECTRAL_WINDOW")
    freqs = spw_tab.getcol("CHAN_FREQ")[0]
    nrows, nchan, ncorr = data.shape

    class Source:
        def __init__(self, name, ra, dec):
            self.name = name
            self.ra = convertra2rad(ra)
            self.dec = convertdec2rad(dec)
            self.spectrum = None
            self.shape = None
            
        
        def radec2lm(self, ra0, dec0):
            dra = self.ra - ra0
            self.l = np.cos(self.dec) * np.sin(dra) 
            self.m = np.sin(self.dec) * np.cos(dec0) - np.cos(self.dec) * np.sin(dec0) * np.cos(dra)
        
            return self.l, self.m

    class Shape:
            def __init__(self, emaj, emin, pa):
                self.emaj = convert2rad(emaj)
                self.emin = convert2rad(emin)
                self.pa = convert2rad(pa)
                

            def set_shape(self):
                if self.emaj != 'null':
                    pass
                else:
                    self.shape = 1

                
    class Spectrum: 
            def __init__(self, stokes_i, stokes_q, stokes_u, stokes_v, line_peak, line_width, line_restfreq, cont_reffreq, cont_coeff_1, cont_coeff_2):
                self.stokes_i = convert2Jy(stokes_i)
                self.stokes_q = convert2Jy(stokes_q)
                self.stokes_u = convert2Jy(stokes_u)
                self.stokes_v = convert2Jy(stokes_v)
                self.line_peak = convert2Hz(line_peak)
                self.line_width = convert2Hz(line_width)
                self.line_restfreq = convert2Hz(line_restfreq)
                self.cont_reffreq = convert2Hz(cont_reffreq)
                self.cont_coeff_1 = convert2float(cont_coeff_1, null_value=0)
                self.cont_coeff_2 = convert2float(cont_coeff_2, null_value=0)
# we need to add a spectra for a polarized light too. 
            def set_spectrum(self,freqs):
                if self.line_width not in [None, "null"]:
                    self.spectrum = singlegauss_1d(freqs, self.stokes_i, self.line_width, self.line_peak)
                elif self.cont_reffreq not in [None, "null"]:
                    self.spectrum = contspec(freqs, self.stokes_i, self.cont_coeff_1, self.cont_reffreq)
                else:
                    self.spectrum = self.stokes_i
                return self.spectrum

    def makesources(data,freqs):
        num_sources = len(data['name'][1])
        
        sources = []
        for i in range(num_sources):
            source = Source(
                name = data['name'][1][i],
                ra = data['ra'][1][i],
                dec = data['dec'][1][i],
                
            )
            #source.set_spectrum(freqs)
            spectrum = Spectrum(
                stokes_i = data['stokes_i'][1][i],
                stokes_q = data['stokes_q'][1][i] if i < len(data['stokes_q'][1]) else None,
                stokes_u = data['stokes_u'][1][i] if i < len(data['stokes_u'][1]) else None,
                stokes_v = data['stokes_v'][1][i] if i < len(data['stokes_v'][1]) else None,
                cont_reffreq = data['cont_reffreq'][1][i] if i < len(data['cont_reffreq'][1]) else None,
                line_peak = data['line_peak'][1][i] if i < len(data['line_peak'][1]) else None,
                line_width = data['line_width'][1][i] if i < len(data['line_width'][1]) else None,
                line_restfreq = data['line_restfreq'][1][i] if i < len(data['line_restfreq'][1]) else None,
                cont_coeff_1 = (data['cont_coeff_1'][1][i]) if i < len(data['cont_coeff_1'][1]) else None,
                cont_coeff_2 = data['cont_coeff_2'][1][i] if i < len(data['cont_coeff_2'][1]) else None,
                
            )

            shape = Shape(
                emaj = data['emaj'][1][i] if i < len(data['emaj'][1]) else None,
                emin = data['emin'][1][i] if i < len(data['emin'][1]) else None,
                pa = data['pa'][1][i] if i < len(data['pa'][1]) else None
                
            )
            source.l, source.m = source.radec2lm(ra0,dec0)
            source.spectrum = spectrum.set_spectrum(freqs)
            source.shape = shape.set_shape()
            sources.append(source)
        return sources

    sources = makesources(mapcols,freqs)
    
    def computevis(srcs, uvw, nchan):
        wavs = 2.99e8 / freqs[:nchan]
        uvw_scaled = uvw[:, None] / wavs 
        vis = 0j
        for source in srcs:
            l, m = source.l, source.m
            n_term = np.sqrt(1 - l*l - m*m) - 1
            arg = uvw_scaled[0] * l + uvw_scaled[1] * m + uvw_scaled[2] * n_term
            vis += source.spectrum * np.exp(2 * np.pi * 1j * arg)
        
        return vis
    

    with tqdm(total=nrow, desc='computing visibilities', unit='rows') as pbar:
        print(f'computing visibilities for {nrows} rows ')
        vischan = np.zeros_like(data)
        for row in range(nrows):
            vischan[row,:,0] = computevis(sources, uvw[row], nchan)  #adding the visibilites to the first diagonal
            vischan[row,:,3] = vischan[row,:,0] #adding the visibilities to the fourth diagonal
            pbar.update(1)

    nrow = tab.nrows()
    with tqdm(total=nrow, desc='simulating', unit='rows') as pbar2:
        print("Starting to add rows to ms file")
        for i in range(nrow):
            if opts.mode == "add":
                datai = tab.getcell(column, i) + vischan[i]
            elif opts.mode == "subtract":
                datai = tab.getcell(column, i) + vischan[i]
            else:
                datai = vischan[i]

            tab.putcell(column, i, datai)
            pbar2.update(1)
        tab.close()
