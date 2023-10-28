from dataclasses import dataclass
from simms.config_spec import SpecBase
from typing import List, Optional, Any
from simms import SCHEMADIR
import os
from simms.utilities import readyaml, ObjDict, File
from simms.skymodel.source_factory import singlegauss_1d
import numpy as np
from .skysims import Source


@dataclass
class Line(SpecBase):
    peak: float
    width: int
    restfreq: float
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Line"

    #@property think this should be part of the function
    #def sigma(self):
        #return self.width / FWHM_E

    def set_linespectrum(self, nchan):
        self.chans = np.arange(nchan)
        self.set_spectrum = singlegauss_1d(self.chans, self.stokes, self.sigma, self.freq_peak)
        return self.set_spectrum

    
@dataclass
class Cont(SpecBase):
    ref_freq: float
    coeffs: Optional[List[float]] = None
    schema_section: str = "Cont"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")

    def set_contspectrum(self, nchan):
        self.chans = np.arange(nchan)
        self.set_spectrum = singlegauss_1d(self.chans, self.stokes, self.sigma, self.freq_peak)
        return self.set_spectrum

@dataclass
class Pointsource(SpecBase):
   stokes: List[float]
   ra: float
   dec: float
   schema_section: str = "Pointsource"
   schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")

   def set_sourcetype(self, source_type):
        self.source_type = source_type
        return self.source_type

@dataclass
class Extendedsource(SpecBase):
    stokes: List[float]
    ra: float
    dec: float
    majoraxis: float
    minoraxis: float
    schema_section: str = "Extendedsource"
    schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")

    ##### we didnt define set_sourcetype for extended, we might need it
   def set_sourcetype(self, source_type)
       self.source_type = source_type
       return self.source_type


class Catalogue(SpecBase):
    def __init__(self, path: Optional[File] = None, sources:Optional[List[Any]] = None,
                  mapping:Optional[File] = None, delimeter: Optional[str] = " "):
        """
        Catalogue definition
        """
        self.path = path
        self.sources = None
        self.mapping = mapping
        self.delimeter = delimeter

        if path is None and sources is None:
            raise RuntimeError("Either one of the path or a source list"
                           " is required to initialise a Catalog instance")

        self.schema_section: str = "Catalogue"
        self.schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")

        self.__set_colnames()

        if path:
            self.__readfromfile()

    def __set_colnames(self):
        mapdict = readyaml(self.mapping)
        colnames = {}
        
        for key,val in mapdict.items():
            # set to custom name if provided, else leave as is
            self.colnames.update[key] = val or key
        # convert dictionary to object
        self.colname = ObjDict(colnames)

    @property
    def colnames(self):
        return list(self.colname.values())
    
    def isline(self, src):
        try:
            gotpeak = src[self.colname.line_peak]
        except ValueError:
            return False
        else:
            return bool(gotpeak)

    def ispoint(self, src):
        shape = 0
        try:
            shape += src[self.colname.emaj]
        except:
            pass

        try:
            shape += src[self.colname.emin]
        except:
            pass

        if shape>0:
            return False
        else:
            return True
    
    def get_stokes(self, src):
        stokes = {}
        for key in self.colname:
            if key.startswith("stokes_"):
                stokes[key] = src[self.colname[key]]
        return stokes

    def get_cont_coeffs(self, src):
        coeffs = {}
        for key in self.colname:
            if key.startswith("cont_coeff_"):
                coeffs[key] = src[self.colname[key]]
        return coeffs

    def __readfromfile(self):
        srcs = np.genfromtxt(self.path, names=True, dtype=float)
        for src in srcs:
            specdict = {}
            srcdict = {}
            srcdict.update(self.get_stokes())
            if self.isline(src):
                for key in "peak width restfreq".split():
                    specdict[key] = src[getattr(self.colname, f"line_{key}")]
                spectrum = Line(**specdict)

            else:
                specdict.update(self.get_cont_coeffs())
                specdict["reffreq"] = src[self.colname.cont_reffreq]
                spectrum = Cont(**specdict)

            if self.ispoint(src):
                for key in "ra dec".split():
                    srcdict[key] = src[getattr(self.colname, key)]
                source = Pointsource(**srcdict)
                source_type = "POINT"
            else:
                for key in "emaj emin pa ra dec".split():
                    srcdict[key] = src[getattr(self.colname, key)]
                source = Extendedsource(**srcdict)
                source_type = "GAUSSIAN"
            self.sources.append(Source(source=source, spectrum=spectrum))
