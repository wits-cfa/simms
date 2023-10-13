from dataclasses import dataclass
from simms.config_spec import SpecBase
from typing import List, Optional
from simms import SCHEMADIR
import os
from simms.utilities import readyaml, get_class_attributes, ValidationError
from simms.skymodel.source_factory import singlegauss_1d
from simms.constants import FWHM_E, C, PI
import numpy as np


@dataclass
class Line(SpecBase):
    freq_peak: float
    width: int
    restfreq: float
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")
    schema_section: str = "Line"

    @property
    def sigma(self):
        return self.width / FWHM_E

    def spectrum(self, nchan):
        self.chans = np.arange(nchan)
        return singlegauss_1d(self.chans, self.stokes, self.sigma, self.freq_peak)

    
@dataclass
class Cont(SpecBase):
    ref_freq: float
    coeffs: Optional[List[float]] = None
    schema_section: str = "Cont"
    schemafile: str = os.path.join(SCHEMADIR, "schema_freq.yaml")

@dataclass
class Pointsource(SpecBase):
   stokes: List[float]
   ra: float
   dec: float
   schema_section: str = "Pointsource"
   schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")


@dataclass
class Extendedsource(SpecBase):
    stokes: List[float]
    ra: float
    dec: float
    majoraxis: float
    minoraxis: float
    schema_section: str = "Extendedsource"
    schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")

@dataclass
class Catalogue(SpecBase):
    cat: str
    racol: int 
    deccol: int 
    fluxcol: int 
    stokes: float
    schema_section: str = "Catalogue"
    schemafile: str = os.path.join(SCHEMADIR, "schema_source.yaml")
