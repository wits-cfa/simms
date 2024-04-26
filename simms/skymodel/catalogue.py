import os
from simms.skymodel.skydef import Catalogue
from scabha.basetypes import File
from simms import BIN, get_logger

log = get_logger(BIN.skysim)

def simulate_catalogue(catalogue_file: File, delimiter: str = " "):
        catalogue = Catalogue(path=catalogue_file, delimiter=delimiter)
        catalogue.readfromfile()

        for index, source_data in enumerate(catalogue.sources, start=0):
            log.info(f"Source {index+1}:")
            for key, value in source_data.items():
                log.info(f"  {key}: {value}")

#simulate_catalogue()
     
